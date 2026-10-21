"""pytest plugin: run upstream LightRAG's own tests against lightrag-ts (d11).

Loaded with `-p lightrag_ts_adapter` (tools/upstream/run-upstream-tests.ts does
it). On import it replaces the upstream functions lightrag-ts ports with
adapters that call the TS code through tools/upstream/ts_bridge.ts; everything
else (Tokenizer, exceptions, constants, mocks) stays upstream's, so the test
files run unchanged.

Selection comes from the manifest (LIGHTRAG_TS_MANIFEST): every collected test
must be listed as `ts` (runs against TS) or skipped with a reason, so a test
cannot quietly exercise upstream's Python instead. A `ts` test that never
reached the bridge fails.
"""

import builtins
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import lightrag.exceptions as lr_exceptions
import lightrag.operate as lr_operate
import lightrag.utils as lr_utils
import lightrag.utils_graph as lr_utils_graph

ROOT = Path(os.environ.get("LIGHTRAG_TS_ROOT", Path(__file__).resolve().parents[2]))
MANIFEST = json.loads(Path(os.environ["LIGHTRAG_TS_MANIFEST"]).read_text(encoding="utf-8"))

# --------------------------------------------------------------------------- bridge


class TsBridgeError(Exception):
    """A lightrag-ts error with no upstream counterpart."""


class Bridge:
    def __init__(self):
        self.proc = None
        self.handles = {}
        self.calls = 0

    def start(self):
        if self.proc is None:
            node = os.environ.get("LIGHTRAG_TS_NODE", "node")
            self.proc = subprocess.Popen(
                [node, str(ROOT / "tools/upstream/ts_bridge.ts")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        return self.proc

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
            self.proc = None

    # values ---------------------------------------------------------------
    def enc(self, v):
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, np.ndarray):  # TS gets a typed array (float32) or plain numbers
            return {"$nd": v.tolist(), "dtype": str(v.dtype)}
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [self.enc(x) for x in v]
        if isinstance(v, dict):
            if all(isinstance(k, str) for k in v):
                return {k: self.enc(x) for k, x in v.items()}
            raise TypeError("bridge: dict with non-str keys")
        hid = id(v)
        self.handles[hid] = v
        is_mock = isinstance(v, mock.NonCallableMock)
        if callable(v) and not is_mock:
            return {"$h": hid, "kind": "fn"}
        # a mock is both: callable (an LLM function) and an object (a storage)
        cls = type(v)
        if is_mock:
            # a mock stands for an object: its methods are the child mocks the
            # test configured (mock.query = AsyncMock(...)), its attributes the
            # plain values set on it (mock.global_config = {...})
            methods = [n for n, c in v._mock_children.items() if not n.startswith("_") and isinstance(c, mock.NonCallableMock)]
        else:
            methods = [n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n, None))]
        attrs = {}
        for k, x in vars(v).items() if hasattr(v, "__dict__") else ():
            if k.startswith("_") or k in methods:
                continue
            try:
                attrs[k] = self.enc(x)
            except TypeError:
                pass
        return {"$h": hid, "kind": "fn" if callable(v) else "obj", "methods": methods, "attrs": attrs}

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _recv(self):
        line = self.proc.stdout.readline()
        if not line:
            raise TsBridgeError("bridge: node exited")
        return json.loads(line)

    def _run_cb(self, cb):
        obj = self.handles[cb["target"]]
        fn = obj if cb["method"] is None else getattr(obj, cb["method"])
        return fn(*dec(cb["args"]), **dec(cb["kwargs"]))

    def _reply_cb(self, value=None, error=None):
        if error is not None:
            self._send({"cberr": {"type": type(error).__name__, "message": str(error)}})
        else:
            self._send({"cbok": self.enc(value)})

    def _request(self, name, params):
        self.start()
        self.calls += 1
        log = lambda level, msg: getattr(lr_operate.logger, level)(msg)  # noqa: E731 - resolved per call, so patch() sees it
        self._send({"call": name, "params": self.enc(params), "log": self.enc(log)})

    def call_sync(self, name, params):
        self._request(name, params)
        while True:
            msg = self._recv()
            if "cb" not in msg:
                return msg
            try:
                r = self._run_cb(msg["cb"])
                if inspect.isawaitable(r):
                    raise TsBridgeError("bridge: async callback from a sync call")
                self._reply_cb(r)
            except Exception as e:  # noqa: BLE001 - goes back to TS
                self._reply_cb(error=e)

    async def call_async(self, name, params):
        self._request(name, params)
        while True:
            msg = self._recv()
            if "cb" not in msg:
                return msg
            try:
                r = self._run_cb(msg["cb"])
                if inspect.isawaitable(r):
                    r = await r
                self._reply_cb(r)
            except Exception as e:  # noqa: BLE001
                self._reply_cb(error=e)


BRIDGE = Bridge()


def dec(v):
    """TS typed arrays ({"$nd": ..., "dtype"}) back to numpy, handles TS got from us ({"$ref"}) back to the object, the rest as is."""
    if isinstance(v, list):
        return [dec(x) for x in v]
    if isinstance(v, dict):
        if "$nd" in v:
            return np.array(v["$nd"], dtype=v.get("dtype") or None)
        if "$ref" in v and len(v) == 1:
            return BRIDGE.handles[v["$ref"]]
        return {k: dec(x) for k, x in v.items()}
    return v


def raise_ts_error(err):
    t, msg, fields = err["type"], err["message"], err.get("fields") or {}
    if t == "ChunkTokenLimitExceededError":
        e = lr_exceptions.ChunkTokenLimitExceededError(fields["chunkTokens"], fields["chunkTokenLimit"], None)
        # the attributes and the message are TS's, not recomputed here
        e.chunk_preview = fields.get("chunkPreview")
        e.args = (msg,)
        raise e
    if t == "PipelineCancelledException":
        raise lr_exceptions.PipelineCancelledException(msg)
    cls = getattr(builtins, t, None)
    if isinstance(cls, type) and issubclass(cls, Exception):
        raise cls(msg)
    raise TsBridgeError(f"{t}: {msg}")


def result(msg, mutable):
    if "err" in msg:
        raise_ts_error(msg["err"])
    for name, value in (msg.get("mut") or {}).items():
        target = mutable.get(name)
        if isinstance(target, dict) and isinstance(value, dict):
            for k, v in value.items():
                if isinstance(target.get(k), list) and isinstance(v, list):
                    target[k][:] = v
                else:
                    target[k] = v
    return msg["ok"]


# --------------------------------------------------------------------------- adapters


def bound(original, args, kwargs):
    b = inspect.signature(original).bind(*args, **kwargs)
    b.apply_defaults()
    return dict(b.arguments)


def sync_adapter(module, name, post=lambda v, p: v):
    original = getattr(module, name)

    def adapter(*args, **kwargs):
        p = bound(original, args, kwargs)
        return post(result(BRIDGE.call_sync(name, p), p), p)

    adapter.__wrapped_upstream__ = original
    setattr(module, name, adapter)


def async_adapter(module, name, post=lambda v, p: v, pre=lambda p: p):
    original = getattr(module, name)

    async def adapter(*args, **kwargs):
        p = bound(original, args, kwargs)
        sent = pre(dict(p))
        return post(result(await BRIDGE.call_async(name, sent), p), p)

    adapter.__wrapped_upstream__ = original
    setattr(module, name, adapter)


def _drop(*names):
    def pre(p):
        for n in names:
            p.pop(n, None)
        return p

    return pre


def _extract_post(value, p):
    return [({k: v for k, v in r["nodes"]}, {tuple(k): v for k, v in r["edges"]}) for r in value]


def _edges_post(value, p):
    if p.get("added_entities") is not None:
        p["added_entities"].extend(value["added"])
    return value["value"]


def _tuple(value, p):
    return tuple(value)


def _kg_search_post(value, p):
    value = dec(value)
    value["chunk_tracking"] = dict(value["chunk_tracking"])
    return value


sync_adapter(lr_operate, "chunking_by_token_size")
sync_adapter(lr_utils, "remove_think_tags")
sync_adapter(lr_operate, "_handle_single_entity_extraction")
sync_adapter(lr_operate, "_handle_single_relationship_extraction")
async_adapter(lr_operate, "extract_entities", _extract_post, _drop("pipeline_status_lock"))
async_adapter(lr_operate, "_handle_entity_relation_summary", _tuple)
async_adapter(lr_operate, "_merge_nodes_then_upsert", pre=_drop("pipeline_status_lock"))
async_adapter(lr_operate, "_merge_edges_then_upsert", _edges_post, _drop("pipeline_status_lock", "added_entities"))
async_adapter(lr_operate, "_perform_kg_search", _kg_search_post)

# utils_graph (B4). A test may monkeypatch the helpers these call by module
# attribute (utils_graph._edit_entity_impl, get_entity_info, ...): whatever is
# installed at call time and is not the adapter's own goes along as a hook, and
# TS calls it back where the Python code would have looked it up.
_GRAPH_HOOKS = ("_edit_entity_impl", "_merge_entities_impl", "get_entity_info", "get_relation_info")


def _with_graph_hooks(p):
    hooks = {n: getattr(lr_utils_graph, n) for n in _GRAPH_HOOKS if getattr(lr_utils_graph, n) is not _GRAPH_INSTALLED[n]}
    if hooks:
        p["__hooks"] = hooks
    return p


def _deletion_result(value, p):
    from lightrag.base import DeletionResult

    return DeletionResult(**value)


for _name in ("aedit_entity", "aedit_relation", "acreate_entity", "acreate_relation", "_merge_entities_impl", "amerge_entities"):
    async_adapter(lr_utils_graph, _name, pre=_with_graph_hooks)
for _name in ("adelete_by_entity", "adelete_by_relation"):
    async_adapter(lr_utils_graph, _name, _deletion_result, _with_graph_hooks)
_GRAPH_INSTALLED = {n: getattr(lr_utils_graph, n) for n in _GRAPH_HOOKS}

# --------------------------------------------------------------------------- selection


def _status(item):
    rel = Path(str(item.fspath)).name
    spec = MANIFEST["files"].get(f"tests/{rel}")
    if spec is None:
        return "skip: file not in the manifest"
    name = item.originalname or item.name
    if name in spec.get("ts", []):
        return "ts"
    if name in spec.get("skip", {}):
        return "skip: " + spec["skip"][name]
    return spec.get("default", "unlisted")


def pytest_collection_modifyitems(config, items):
    for item in items:
        status = _status(item)
        item.stash[STATUS] = status
        if status.startswith("skip"):
            item.add_marker(pytest.mark.skip(reason=status[len("skip: "):]))
        elif status != "ts":
            UNLISTED.append(item.nodeid)


UNLISTED = []


def pytest_collection_finish(session):
    if UNLISTED:
        pytest.exit("tests not in the manifest (mark them ts or skip with a reason): " + ", ".join(UNLISTED), returncode=4)


STATUS = pytest.StashKey[str]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    before = BRIDGE.calls
    res = yield
    if item.stash.get(STATUS, "") == "ts" and BRIDGE.calls == before:
        raise AssertionError("this test never reached lightrag-ts (no bridge call)")
    return res


def pytest_unconfigure(config):
    BRIDGE.close()


def pytest_report_header(config):
    return f"lightrag-ts adapter: {ROOT} (upstream {MANIFEST.get('upstream')}, {sys.executable})"
