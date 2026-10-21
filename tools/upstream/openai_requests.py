"""Record what upstream's llm/openai.py sends and does, for test/llm-openai.test.ts.

Runs in the rag-venv (lightrag-hku 1.4.16, openai 2.36, tenacity 9.1):

  python tools/upstream/openai_requests.py <out.json>

Each scenario calls openai_complete_if_cache or openai_embed.func against a
local recording HTTP server that plays scripted responses (nothing leaves the
machine; the API key is fake). Recorded per scenario: every request (method,
path, headers, body bytes), every sleep the two retry layers asked for (the
SDK's own retries and tenacity's @retry; sleeps return at once, the SDK's
jitter is pinned with random() = 0.5), and the result or the exception.
"""

import asyncio
import base64
import json
import os
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["OPENAI_API_KEY"] = "sk-test-env-key"
os.environ.pop("OPENAI_API_BASE", None)
os.environ.pop("EMBEDDING_USE_BASE64", None)

import anyio  # noqa: E402
import openai._base_client as obc  # noqa: E402

import lightrag.llm.openai as lo  # noqa: E402

SLEEPS: list[float] = []


async def _record_sleep(seconds, *a, **k):
    if seconds > 0:  # zero sleeps are event-loop checkpoints, not retry waits
        SLEEPS.append(round(float(seconds), 6))


anyio.sleep = _record_sleep
asyncio.sleep = _record_sleep  # tenacity's asyncio sleep looks it up per call
obc.random = lambda: 0.5


class Server:
    def __init__(self):
        self.requests = []
        self.responses = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(n)
                outer.requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items() if k.lower() != "host"},
                        "body": body.decode("utf-8"),
                    }
                )
                r = outer.responses.pop(0) if len(outer.responses) > 1 else outer.responses[0]
                if r.get("close"):
                    self.close_connection = True
                    self.wfile.flush()
                    self.connection.shutdown(2)
                    return
                if "sse" in r:
                    payload = "".join(f"data: {e}\n\n" for e in r["sse"]).encode()
                    ctype = "text/event-stream"
                else:
                    payload = r["body"].encode()
                    ctype = "application/json"
                self.send_response(r.get("status", 200))
                self.send_header("content-type", ctype)
                for k, v in (r.get("headers") or {}).items():
                    self.send_header(k, v)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def chat(content, **message):
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1760000000,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content, **message}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )


def delta(content=None, **extra):
    d = {} if content is None else {"content": content}
    d.update(extra)
    return json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [{"index": 0, "delta": d, "finish_reason": None}]})


def emb(vectors, b64=True):
    data = []
    for i, v in enumerate(vectors):
        e = base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode() if b64 else v
        data.append({"object": "embedding", "index": i, "embedding": e})
    return json.dumps({"object": "list", "data": data, "model": "text-embedding-3-large", "usage": {"prompt_tokens": 4, "total_tokens": 4}})


QUOTA = json.dumps(
    {
        "error": {
            "message": "You exceeded your current quota, please check your plan and billing details.",
            "type": "insufficient_quota",
            "param": None,
            "code": "insufficient_quota",
        }
    }
)
BAD = json.dumps({"error": {"message": "Invalid 'messages'", "type": "invalid_request_error", "param": "messages", "code": None}})

KW = {"high_level_keywords": ["Тема", "a"], "low_level_keywords": ["b"]}

SCENARIOS = [
    dict(
        name="complete-basic",
        fn="complete",
        args=dict(
            model="gpt-4o-mini",
            prompt="Hello \"world\" ünïcode ☃\nline2",
            system_prompt="You are \\ a system\t",
            history_messages=[{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
            kwargs={"max_tokens": 4096, "temperature": 0.2, "hashing_kv": None},
        ),
        responses=[{"body": chat("Answer ✓")}],
    ),
    dict(
        name="complete-minimal-timeout",
        fn="complete",
        args=dict(model="m1", prompt="p", api_key="sk-explicit", timeout=30, kwargs={}),
        responses=[{"body": chat("x")}],
    ),
    dict(
        name="complete-keywords",
        fn="complete",
        args=dict(model="gpt-4o-mini", prompt="kw?", keyword_extraction=True, stream=True, kwargs={}),
        responses=[{"body": chat(' {"low_level_keywords": ["b"], "high_level_keywords": ["Тема", "a"]} ')}],
    ),
    dict(
        name="complete-stream",
        fn="complete",
        args=dict(model="gpt-4o-mini", prompt="s", stream=True, kwargs={}),
        responses=[
            {
                "sse": [
                    json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": []}),
                    delta("", role="assistant"),
                    delta("Hel"),
                    delta("lo \\u00e9"),
                    delta(None),
                    json.dumps({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}),
                    "[DONE]",
                ]
            }
        ],
    ),
    dict(
        name="complete-stream-cot",
        fn="complete",
        args=dict(model="r1", prompt="s", stream=True, enable_cot=True, kwargs={}),
        responses=[{"sse": [delta(None, reasoning_content="think "), delta(None, reasoning_content="more"), delta("done"), "[DONE]"]}],
    ),
    dict(
        name="complete-cot",
        fn="complete",
        args=dict(model="r1", prompt="c", enable_cot=True, kwargs={}),
        responses=[{"body": chat("", reasoning_content="why \\u0041")}],
    ),
    dict(
        name="complete-unicode-escape",
        fn="complete",
        args=dict(model="m", prompt="u", kwargs={}),
        responses=[{"body": chat("caf\\u00e9 and \\u263a")}],
    ),
    dict(
        name="complete-quota",
        fn="complete",
        args=dict(model="m", prompt="q", kwargs={}),
        responses=[{"status": 429, "body": QUOTA, "headers": {"x-request-id": "req_1"}}],
    ),
    dict(
        name="complete-bad-request",
        fn="complete",
        args=dict(model="m", prompt="b", kwargs={}),
        responses=[{"status": 400, "body": BAD}],
    ),
    dict(
        name="complete-500-then-ok",
        fn="complete",
        args=dict(model="m", prompt="r", kwargs={}),
        responses=[{"status": 500, "body": "upstream exploded"}, {"body": chat("ok")}],
    ),
    dict(
        name="complete-retry-after",
        fn="complete",
        args=dict(model="m", prompt="r", kwargs={}),
        responses=[{"status": 429, "body": "{}", "headers": {"retry-after-ms": "250"}}, {"status": 429, "body": "{}", "headers": {"retry-after": "2"}}, {"body": chat("ok")}],
    ),
    dict(
        name="complete-no-retry-header",
        fn="complete",
        args=dict(model="m", prompt="r", kwargs={}),
        responses=[{"status": 503, "body": "{}", "headers": {"x-should-retry": "false"}}],
    ),
    dict(
        name="complete-empty",
        fn="complete",
        args=dict(model="m", prompt="e", kwargs={}),
        responses=[{"body": chat("  ")}],
    ),
    dict(
        name="complete-connection",
        fn="complete",
        args=dict(model="m", prompt="c", kwargs={}),
        responses=[{"close": True}],
    ),
    dict(
        name="embed-vault",
        fn="embed",
        args=dict(texts=["alpha", "ünï ☃", ""], model="text-embedding-3-large", embedding_dim=4, max_token_size=8192),
        responses=[{"body": emb([[0.1, -0.2, 0.3, 1e-8], [1.5, 2.5, -3.5, 0.0], [0.0, 0.0, 0.0, 1.0]])}],
    ),
    dict(
        name="embed-truncate-prefix",
        fn="embed",
        args=dict(
            texts=["one two three four five six seven eight nine ten", "short"],
            model="text-embedding-3-small",
            max_token_size=5,
            context="query",
            query_prefix="search_query: ",
            document_prefix="doc: ",
            api_key="sk-explicit",
        ),
        responses=[{"body": emb([[1.0, 2.0], [3.0, 4.0]])}],
    ),
    dict(
        name="embed-float",
        fn="embed",
        base64=False,
        args=dict(texts=["x"], model="text-embedding-3-small", context="document", document_prefix="doc: "),
        responses=[{"body": emb([[0.25, 0.5]], b64=False)}],
    ),
    dict(
        name="embed-quota",
        fn="embed",
        args=dict(texts=["x"], model="text-embedding-3-large", embedding_dim=4),
        responses=[{"status": 429, "body": QUOTA}],
    ),
]


def err_info(e):
    out = {"class": type(e).__name__, "message": str(e)}
    inner = e
    if type(e).__name__ == "RetryError":
        inner = e.last_attempt.exception()
        out["last"] = {"class": type(inner).__name__, "message": str(inner)}
    for k in ("status_code", "code", "type", "param", "request_id"):
        if hasattr(inner, k):
            out[k] = getattr(inner, k)
    return out


async def run(sc, srv):
    srv.requests.clear()
    srv.responses[:] = sc["responses"]
    SLEEPS.clear()
    a = dict(sc["args"])
    lo.EMBEDDING_USE_BASE64 = sc.get("base64", True)
    out = {"name": sc["name"], "fn": sc["fn"], "args": json.loads(json.dumps(sc["args"])), "responses": sc["responses"]}
    if sc.get("base64") is False:
        out["use_base64"] = False
    try:
        if sc["fn"] == "complete":
            kw = dict(a.pop("kwargs"))
            r = await lo.openai_complete_if_cache(a.pop("model"), a.pop("prompt"), base_url=srv.url, **a, **kw)
            if hasattr(r, "__aiter__"):
                out["result"] = {"stream": [x async for x in r]}
            else:
                out["result"] = {"text": r}
        else:
            r = await lo.openai_embed.func(base_url=srv.url, **a)
            out["result"] = {"vectors": [[float(x) for x in row] for row in r]}
            assert str(r.dtype) == "float32"
    except Exception as e:  # noqa: BLE001
        out["error"] = err_info(e)
    out["requests"] = list(srv.requests)
    out["sleeps"] = list(SLEEPS)
    return out


async def main(path):
    srv = Server()
    results = []
    for sc in SCENARIOS:
        r = await run(sc, srv)
        if sc["name"] == "complete-connection":
            r["requests"] = len(r["requests"])
        results.append(r)
    doc = {
        "upstream": "lightrag-hku 1.4.16, openai 2.36.0, tenacity 9.1.4",
        "base_url": "<server>",
        "scenarios": results,
    }
    text = json.dumps(doc, ensure_ascii=False, indent=1).replace(srv.url, "<server>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"{len(results)} scenarios -> {path}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
