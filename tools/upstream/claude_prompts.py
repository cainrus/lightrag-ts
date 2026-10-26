"""Record what the vault's LLM function hands the claude-complete bridge.

  python tools/upstream/claude_prompts.py <vault_scripts_dir> <fake_bin> <out.json>

Imports vault_rag_core (read only, from <vault_scripts_dir>; its _claude_call
module from CLAUDE_CALL_DIR when it lives elsewhere), points
_claude_call.CLI at <fake_bin> (test/fixtures/fake-claude-complete.ts, which
logs argv and stdin and never calls a model) and runs
_anthropic_complete_str on a few calls shaped like LightRAG's. Written per
call: the arguments, the bridge's argv and stdin, the returned string, or the
error. Nothing private: every string is made up here.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

CALLS = [
    dict(name="plain", prompt="Extract entities from: Alice met Bob.", kwargs={}),
    dict(
        name="system-history-lightrag-kwargs",
        prompt="---Role---\nAnswer.\n\n",
        kwargs=dict(
            system_prompt="  You are a helpful assistant. ☃\n",
            history_messages=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "second\nline"}, {"content": "no role"}, {"role": "system"}],
            hashing_kv=None,
            keyword_extraction=True,
            enable_cot=True,
            stream=True,
            max_tokens=4096,
            temperature=0.0,
            top_p=1,
            top_k=40,
            stop_sequences=["x"],
        ),
    ),
    dict(name="empty-system-timeout", prompt="p", kwargs=dict(system_prompt="", history_messages=[], timeout=42.7)),
    dict(name="unicode", prompt="Тест «кавычки» — émoji 🎉 \t tab", kwargs=dict(system_prompt="\tsys\t")),
    dict(name="fail", prompt="p", kwargs={}, mode="fail"),
    dict(name="limit", prompt="p", kwargs={}, mode="limit"),
]


async def main(scripts_dir, fake_bin, out):
    sys.path.insert(0, scripts_dir)
    import vault_rag_core as core  # noqa: E402

    if os.environ.get("CLAUDE_CALL_DIR"):
        sys.path.insert(0, os.environ["CLAUDE_CALL_DIR"])
    import _claude_call  # noqa: E402  same module object core imports

    _claude_call.CLI = Path(fake_bin)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "log.jsonl"
        os.environ["FAKE_CLAUDE_LOG"] = str(log)
        for c in CALLS:
            if log.exists():
                log.unlink()
            os.environ["FAKE_CLAUDE_MODE"] = c.get("mode", "reply")
            rec = {"name": c["name"], "prompt": c["prompt"], "kwargs": c["kwargs"], "mode": c.get("mode", "reply")}
            try:
                rec["result"] = await core._anthropic_complete_str(c["prompt"], **dict(c["kwargs"]))
            except Exception as e:  # noqa: BLE001
                rec["error"] = {"class": type(e).__name__, "message": str(e)}
            rec["bridge"] = [json.loads(line) for line in log.read_text().splitlines()]
            results.append(rec)
    Path(out).write_text(json.dumps({"source": "vault_rag_core._anthropic_complete_str", "calls": results}, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(results)} calls -> {out}")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:4]))
