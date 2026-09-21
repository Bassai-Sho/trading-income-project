"""
tests/test_scout_flag.py -- Scout is OFF by default (P2-098, option B) and switchable with SCOUT_ENABLED.

Why: in every live run examined (20-21 Sep 2026) the Scout call cost 1.6-16 s and never contributed to an answer.
Drives the real on_message with a fake model / tools / network and a recording emitter.

Run from anywhere:   python tests/test_scout_flag.py     (needs chainlit, openai, httpx)
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.chdir(tempfile.mkdtemp())                     # chainlit writes .chainlit/ into the cwd on import

import chainlit as cl                            # noqa: E402
import httpx                                     # noqa: E402
from chainlit.context import context_var, local_steps   # noqa: E402
import chainlit_app as app                       # noqa: E402

EVENTS: list = []
CALLS: list = []
POSTS: list = []


class Emitter:
    async def send_step(self, d):   EVENTS.append(("send", d["id"], d.get("name"), d.get("type")))
    async def update_step(self, d): EVENTS.append(("update", d["id"], d.get("name"), d.get("type")))
    def __getattr__(self, n):
        async def _noop(*a, **k): return None
        return _noop


class Ctx:
    session = SimpleNamespace(thread_id="t", id="s", chat_profile="Research")
    emitter = Emitter()
    @property
    def current_step(self):
        s = local_steps.get(); return s[-1] if s else None


class Sess(dict):
    def get(self, k, d=None): return dict.get(self, k, d)
    def set(self, k, v): self[k] = v


_tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="web_search", arguments=json.dumps({"query": "SPY price today"})))
_msg = SimpleNamespace(content="", tool_calls=[_tc], model_dump=lambda exclude_none=True: {"role": "assistant", "content": ""})
_resp = SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=_msg)])


class Completions:
    async def create(self, **kw): CALLS.append(kw); return _resp


class FakeHttp:
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, json=None):
        POSTS.append(url)
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "prose"}}]})


async def fake_tools(): return [{"type": "function", "function": {"name": "web_search"}}]
async def fake_run_tool(name, args):
    if name == "web_search":
        return "1. https://www.247wallst.com/a-1\n2. https://www.cnbc.com/2026/09/20/b-2.html"
    return "SPY closed at $761.69, up 0.4%, per " + args["url"]
async def fake_stream(stream, thinking, thinking_step_name="x"): return "", "ANSWER"

app._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
app.AsyncOpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
app._get_tools = fake_tools
app.run_tool = fake_run_tool
app.httpx = SimpleNamespace(AsyncClient=FakeHttp, Timeout=httpx.Timeout, ConnectError=httpx.ConnectError)
app._stream_thinking_then_answer = fake_stream
app._db_save_message = lambda *a, **k: None


def run(scout_enabled: bool):
    EVENTS.clear(); CALLS.clear(); POSTS.clear()
    app.SCOUT_ENABLED = scout_enabled
    sess = Sess(history=[], use_tools=True, thinking_mode=False, temperature=0.2, max_tokens=256, chat_profile="Research")

    async def go():
        context_var.set(Ctx())
        with patch.object(cl, "user_session", sess):
            await app.on_message(cl.Message(content="What is SPY trading at today?"))
    asyncio.run(go())
    rows = {e[1]: e[2] for e in EVENTS if e[0] == "update" and e[3] == "tool"}.values()
    synth = [c for c in CALLS if c.get("stream")][-1]["messages"][-1]["content"]
    return list(rows), list(POSTS), synth


def main() -> None:
    assert app.SCOUT_ENABLED is False or os.getenv("SCOUT_ENABLED") == "1", "default must be OFF"

    rows, posts, synth = run(False)
    assert not any("scout" in r for r in rows), f"Scout must not run when disabled: {rows}"
    assert posts == [], f"no request may be sent to port 8001 when disabled: {posts}"
    assert "$761.69" in synth, "the fetched article text must reach synthesis"
    assert "retrieved from the web" in synth and "pre-filtered by Scout" not in synth, "context label must not claim a Scout pass"

    rows, posts, synth = run(True)
    assert any("scout" in r for r in rows), f"Scout row expected when enabled: {rows}"
    assert len(posts) == 1 and posts[0].endswith(":8001/v1/chat/completions"), posts
    assert "$761.69" in synth, "ungrounded Scout output must still fall back to the raw context"
    print("PASS: Scout is skipped entirely when disabled (no row, no 8001 call, honest context label) and works when enabled")


if __name__ == "__main__":
    main()
