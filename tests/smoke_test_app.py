"""
tests/smoke_test_app.py -- drive the real on_message in src/chainlit_app.py end-to-end
with a recording emitter (real Chainlit Step/Message code paths; fake model, tools and
network).  Proves: ONE collapsed tool section, calls nested in order, headline relabels
per call, failed fetches marked, Scout row always has text, and the section closes
BEFORE the reasoning step and answer (which must not be nested in it).
Run from anywhere:  python tests/smoke_test_app.py     (needs chainlit, openai, httpx)
Does NOT prove browser rendering -- check that by eye.
"""
import asyncio, json, os, sys, tempfile
from types import SimpleNamespace
from unittest.mock import patch

from pathlib import Path
SRC = str(Path(__file__).resolve().parent.parent / "src")   # repo/src (chainlit_app.py + tool_progress.py)
sys.path[:0] = [SRC]
os.chdir(tempfile.mkdtemp())            # chainlit writes .chainlit/ into cwd

import chainlit as cl
from chainlit.context import context_var, local_steps
import chainlit_app as app              # the patched file, imported for real

EVENTS = []                              # (kind, id, parentId, name, type, output)

class Emitter:
    async def send_step(self, d):   EVENTS.append(("send",   d["id"], d.get("parentId"), d.get("name"), d.get("type"), d.get("output")))
    async def update_step(self, d): EVENTS.append(("update", d["id"], d.get("parentId"), d.get("name"), d.get("type"), d.get("output")))
    def __getattr__(self, n):
        async def _noop(*a, **k): return None
        return _noop

class Ctx:
    session = SimpleNamespace(thread_id="t1", id="s1", chat_profile="Research")
    emitter = Emitter()
    @property
    def current_step(self):
        s = local_steps.get(); return s[-1] if s else None

class FakeSession(dict):
    def get(self, k, d=None): return dict.get(self, k, d)
    def set(self, k, v): self[k] = v

sess = FakeSession(history=[], use_tools=True, thinking_mode=True, temperature=0.2,
                   max_tokens=256, chat_profile="Research", profile="Research", model_port="8000")

# ---- fakes -----------------------------------------------------------------
tc = SimpleNamespace(id="call1", function=SimpleNamespace(name="web_search",
                     arguments=json.dumps({"query": "SPY premarket news"})))
msg = SimpleNamespace(content="<think>plan</think>", tool_calls=[tc],
                      model_dump=lambda exclude_none=True: {"role": "assistant", "content": "x"})
resp = SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=msg)])

class FakeCompletions:
    async def create(self, **kw): return resp
app._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
class FakeOpenAI:
    def __init__(self, **kw): self.chat = SimpleNamespace(completions=FakeCompletions())
app.AsyncOpenAI = FakeOpenAI

async def fake_tools(): return [{"type": "function", "function": {"name": "web_search"}}]
app._get_tools = fake_tools

SEARCH = ("1. https://www.bbc.com/news/business/2026/09/20/spy-rally\n"
          "2. https://apnews.com/article/markets-stocks-abc123\n"
          "3. https://www.cnbc.com/2026/09/20/futures-flat.html\n"
          "4. https://www.theguardian.com/business/2026/sep/20/markets\n")
async def fake_run_tool(name, args):
    await asyncio.sleep(0)
    if name == "web_search": return SEARCH
    if "apnews" in args["url"]: return "HTTP fetch failed: 403"     # error STRING, no exception
    return "Article body for " + args["url"]
app.run_tool = fake_run_tool

class FakeHttp:
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, json=None):
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "Phi says prose, not xml <|im_end|>"}}]})
import httpx
app.httpx = SimpleNamespace(AsyncClient=FakeHttp, Timeout=httpx.Timeout, ConnectError=httpx.ConnectError)

async def fake_stream(stream, thinking, thinking_step_name="x"):
    async with cl.Step(name=thinking_step_name, type="run", show_input=False) as s:
        s.output = "reasoning text"
    return "reasoning text", "FINAL ANSWER"
app._stream_thinking_then_answer = fake_stream
app._db_save_message = lambda *a, **k: None

async def main():
    context_var.set(Ctx())
    with patch.object(cl, "user_session", sess):
        await app.on_message(cl.Message(content="what is spy doing today"))

asyncio.run(main())

# ---------------------------- assertions ------------------------------------
sends = [e for e in EVENTS if e[0] == "send"]
parents = [e for e in sends if e[4] == "run" and e[3].startswith("⏳")]
assert len(parents) == 1, f"expected ONE progress section, got {parents}"
pid = parents[0][1]
kids = [e for e in sends if e[2] == pid]
kid_names = [k[3] for k in kids]
print("Children created under the section:"); [print("   ", n) for n in kid_names]
assert len(kids) == 6, kids   # 1 search + 4 fetch attempts + 1 scout
tools = [k[3].split(" ")[1] for k in kids]
assert tools[0] == "web_search" and tools[-1] == "scout", tools
assert tools.count("fetch_url") == 4, tools    # bbc ok, ap FAIL, cnbc ok, guardian ok -> 3 successes needs 4 attempts

# headline history + final summary
headline = [e[3] for e in EVENTS if e[0] == "update" and e[1] == pid]
print("\nHeadline history:"); [print("   ", h) for h in headline]
assert headline[0].startswith("⏳ Searching the web: SPY premarket news")
assert "Reading: bbc.com" in headline[1]
assert headline[-1].startswith("✓ 1 search · 4 pages read · 1 scout pass · 1 failed"), headline[-1]

# final rows: ✓/✗ marks and non-empty output
final = {}
for e in EVENTS:
    if e[0] == "update" and e[2] == pid: final[e[1]] = (e[3], e[5])
print("\nFinal rows:"); [print("   ", n, "|", repr(o)[:55]) for n, o in final.values()]
assert any(n.startswith("✗ fetch_url — apnews.com") for n, _ in final.values())
assert all(o for _, o in final.values()), "every row needs output so the UI renders an accordion"
scout = [v for v in final.values() if "scout" in v[0]][0]
assert "not a valid XML dossier" in scout[1]

# ordering + nesting: reasoning step and answer come AFTER the section closed, NOT nested
order = [e for e in EVENTS if e[0] == "send"]
idx = lambda pred: next(i for i, e in enumerate(order) if pred(e))
i_last_kid = max(i for i, e in enumerate(order) if e[2] == pid)
i_reason  = idx(lambda e: e[3] == "💭 Reasoning")
i_answer  = idx(lambda e: e[4] == "assistant_message")
assert i_last_kid < i_reason < i_answer, (i_last_kid, i_reason, i_answer)
assert order[i_reason][2] is None, "thinking step must not be nested in the tool section"
assert order[i_answer][2] is None, "answer must not be nested in the tool section"
print("\nSMOKE TEST PASSED: 1 section, %d rows, section closed before reasoning + answer" % len(kids))
