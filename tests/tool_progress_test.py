"""
Behaviour test for tool_progress.py.  Runs without a browser or a Chainlit server:
Step.send / Step.update are replaced with recorders, while Step's real
contextvar-based nesting (parent_id assignment) still executes.

What this proves:  nesting, headline relabelling order, final summary, error
handling, output guarantees, and that a step created AFTER the block is not nested.
What it cannot prove: how the frontend draws it -- check that in the browser.

Run:  python tests/tool_progress_test.py
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # repo/src holds tool_progress.py

import chainlit as cl
from chainlit.step import Step

import tool_progress as tp

EVENTS: list[tuple] = []


async def fake_send(self):
    EVENTS.append(("send", self.id, self.parent_id, self.name, self.type))
    return self


async def fake_update(self):
    EVENTS.append(("update", self.id, self.parent_id, self.name, self.output))
    return True


async def scenario():
    async with tp.ToolProgress() as prog:
        parent_id = prog._parent.id
        async with prog.call("web_search", "SPY premarket gap", args={"query": "x"}) as c:
            await asyncio.sleep(0.01)
            c.result("results…")
        async with prog.call("fetch_url", "reuters.com", args={"url": "u"}) as c:
            c.result("article text")
        try:
            async with prog.call("fetch_url", "bloomberg.com") as c:
                raise RuntimeError("paywall")
        except RuntimeError:
            pass  # tool failure must not kill the whole run
        async with prog.call("web_search", "second query") as c:
            pass  # no result() -> must still be marked
        async with prog.call("fetch_url", "ft.com") as c:
            c.result("HTTP fetch failed")   # tool returns an error STRING, no exception
            c.fail()
    # a step made after the block must NOT be nested under the progress section
    after = cl.Step(name="answer")
    async with after:
        pass
    return parent_id, prog, after


def main():
    import sys
    step_mod = sys.modules["chainlit.step"]  # chainlit.step the attr is the decorator, not the module
    # Step.__init__ reads chainlit's session context for thread_id; give it a stub.
    stub = SimpleNamespace(session=SimpleNamespace(thread_id="t"), emitter=None)
    with patch.object(Step, "send", fake_send), \
         patch.object(Step, "update", fake_update), \
         patch.object(step_mod, "context", stub):
        parent_id, prog, after = asyncio.run(scenario())

    sends = [e for e in EVENTS if e[0] == "send"]
    kids = [e for e in sends if e[4] == "tool"]
    assert len(kids) == 5, kids
    assert all(k[2] == parent_id for k in kids), "every call must nest under the parent"
    assert after.parent_id is None, "step after the block must not be nested"

    # headline order == order of calls, then the summary
    headline = [e[3] for e in EVENTS if e[0] == "update" and e[1] == parent_id]
    print("Headline history:")
    for h in headline:
        print("   ", h)
    assert headline[0].startswith("⏳ Searching the web: SPY premarket gap")
    assert headline[1].startswith("⏳ Reading: reuters.com")
    assert headline[-1].startswith("✓ 2 searches · 3 pages read · 2 failed"), headline[-1]  # 2 of 3 fetches failed but 1 succeeded -> still ✓

    # child rows end with a status mark and always have output (=> collapsible)
    finals = {}
    for e in EVENTS:
        if e[0] == "update" and e[2] == parent_id:
            finals[e[1]] = (e[3], e[4])
    print("Child rows (final):")
    for name, out in finals.values():
        print("   ", name, "|", repr(out)[:40])
        assert out, "child must have output so the frontend renders an accordion"
    names = [n for n, _ in finals.values()]
    assert names[0].startswith("✓ web_search — SPY premarket gap")
    assert names[2].startswith("✗ fetch_url — bloomberg.com")
    assert names[3].startswith("✓ web_search — second query")
    assert names[4].startswith("✗ fetch_url — ft.com"), "c.fail() must mark the row without raising"
    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
