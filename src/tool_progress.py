"""
tool_progress.py -- collapsed, self-labelling tool-call display for Chainlit.

UX
    While tools run, ONE collapsed section is visible and its title tracks the
    latest call:        ⏳ Searching the web: "SPY premarket gap"
                        ⏳ Reading reuters.com
    When the last call finishes the title becomes a summary:
                        ✓ 2 searches · 3 pages read · 6.4s
    Open the section to see one row per call; open a row for its input/output.
    Nothing is expanded by default, so the answer is not buried under detail.

Verified against chainlit 2.12.0 (Step signature, contextvar nesting, the
frontend's accordion logic).  See tool_progress_test.py for the behaviour test.

Usage
    from tool_progress import ToolProgress

    async with ToolProgress() as progress:                 # before the tool loop
        async with progress.call("web_search", query, args={"query": query}) as c:
            result = await dispatch_tool("web_search", {"query": query})
            c.result(result)
        async with progress.call("fetch_url", url, args={"url": url}) as c:
            c.result(await dispatch_tool("fetch_url", {"url": url}))
    # leave the block BEFORE sending the answer, otherwise the answer message
    # is nested inside the section.
    await cl.Message(content=answer).send()
"""
from __future__ import annotations

import contextlib
import time
from collections import Counter
from typing import Any, AsyncIterator, Optional

import chainlit as cl

# tool name -> (title while running, singular noun, plural noun for the summary)
# Add your own tools here; unknown tools fall back to the raw tool name.
LABELS: dict[str, tuple[str, str, str]] = {
    "web_search": ("Searching the web", "search", "searches"),
    "fetch_url": ("Reading", "page read", "pages read"),
    "yfinance": ("Checking market data", "market-data check", "market-data checks"),
    "scout": ("Scout pre-filtering", "scout pass", "scout passes"),
}

MAX_OUTPUT_CHARS = 6000  # stored inside the collapsed row, not shown until opened


def _clip(text: Any, n: int) -> str:
    s = text if isinstance(text, str) else str(text)
    return s if len(s) <= n else s[:n] + f"\n… [{len(s) - n} more chars]"


class _Call:
    """Handle yielded by ToolProgress.call(); use .result() to attach output."""

    def __init__(self, step: cl.Step):
        self._step = step
        self.failed = False

    def fail(self) -> None:
        """Mark this call as failed WITHOUT raising (for tools that return error
        strings instead of exceptions).  Row shows ✗ and the summary shows ⚠."""
        self.failed = True

    def result(self, output: Any) -> None:
        self._step.output = _clip(output, MAX_OUTPUT_CHARS)

    def note(self, text: str) -> None:
        """Append a short progress line to the row's output (optional)."""
        self._step.output = (self._step.output + "\n" if self._step.output else "") + text


class ToolProgress:
    def __init__(self, title: str = "Working…"):
        # No default_open -> collapsed.  auto_collapse re-closes it if the user
        # opened it mid-run and the run then finishes.
        self._parent = cl.Step(
            name=f"⏳ {title}",
            type="run",
            default_open=False,
            auto_collapse=True,
            show_input=False,
        )
        self._counts: Counter[str] = Counter()
        self._failed_by_tool: Counter[str] = Counter()
        self._failed = 0
        self._t0 = 0.0

    # ---- parent lifecycle -------------------------------------------------
    async def __aenter__(self) -> "ToolProgress":
        self._t0 = time.monotonic()
        await self._parent.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        elapsed = time.monotonic() - self._t0
        self._parent.name = self._summary(elapsed, crashed=exc_type is not None)
        await self._parent.__aexit__(exc_type, exc, tb)

    # ---- one tool call ----------------------------------------------------
    @contextlib.asynccontextmanager
    async def call(
        self, tool: str, detail: str = "", args: Optional[dict] = None
    ) -> AsyncIterator[_Call]:
        running, _, _ = LABELS.get(tool, (tool, tool, tool))
        short = _clip(detail, 60).replace("\n", " ")
        label = f"{running}: {short}" if short else running

        # 1. headline follows the latest call
        self._parent.name = f"⏳ {label}"
        await self._parent.update()

        # 2. collapsed child row.  show_input="json" keeps args inside the row.
        child = cl.Step(
            name=f"⏳ {tool}" + (f" — {short}" if short else ""),
            type="tool",
            default_open=False,
            show_input="json",
        )
        if args is not None:
            child.input = args
        # A step with no input/output/children renders as a flat, non-collapsible
        # header in the Chainlit frontend, so guarantee it has some output.
        child.output = "running…"

        t0 = time.monotonic()
        failed = False
        try:
            async with child:
                handle = _Call(child)
                try:
                    yield handle
                except BaseException:
                    failed = True
                    raise
                finally:
                    failed = failed or handle.failed
                    if handle.failed:
                        child.is_error = True  # red styling in the UI
                    dt = time.monotonic() - t0
                    mark = "✗" if failed else "✓"
                    child.name = f"{mark} {tool}" + (f" — {short}" if short else "") + f" · {dt:.1f}s"
                    if child.output == "running…":
                        child.output = "(no output)"
        finally:
            self._counts[tool] += 1
            if failed:
                self._failed += 1
                self._failed_by_tool[tool] += 1

    # ---- summary ----------------------------------------------------------
    def _summary(self, elapsed: float, crashed: bool) -> str:
        if not self._counts:
            return "No tools used"
        parts = []
        for tool, n in self._counts.items():
            _, one, many = LABELS.get(tool, (tool, tool, tool))
            parts.append(f"{n} {one if n == 1 else many}")
        # ⚠ only when something is actually wrong: the run crashed, or EVERY call
        # of some tool failed.  A few skipped pages (paywalls, retries) still get ✓.
        all_failed = any(self._failed_by_tool[t] == n for t, n in self._counts.items() if n)
        icon = "⚠" if (crashed or all_failed) else "✓"
        tail = f" · {self._failed} failed" if self._failed else ""
        return f"{icon} {' · '.join(parts)}{tail} · {elapsed:.1f}s"
