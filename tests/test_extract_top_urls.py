"""
tests/test_extract_top_urls.py -- regression test for the URL picker in src/chainlit_app.py.

Bug fixed 21 Sep 2026: the picker derived each domain with netloc.lstrip("www."), which strips a
CHARACTER SET, not a prefix -- "www.wsj.com" became "sj.com", so the paywall skip-list entry
"wsj.com" never matched and WSJ links were fetched anyway (then quarantined after a 401/403).

Run from anywhere:   python tests/test_extract_top_urls.py     (needs chainlit, openai, httpx)
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.chdir(tempfile.mkdtemp())            # chainlit writes .chainlit/ into the cwd on import

import chainlit_app as app              # noqa: E402  (the real module)

SEARCH = "\n".join([
    "1. https://www.wsj.com/articles/markets-rally-as-fed-signals-pause-123456",
    "2. https://wsj.com/livecoverage/stock-market-today-dow-sp500-nasdaq-09-20-2026",
    "3. https://www.bbc.com/news/business/2026/09/20/markets-close-higher",
    "4. https://www.theguardian.com/business/2026/sep/20/ftse-closes-up",
])


def main() -> None:
    urls = app._extract_top_urls(SEARCH, n=6)
    print("picked:", *urls, sep="\n   ")
    assert not any("wsj.com" in u for u in urls), "paywalled wsj.com must be skipped (www. and bare forms)"
    assert any("bbc.com" in u for u in urls), "a normal news article must still be picked"
    assert any("theguardian.com" in u for u in urls), "a second normal domain must still be picked"
    print("\nPASS: wsj.com is skipped in both 'www.' and bare form; other articles are kept")


if __name__ == "__main__":
    main()
