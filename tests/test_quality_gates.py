"""
tests/test_quality_gates.py -- P2-096 guards in src/chainlit_app.py, using text from a real failing run.

Live run 21 Sep 2026, "What is SPY trading at today ...", Pair 4: two of the three "successfully read"
pages were a YouTube cookie wall and a TradingView script blob, and the Scout model (Qwen2.5-Coder-1.5B)
invented a "definition of spy trading" whose XML dossier then REPLACED the fetched text in the synthesis
prompt (server log: synthesis prompt_tokens=130 vs 466 for the Scout call) -> a nonsense answer.

Run from anywhere:   python tests/test_quality_gates.py     (needs chainlit, openai, httpx)
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.chdir(tempfile.mkdtemp())            # chainlit writes .chainlit/ into the cwd on import

import chainlit_app as app              # noqa: E402

YOUTUBE_WALL = ("* EnglishUnited Kingdom - Deutsch - Espanol - Francais - Italiano - All languages - Afrikaans - "
                "azerbaycan - bosanski - catala - Cestina - Cymraeg - Dansk - eesti - Before you continue to YouTube "
                "We use cookies and data, including IP addresses, to - Deliver and maintain Google services - "
                "Track outages and protect against spam, fraud and abuse - Measure audience engagement ...")
TRADINGVIEW_JS = ("window.initData = {}; Live stock, index, futures, Forex and Bitcoin charts on TradingView html, body "
                  "{ min-width: 320px; height: 100%; width: 100%; overflow: hidden; } window.locale = 'en'; "
                  "window.language = 'en'; window.initData = window.initData || {}; (()=>{\"use strict\";const t="
                  "/(?:^|;)\\s*theme=(dark|light)(?:;|$)|$/.exec(document.cookie)[1];if(t){document.documentElement"
                  ".classList.toggle(\"theme-dark\",\"dark\"===t)}})(); var environment = \"battle\"; "
                  "window.WS_HOST_PING_REQUIRED = true;")
TRADINGVIEW_PROSE = ("Compare with SPDR S&P 500 ETF Trust Fund info SPY Fund Summary The investment seeks to provide "
                     "investment results that, before expenses, generally correspond to the price and yield performance "
                     "of the S&P 500 Index. An exchange-traded fund (ETF) is a collection of assets that track an "
                     "underlying index and can be bought on an exchange like individual stocks.")
NEWS = ("Stocks ended higher on Friday as the S&P 500 gained 0.4% to close at 6,120.55, with analysts at several "
        "banks raising year-end targets. The SPDR S&P 500 ETF (SPY) rose to $761.69. Futures had climbed earlier "
        "ahead of the Trump-Xi summit, the document said. The window for a rate cut is narrowing, one strategist noted.")

STOCKSTOTRADE_WALL = ("We don't currently have information about State Street SPDR S&P 500 ETF Trust's earnings. The Game "
                      "is Rigged But Our AI-driven analysis Has Leveled the Playing Field Sign up for access to institutional "
                      "grade tools and insights and join 10,000+ traders Enter a valid email address. Enter a valid phone "
                      "number. I AGREE TO RECEIVE MARKETING By entering your info and clicking I AGREE below, you provide "
                      "your electronic signature, express written consent and binding agreement to our Terms of Use ...")

RAW = ("[Stocktwits] Futures rise ahead of the summit URL: https://stocktwits.com/news-articles/x "
       "[Yahoo Finance] https://finance.yahoo.com/quote/SPY/ latest SPY quote\n\nFetched articles:\n\n"
       "[https://www.247wallst.com/a-1]\nSPY closed at $761.69, up 0.4% on Friday.")

HALLUCINATED = ("<dossier>\n<key_facts>\n - Spy trading is a practice where traders buy and sell stocks without "
                "disclosing their identities or intentions. Analysts often warn against this practice, as it can lead "
                "to market manipulation and unfair advantage.\n - The stock market is expected to open up today, with "
                "the S&P 500 and Dow Futures rising ahead of the Trump-Xi summit.\n</key_facts>\n<gaps>None</gaps>\n</dossier>")
GROUNDED = ("<dossier>\n<key_facts>\n - SPY closed at $761.69, up 0.4% [247wallst.com]\n - Futures rose ahead of the "
            "summit [stocktwits.com]\n</key_facts>\n<gaps>NONE</gaps>\n</dossier>")


def main() -> None:
    # --- junk-page gate ----------------------------------------------------------
    assert app._looks_like_junk_page(YOUTUBE_WALL), "YouTube cookie wall must be junk"
    assert app._looks_like_junk_page(TRADINGVIEW_JS), "TradingView script blob must be junk"
    assert app._looks_like_junk_page(""), "empty page must be junk"
    assert app._looks_like_junk_page(STOCKSTOTRADE_WALL), "sign-up / marketing wall must be junk"
    assert not app._looks_like_junk_page(TRADINGVIEW_PROSE), "prose page must NOT be junk"
    assert not app._looks_like_junk_page(NEWS), "a news paragraph must NOT be junk (even with 'document.' / 'window.' in it)"

    # --- Scout dossier gate --------------------------------------------------------
    assert not app._scout_dossier_usable(HALLUCINATED, RAW), "ungrounded (hallucinated) dossier must be rejected"
    assert app._scout_dossier_usable(GROUNDED, RAW), "dossier repeating real figures + domains must be accepted"
    assert not app._scout_dossier_usable("Spy trading is a practice ...", RAW), "prose (no XML) must be rejected"
    assert not app._scout_dossier_usable(GROUNDED, "no figures or links here at all"), \
        "with nothing to verify against, do not trust the dossier"

    # --- quarantine pre-filter -------------------------------------------------------
    import types
    fake = types.ModuleType("domain_telemetry")
    fake.get_domain_strategy = lambda url: "SKIP" if ("stocktwits.com" in url or "benzinga.com" in url) else "OK"
    sys.modules["domain_telemetry"] = fake
    assert app._is_quarantined("https://stocktwits.com/news-articles/x")
    assert app._is_quarantined("https://www.benzinga.com/quote/SPY")
    assert not app._is_quarantined("https://www.cnbc.com/2026/09/20/markets.html")
    fake.get_domain_strategy = lambda url: (_ for _ in ()).throw(RuntimeError("db locked"))
    assert not app._is_quarantined("https://stocktwits.com/x"), "a telemetry failure must fail OPEN (fetch anyway)"
    del sys.modules["domain_telemetry"]
    print("PASS: cookie/JS/sign-up walls are junk, real prose is not; hallucinated dossier rejected, grounded accepted; "
          "quarantined domains pre-filtered (fails open)")


if __name__ == "__main__":
    main()
