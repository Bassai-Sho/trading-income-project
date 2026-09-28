"""pii_scan.py — the guard that keeps personal data out of the public repo. Every test calls
the REAL scanner. Strings that look like personal data are built at runtime from harmless
pieces, so this file never contains a real-looking address itself."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pii_scan

AT = "@"


def _rules(text, deny=()):
    return {f["rule"] for f in pii_scan.scan_text(text, "x", deny)}


# ── rules: positive ───────────────────────────────────────────────────────────

def test_flags_a_real_looking_email():
    assert "email" in _rules(f"EMAIL=someone{AT}mail-provider.com")


def test_flags_a_home_path_with_a_real_username():
    assert "home_path" in _rules("cd /home/alice/projects")


def test_flags_a_hostname_with_a_user_prefix():
    assert "user_hostname" in _rules("Node: bob-NUC14RVH-B, Intel")


def test_flags_private_lan_ips():
    for ip in ("192.168.1.20", "10.0.0.5", "172.16.4.9", "172.31.255.1"):
        assert "private_ip" in _rules(f"host={ip}"), ip


def test_flags_credential_shaped_literals():
    assert "secret_literal" in _rules('api_key = "abcd1234abcd1234abcd1234"')
    assert "secret_literal" in _rules("PASSWORD: 'correct-horse-battery-staple'")
    assert "secret_env_literal" in _rules("WEBUI_SECRET_KEY=trading-income-debug-key \\")
    assert "token_shape" in _rules("k = 'sk-" + "A" * 30 + "'")
    assert "token_shape" in _rules("-----BEGIN RSA PRIVATE KEY-----")


def test_flags_notion_links_and_workspace_ids():
    assert "notion_link" in _rules("see https://www.notion.so/" + "a" * 32)
    assert "notion_link" in _rules("collection://" + "b" * 8 + "-" + "c" * 12)


# ── rules: negative (no false alarms on ordinary content) ─────────────────────

def test_allows_placeholders_and_localhost():
    ok = ["your" + AT + "email.com", "git" + AT + "github.com", "someone" + AT + "example.org",
          "12345+x" + AT + "users.noreply.github.com"]
    for e in ok:
        assert _rules(f"contact {e}") == set(), e
    assert _rules("cd /home/user/project") == set()
    assert _rules("http://127.0.0.1:8000/v1  and 0.0.0.0") == set()
    assert _rules("Chrome/128.0.0.0 Safari/537.36") == set()


def test_allows_secrets_read_from_the_environment_or_placeholders():
    clean = ['api_key = os.environ["API_KEY"]', 'TOKEN=$(cat token.txt)', 'SECRET_KEY="${SECRET_KEY:-x}"',
             'ALPACA_API_KEY=your_key_here', 'password = "${PW}"', 'token = request.args["t"]']
    for line in clean:
        assert _rules(line) == set(), line


def test_short_or_ordinary_assignments_are_not_flagged():
    assert _rules('token = "abc"') == set()               # too short to be a credential
    assert _rules("Sharpe ratio 0.80, sharpe = 0.8") == set()


# ── denylist (local, untracked personal terms) ────────────────────────────────

def test_denylist_terms_are_enforced_case_insensitively():
    assert "denylist" in _rules("Lives near Exampletown", deny=("exampletown",))
    assert _rules("nothing here", deny=("exampletown",)) == set()


def test_findings_never_echo_the_full_sensitive_string():
    f = pii_scan.scan_text(f"EMAIL=verylongpersonalname{AT}mail-provider.com", "x")[0]
    assert "verylongpersonalname" not in f["snippet"] and len(f["snippet"]) < 12


def test_load_denylist_skips_comments_and_blanks(tmp_path):
    (tmp_path / ".pii_denylist").write_text("# comment\n\nalpha\n  beta  \n")
    assert pii_scan.load_denylist(tmp_path) == ("alpha", "beta")
    assert pii_scan.load_denylist(tmp_path / "nowhere") == ()


# ── real git behaviour on a temporary repo ────────────────────────────────────

@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.name", "t")
    run("config", "user.email", "t" + AT + "example.org")
    return tmp_path, run


def test_scan_repo_finds_a_planted_leak_in_a_tracked_file(repo):
    root, run = repo
    (root / "ok.py").write_text("x = 1\n")
    (root / "leak.sh").write_text(f"EMAIL=someone{AT}mail-provider.com\n")
    run("add", ".")
    found = pii_scan.scan_repo(root)
    assert [(f["path"], f["rule"]) for f in found] == [("leak.sh", "email")]


def test_staged_mode_scans_only_what_is_staged(repo):
    root, run = repo
    (root / "a.txt").write_text("clean\n")
    run("add", "a.txt")
    run("commit", "-q", "-m", "init")
    (root / "b.txt").write_text(f"me{AT}mail-provider.com\n")            # exists but NOT staged
    assert pii_scan.scan_repo(root, staged=True) == []
    run("add", "b.txt")
    assert [f["path"] for f in pii_scan.scan_repo(root, staged=True)] == ["b.txt"]


def test_staged_mode_reads_the_index_not_the_working_tree(repo):
    """A pre-commit hook must check what WILL BE COMMITTED. Make index and working tree
    disagree in both directions."""
    root, run = repo
    (root / "c.txt").write_text("clean\n")
    run("add", "c.txt")
    run("commit", "-q", "-m", "init")
    # (a) staged content clean, working tree dirty: staged scan must be clean
    (root / "c.txt").write_text(f"me{AT}mail-provider.com\n")
    assert pii_scan.scan_repo(root, staged=True) == []
    # (b) staged content dirty, working tree cleaned afterwards: staged scan must flag it
    run("add", "c.txt")
    (root / "c.txt").write_text("clean again\n")
    assert [f["rule"] for f in pii_scan.scan_repo(root, staged=True)] == ["email"]


def test_untracked_files_are_not_scanned_and_fixtures_are_skipped(repo):
    root, run = repo
    (root / "untracked.txt").write_text(f"me{AT}mail-provider.com\n")     # never added
    (root / "tests" / "fixtures").mkdir(parents=True)
    (root / "tests" / "fixtures" / "f.txt").write_text(f"me{AT}mail-provider.com\n")
    run("add", "tests")                                      # stage ONLY the fixture; untracked.txt stays out
    assert pii_scan.scan_repo(root) == []


def test_denylist_applies_to_tracked_files_of_a_repo(repo):
    root, run = repo
    (root / ".pii_denylist").write_text("secret-town\n")
    (root / "notes.md").write_text("I live in Secret-Town\n")
    run("add", "notes.md")                                   # the denylist itself stays untracked
    assert [(f["path"], f["rule"]) for f in pii_scan.scan_repo(root)] == [("notes.md", "denylist")]


# ── the actual repository ─────────────────────────────────────────────────────

def test_this_repository_has_no_findings():
    """The point of the whole file: the real tracked tree is clean. (Skipped when the
    tests are run from an export with no git metadata.)"""
    if shutil.which("git") is None or not (pii_scan.ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    found = pii_scan.scan_repo()
    assert found == [], "\n".join(f"{f['path']}:{f['line']} [{f['rule']}]" for f in found)


def test_the_exemption_list_is_exactly_the_scanners_own_two_files():
    """Anything added here escapes the scan, so growth must be a deliberate, reviewed change."""
    assert pii_scan.SELF_EXEMPT == {"src/pii_scan.py", "tests/test_pii_scan.py"}


def test_exempt_files_are_skipped_but_a_lookalike_path_is_not(repo):
    root, run = repo
    (root / "src").mkdir()
    (root / "src" / "pii_scan.py").write_text(f"x = 'me{AT}mail-provider.com'\n")        # exempt
    (root / "src" / "pii_scan_copy.py").write_text(f"x = 'me{AT}mail-provider.com'\n")   # NOT exempt
    run("add", ".")
    assert [f["path"] for f in pii_scan.scan_repo(root)] == ["src/pii_scan_copy.py"]


def test_local_denylist_is_gitignored():
    assert ".pii_denylist" in (pii_scan.ROOT / ".gitignore").read_text()
