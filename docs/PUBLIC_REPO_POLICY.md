# Public Repository Policy

*Written 28 Sep 2026. Code: `src/pii_scan.py`, `tests/test_pii_scan.py`.*

This repository is **public on purpose**: an assistant can read it directly, which makes
development faster. That is only safe while nothing personal or secret is ever committed.
The strategy code is a replication of a published paper and holds no data, so the *method*
is not secret. Everything below is about keeping the *person and the account* out of it.

## Never commit

* Personal figures: capital, drawdown tolerance, income targets, account balances.
  (These live in the private tracker, not here. The sanitised risk memo, with parameters and
  scenario tables instead of personal numbers, may be committed.)
* Email addresses, real usernames or home paths, machine hostnames with a user prefix,
  LAN/private IPs.
* API keys, tokens, passwords, private keys, broker account identifiers.
* `.env`, `DATA/`, databases, run artifacts. (Already gitignored.)
* Tracker (Notion) links or workspace IDs.

## How it is enforced

* `python src/pii_scan.py` scans every tracked file; `tests/test_pii_scan.py` runs it as part
  of the normal suite, so a slip fails the tests.
* Optional pre-commit hook (recommended). Create `.git/hooks/pre-commit`:
  ```bash
  #!/usr/bin/env bash
  exec .venv/bin/python src/pii_scan.py --staged
  ```
  then `chmod +x .git/hooks/pre-commit`.
* **Personal terms** (surname, town, employer, family names) are enforced through an
  *untracked* file, `.pii_denylist`, one term per line, `#` comments. It is gitignored so the
  terms themselves never enter the public repo, but the scanner still blocks them locally.

## Honest limits

* Only the **current tree** (or staged changes) is scanned. A one-off history scan on
  28 Sep 2026 found no API keys and one personal email in a debug script (now removed from the
  current files). **The history still contains it.** The repo was public before, so treat that
  email as already exposed. History was deliberately *not* rewritten: it would change every
  commit hash that the audit trail (tracker entries, test docstrings, reports) cites.
  If that trade-off ever changes, `git filter-repo --replace-text` plus a force-push is the
  route, and every cited hash would need remapping.
* Regex rules can miss things and can occasionally flag harmless text. Treat a finding as a
  prompt to look, and a clean scan as "no known pattern found", not proof of absence.
* Commit metadata is clean: all commits carry an empty author email.

## If the repo ever needs to be private again

Nothing here prevents it. An assistant without repo access can be given a snapshot with
`git bundle create ~/tic.bundle --all` (no credentials involved). Do **not** paste access
tokens into a chat; they persist in the transcript.
