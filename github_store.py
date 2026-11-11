"""
Keeps your journal and views in a private GitHub repo, so the phone app
remembers them across reloads and devices, and the command line can use the
same files (clone the repo, `git pull`).

The app's URL is public, so the store is locked behind a passphrase. Both the
passphrase and the GitHub token live in Streamlit secrets, never in this repo:

    # .streamlit/secrets.toml (or the app's Settings -> Secrets on Streamlit Cloud)
    [storage]
    repo = "you/kelly-sizer-data"     # a PRIVATE repo
    token = "github_pat_..."          # fine-grained: that repo only, Contents read/write
    passphrase = "a long phrase only you know"
    branch = "main"                   # optional

Uses GitHub's contents API: one small file per save, every save a commit, so
the repo's history is a full record of every change.
"""
from __future__ import annotations

import base64
import hmac

import requests

API = "https://api.github.com"
TIMEOUT = 20
VIEWS_PATH = "views.csv"
JOURNAL_PATH = "journal.csv"


class Conflict(Exception):
    """The file changed since we last read it (another device saved first)."""


class StoreError(Exception):
    """GitHub refused or couldn't be reached; the message says why."""


class GitHubStore:
    def __init__(self, repo: str, token: str, branch: str = "main", http=None):
        self.repo, self.branch = repo.strip(), (branch or "main").strip()
        self.http = http or requests.Session()
        self.headers = {"Authorization": f"Bearer {token.strip()}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28"}

    def _url(self, path: str) -> str:
        return f"{API}/repos/{self.repo}/contents/{path}"

    def read(self, path: str) -> tuple[str | None, str | None]:
        """(text, sha) of a file, or (None, None) if it doesn't exist yet."""
        try:
            r = self.http.get(self._url(path), headers=self.headers,
                              params={"ref": self.branch}, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise StoreError(f"couldn't reach GitHub ({type(e).__name__})") from e
        if r.status_code == 404:
            return None, None
        _raise_for(r)
        body = r.json()
        return base64.b64decode(body["content"]).decode("utf-8"), body["sha"]

    def write(self, path: str, text: str, message: str, sha: str | None) -> str:
        """Create or replace a file; returns its new sha. sha must be the one
        last read (None for a new file), or Conflict is raised."""
        payload = {"message": message, "branch": self.branch,
                   "content": base64.b64encode(text.encode("utf-8")).decode("ascii")}
        if sha:
            payload["sha"] = sha
        try:
            r = self.http.put(self._url(path), headers=self.headers, json=payload,
                              timeout=TIMEOUT)
        except requests.RequestException as e:
            raise StoreError(f"couldn't reach GitHub ({type(e).__name__})") from e
        if r.status_code in (409, 422) and "sha" in r.text.lower():
            raise Conflict(path)
        _raise_for(r)
        return r.json()["content"]["sha"]


def _raise_for(r) -> None:
    if r.status_code < 400:
        return
    why = {401: "the token was rejected (expired or revoked?)",
           403: "the token can't write to that repo (needs Contents: read and write)",
           404: "repo not found (check the name, and that the token can see it)"}
    raise StoreError(why.get(r.status_code, f"GitHub answered {r.status_code}"))


def from_secrets(secrets) -> tuple[GitHubStore | None, str | None]:
    """(store, passphrase) from Streamlit secrets, or (None, None) when storage
    isn't set up. Missing or partial settings mean no storage, never an error."""
    try:
        cfg = secrets["storage"]
        repo, token, passphrase = cfg["repo"], cfg["token"], cfg["passphrase"]
    except Exception:
        return None, None
    if not (repo and token and passphrase):
        return None, None
    return GitHubStore(repo, token, cfg.get("branch", "main")), str(passphrase)


def check_passphrase(given: str, expected: str) -> bool:
    """Constant-time comparison, so response timing doesn't leak the passphrase."""
    return bool(given) and hmac.compare_digest(given.encode(), expected.encode())
