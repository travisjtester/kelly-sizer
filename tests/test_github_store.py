"""The private-repo store, against a fake GitHub contents API."""
import base64
import json

import pytest

import github_store as gs


class Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeGitHub:
    """Just enough of the contents API: files keyed by path, sha bumps on write."""
    def __init__(self):
        self.files, self.calls = {}, []

    def get(self, url, headers, params, timeout):
        path = url.rsplit("/contents/", 1)[1]
        self.calls.append(("GET", path, headers["Authorization"]))
        if path not in self.files:
            return Resp(404, {"message": "Not Found"})
        text, sha = self.files[path]
        return Resp(200, {"content": base64.encodebytes(text.encode()).decode(), "sha": sha})

    def put(self, url, headers, json, timeout):
        path = url.rsplit("/contents/", 1)[1]
        self.calls.append(("PUT", path, json.get("sha")))
        current = self.files.get(path, (None, None))[1]
        if json.get("sha") != current:
            return Resp(409, {"message": f"{path} does not match sha"})
        sha = f"sha{len(self.calls)}"
        self.files[path] = (base64.b64decode(json["content"]).decode(), sha)
        return Resp(200, {"content": {"sha": sha}})


@pytest.fixture
def store():
    return gs.GitHubStore("me/kelly-data", " tok ", http=FakeGitHub())


def test_missing_file_reads_as_none(store):
    assert store.read("views.csv") == (None, None)


def test_write_then_read_round_trips_with_the_token(store):
    sha = store.write("views.csv", "ticker,p_bear\nNVDA,0.2\n", "update views", None)
    assert store.read("views.csv") == ("ticker,p_bear\nNVDA,0.2\n", sha)
    assert all(c[2] == "Bearer tok" for c in store.http.calls if c[0] == "GET")


def test_stale_sha_is_a_conflict_not_an_overwrite(store):
    first = store.write("journal.csv", "a\n", "one", None)
    store.write("journal.csv", "b\n", "two", first)              # another device saves
    with pytest.raises(gs.Conflict):
        store.write("journal.csv", "c\n", "three", first)         # we still hold the old sha
    assert store.read("journal.csv")[0] == "b\n"


@pytest.mark.parametrize("status, words", [(401, "rejected"), (403, "Contents: read and write"),
                                           (500, "500")])
def test_errors_say_what_to_fix(store, status, words):
    store.http.put = lambda *a, **k: Resp(status, {"message": "nope"})
    with pytest.raises(gs.StoreError, match=words):
        store.write("views.csv", "x", "m", None)


@pytest.mark.parametrize("secrets", [{}, {"storage": {"repo": "me/d", "token": ""}},
                                     {"storage": {"repo": "me/d", "token": "t"}}])
def test_no_or_partial_secrets_means_no_storage(secrets):
    assert gs.from_secrets(secrets) == (None, None)


def test_secrets_build_a_store_and_passphrase():
    store, pw = gs.from_secrets({"storage": {"repo": "me/d", "token": "t", "passphrase": "pp"}})
    assert store.repo == "me/d" and store.branch == "main" and pw == "pp"


def test_passphrase_check():
    assert gs.check_passphrase("pp", "pp")
    assert not gs.check_passphrase("pq", "pp") and not gs.check_passphrase("", "pp")
