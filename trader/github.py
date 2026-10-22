"""Commit files to a GitHub branch over the REST API: no git binary, one commit for many files.

The token is a fine-grained PAT with Contents read/write on the repo (``GITHUB_DATA_TOKEN``).
"""

from __future__ import annotations

import base64

import requests

API = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


class GitHubRepo:
    def __init__(self, repo: str, token: str, branch: str = "main", http=None):
        self.repo = repo
        self.branch = branch
        self._http = http or requests.Session()
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _call(self, method: str, path: str, *, json=None, accept: str | None = None, ok=(200, 201)):
        headers = {**self._headers, **({"Accept": accept} if accept else {})}
        response = self._http.request(method, f"{API}/repos/{self.repo}{path}", headers=headers, json=json, timeout=30)
        if response.status_code == 404 and method == "GET":
            return None
        if response.status_code not in ok:
            raise GitHubError(f"GitHub {method} {path} 실패 (HTTP {response.status_code}): {response.text[:200]}")
        return response

    def head(self) -> str:
        """The branch's latest commit sha."""
        response = self._call("GET", f"/git/ref/heads/{self.branch}")
        if response is None:
            raise GitHubError(f"{self.repo}에 {self.branch} 브랜치가 없습니다")
        return response.json()["object"]["sha"]

    def read(self, commit: str, path: str) -> bytes | None:
        """A file's bytes at ``commit``, or None when it does not exist there."""
        response = self._call("GET", f"/contents/{path}?ref={commit}", accept="application/vnd.github.raw+json")
        return None if response is None else response.content

    def commit(self, parent: str, files: dict[str, bytes], message: str) -> str:
        """One commit on top of ``parent`` with ``files`` replaced; moves the branch only as a fast-forward."""
        base_tree = self._call("GET", f"/git/commits/{parent}").json()["tree"]["sha"]
        tree = []
        for path, content in sorted(files.items()):
            blob = self._call("POST", "/git/blobs", json={
                "content": base64.b64encode(content).decode(), "encoding": "base64",
            }).json()["sha"]
            tree.append({"path": path, "mode": "100644", "type": "blob", "sha": blob})
        tree_sha = self._call("POST", "/git/trees", json={"base_tree": base_tree, "tree": tree}).json()["sha"]
        sha = self._call("POST", "/git/commits", json={
            "message": message, "tree": tree_sha, "parents": [parent],
        }).json()["sha"]
        self._call("PATCH", f"/git/refs/heads/{self.branch}", json={"sha": sha, "force": False})
        return sha
