import hashlib
import json
import threading
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from leatengine import cli, pull
from leatengine.gguf import GGUF
from tests.helpers import write_tiny_model


class Hub:
    """A Hugging Face of files on this machine: the API's tree of a repository, a page of two
    files at a time, and its files, each redirected to a storage host of its own, which takes
    ranges, as Hugging Face's does; and the requests each host had, with their headers."""

    def __init__(self, files: dict[str, bytes], lfs: bool = True):
        self.files, self.lfs = files, lfs
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.broken: set[str] = set()  # files the storage sends other bytes of
        self.hub, self.storage = self._server("hub"), self._server("storage")

    def _server(self, name: str) -> ThreadingHTTPServer:
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                hub.requests.append((name, self.path, dict(self.headers)))
                getattr(hub, f"_{name}")(self)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.hub.server_port}"

    def _hub(self, h: BaseHTTPRequestHandler) -> None:
        url = urllib.parse.urlsplit(h.path)
        if url.path == "/api/models/owner/repo/tree/main":
            page = int(urllib.parse.parse_qs(url.query).get("page", ["0"])[0])
            names = sorted(self.files)[2 * page : 2 * page + 2]
            entries = [self._entry(name) for name in names]
            h.send_response(200)
            if 2 * page + 2 < len(self.files):
                h.send_header("Link", f'<{self.endpoint}{url.path}?page={page + 1}>; rel="next"')
            body = json.dumps(entries).encode()
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
        elif url.path.startswith("/owner/repo/resolve/main/"):
            name = urllib.parse.unquote(url.path.removeprefix("/owner/repo/resolve/main/"))
            h.send_response(302)
            h.send_header("Location", f"http://127.0.0.1:{self.storage.server_port}/{name}")
            h.send_header("Content-Length", "0")
            h.end_headers()
        else:
            h.send_error(401 if url.path.startswith("/api/models/gated") else 404)

    def _entry(self, name: str) -> dict:
        data = self.files[name]
        if self.lfs:
            lfs = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}
            return {"type": "file", "path": name, "size": len(data), "lfs": lfs}
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        return {"type": "file", "path": name, "size": len(data), "oid": blob}

    def _storage(self, h: BaseHTTPRequestHandler) -> None:
        name = urllib.parse.unquote(h.path.lstrip("/"))
        data = self.files[name]
        if name in self.broken:
            data = bytes(len(data))
        start = int(h.headers.get("Range", "bytes=0-").removeprefix("bytes=").rstrip("-"))
        h.send_response(206 if start else 200)
        h.send_header("Content-Length", str(len(data) - start))
        h.end_headers()
        h.wfile.write(data[start:])

    def close(self) -> None:
        for server in (self.hub, self.storage):
            server.shutdown()
            server.server_close()


@pytest.fixture
def repo(tmp_path) -> dict[str, bytes]:
    # a repository's files: a model split in two parts in a folder of its quantization, as
    # Unsloth's are, another quantization, two projectors and a README
    made = tmp_path / "made"
    made.mkdir()
    write_tiny_model(made / "tiny-Q4_K_M.gguf", split=20)
    files = {f"Q4_K_M/{p.name}": p.read_bytes() for p in sorted(made.iterdir())}
    assert len(files) == 2
    return files | {"tiny-Q8_0.gguf": b"q8" * 100, "mmproj-F32.gguf": b"f32" * 10,
                    "mmproj-F16.gguf": b"f16" * 10, "README.md": b"# tiny"}  # fmt: skip


@pytest.fixture
def hub(repo, monkeypatch) -> Iterator[Hub]:
    hub = Hub(repo)
    monkeypatch.setenv("HF_ENDPOINT", hub.endpoint)
    monkeypatch.setenv("HF_HOME", "/nonexistent")  # no login's token
    monkeypatch.delenv("HF_TOKEN", raising=False)
    yield hub
    hub.close()


def test_pull(hub, repo, tmp_path):
    # a quantization's every part, the projector of f16, each checked, into OWNER/NAME; and the
    # model's first part, which run and serve find by its name
    models = tmp_path / "models"
    said: list[str] = []
    path = pull.pull("owner/repo", models, say=said.append)
    folder = models / "owner/repo"
    assert path == folder / "tiny-Q4_K_M-00001-of-00002.gguf"
    assert sorted(p.name for p in folder.iterdir()) == [
        "mmproj-F16.gguf", "tiny-Q4_K_M-00001-of-00002.gguf", "tiny-Q4_K_M-00002-of-00002.gguf"
    ]  # fmt: skip
    for name, data in repo.items():
        if (folder / name.rsplit("/", 1)[-1]).exists():
            assert (folder / name.rsplit("/", 1)[-1]).read_bytes() == data
    assert len(GGUF.open(path).parts) == 2 and all(s.endswith(", checked") for s in said)
    assert pull.local(models, "owner/repo") == pull.local(models, "owner/repo:Q4_K_M") == path
    assert pull.local(models, "owner/repo:Q8_0") is None
    # pulled again, checked where it is rather than downloaded
    hub.requests.clear()
    pull.pull("owner/repo", models, say=said.append)
    assert not [r for r in hub.requests if r[0] == "storage"]
    # another quantization by its word, or a file by its whole name, without its projector
    assert pull.pull("owner/repo:q8_0", models, mmproj=False) == folder / "tiny-Q8_0.gguf"


def test_resume(hub, repo, tmp_path):
    # a download resumes from what an interrupted one left, asking for the rest alone
    folder = tmp_path / "models/owner/repo"
    folder.mkdir(parents=True)
    data = repo["tiny-Q8_0.gguf"]
    (folder / "tiny-Q8_0.gguf.part").write_bytes(data[:77])
    pull.pull("owner/repo:Q8_0", tmp_path / "models", mmproj=False)
    ranges = [r[2].get("Range") for r in hub.requests if r[0] == "storage"]
    assert ranges == ["bytes=77-"] and (folder / "tiny-Q8_0.gguf").read_bytes() == data
    assert not (folder / "tiny-Q8_0.gguf.part").exists()


def test_whole_part(hub, repo, tmp_path):
    # a download cut off between its last byte and its rename is kept without asking again
    folder = tmp_path / "models/owner/repo"
    folder.mkdir(parents=True)
    (folder / "tiny-Q8_0.gguf.part").write_bytes(repo["tiny-Q8_0.gguf"])
    pull.pull("owner/repo:Q8_0", tmp_path / "models", mmproj=False)
    assert not [r for r in hub.requests if r[0] == "storage"]
    assert (folder / "tiny-Q8_0.gguf").read_bytes() == repo["tiny-Q8_0.gguf"]


def test_checked(hub, tmp_path):
    # a file whose bytes are not those Hugging Face lists is not kept
    hub.broken.add("tiny-Q8_0.gguf")
    with pytest.raises(ValueError, match="tiny-Q8_0.gguf: downloaded, but not the file"):
        pull.pull("owner/repo:Q8_0", tmp_path / "models", mmproj=False)
    assert not list((tmp_path / "models/owner/repo").iterdir())


def test_git_files(repo, tmp_path, monkeypatch):
    # a file Git holds rather than LFS, checked by its blob's hash
    hub = Hub(repo, lfs=False)
    monkeypatch.setenv("HF_ENDPOINT", hub.endpoint)
    try:
        path = pull.pull("owner/repo:Q8_0", tmp_path / "models", mmproj=False)
        assert path.read_bytes() == repo["tiny-Q8_0.gguf"]
    finally:
        hub.close()


def test_token(hub, tmp_path, monkeypatch):
    # a token goes to Hugging Face, not to the storage it redirects to
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    pull.pull("owner/repo:Q8_0", tmp_path / "models", mmproj=False)
    sent = {(host, "Authorization" in headers) for host, _, headers in hub.requests}
    assert sent == {("hub", True), ("storage", False)}


@pytest.mark.parametrize("spec, error", [
    ("owner/repo:Q5_K_M", "owner/repo has no GGUF of Q5_K_M; it has Q4_K_M, Q8_0"),
    ("owner/repo:tiny", "tiny names more than one model: tiny-Q4_K_M.gguf, tiny-Q8_0.gguf"),
    ("gated/repo", "the repository is gated or private: set HF_TOKEN"),
    ("nobody/repo", "there is no such repository or file"),
    ("not a name", "'not a name' is not OWNER/NAME or OWNER/NAME:QUANT"),
])  # fmt: skip
def test_refused(hub, tmp_path, spec, error):
    with pytest.raises(ValueError, match=error):
        pull.pull(spec, tmp_path / "models")


def test_parts_missing(repo, tmp_path, monkeypatch):
    # a split model of which the repository lacks a part
    hub = Hub({k: v for k, v in repo.items() if "-00002-of-" not in k})
    monkeypatch.setenv("HF_ENDPOINT", hub.endpoint)
    try:
        with pytest.raises(ValueError, match="a split GGUF of 2 parts lacks some"):
            pull.pull("owner/repo", tmp_path / "models")
    finally:
        hub.close()


def test_dry_run(hub, tmp_path):
    said: list[str] = []
    pull.pull("owner/repo", tmp_path / "models", dry_run=True, say=said.append)
    assert [s.split(":")[0] for s in said] == [
        "Q4_K_M/tiny-Q4_K_M-00001-of-00002.gguf", "Q4_K_M/tiny-Q4_K_M-00002-of-00002.gguf",
        "mmproj-F16.gguf"]  # fmt: skip
    assert not (tmp_path / "models").exists()


def test_cli(hub, tmp_path, monkeypatch, capsys):
    # pull, then a pulled model by its name, as run, serve, bench and perplexity take it
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    cli.main(["pull", "owner/repo", "--no-mmproj"])
    first = tmp_path / "data/leatengine/models/owner/repo/tiny-Q4_K_M-00001-of-00002.gguf"
    assert capsys.readouterr().out.endswith(f"{first}\nrun it: leatengine run owner/repo\n")
    assert cli._pulled(Path("owner/repo:Q4_K_M")) == first
    assert cli._pulled(Path(__file__)) == Path(__file__)
    with pytest.raises(SystemExit, match="owner/other is no file, nor a model pulled"):
        cli._pulled(Path("owner/other"))
    with pytest.raises(SystemExit, match="pull failed: owner/repo has no GGUF of Q5_K_M"):
        cli.main(["pull", "owner/repo:Q5_K_M"])
    # a split model a part of which is gone, as an interrupted pull leaves one
    first.with_name("tiny-Q4_K_M-00002-of-00002.gguf").unlink()
    with pytest.raises(SystemExit, match="a split GGUF of 2 parts lacks some; leatengine pull"):
        cli._pulled(Path("owner/repo"))
