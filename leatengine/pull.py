"""Models pulled from Hugging Face: a repository's GGUF of a quantization, every part of a split
one, and the vision projector beside it, as llama.cpp's -hf takes them, each file checked against
the SHA-256 Hugging Face gives it.

A repository and a quantization, OWNER/NAME:QUANT, name a model: QUANT a word of its files' names,
Q4_K_M by default, as llama.cpp's. Files go to MODELS/OWNER/NAME/, where `run` and `serve` find
them by the same name. A download resumes from what an interrupted one left, NAME.part. Gated
repositories need a token: HF_TOKEN, or huggingface-cli's login. HF_ENDPOINT stands in for
huggingface.co, as for Hugging Face's own tools.
"""

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leatengine.files import PART, model_name, projector

DEFAULT_QUANT = "Q4_K_M"
_NAME = re.compile(r"([\w.-]+)/([\w.-]+)(?::([\w.-]+))?")
# projectors by the precision llama.cpp's converter writes, the closest to the model's own first
_PRECISIONS = ("f16", "bf16", "f32", "q8_0")
_BLOCK = 8 << 20  # bytes read and hashed at a time


@dataclass(frozen=True)
class File:
    """A repository's file: its path there, its size, and its hash, SHA-256 of a file Git LFS
    holds, else Git's own of its blob."""

    path: str
    size: int
    sha256: str | None
    git_sha1: str | None

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


def parse(spec: str) -> tuple[str, str | None] | None:
    """A model's name, OWNER/NAME[:QUANT], as its repository and quantization, if given; None
    if it is no such name."""
    named = _NAME.fullmatch(spec)
    return (f"{named[1]}/{named[2]}", named[3]) if named else None


def directory(models: Path, repo: str) -> Path:
    return models / repo


def local(models: Path, spec: str) -> Path | None:
    """The model a name, OWNER/NAME[:QUANT], names among those pulled: the GGUF of its
    quantization, of a split one its first part; None if it names none pulled."""
    if (named := parse(spec)) is None or not (folder := directory(models, named[0])).is_dir():
        return None
    files = [p for p in sorted(folder.glob("*.gguf")) if not projector(p)]
    chosen = _chosen([p.name for p in files], named[1])
    return folder / _first(chosen) if chosen else None


def pull(
    spec: str, models: Path, mmproj: bool = True, dry_run: bool = False,
    say: Callable[[str], None] = print,
) -> Path:  # fmt: skip
    """Downloads a model, OWNER/NAME[:QUANT], into MODELS/OWNER/NAME/, its projector beside it if
    the repository has one and `mmproj`, each file checked against its hash; the model's
    path, of a split one its first part's. Files already there are checked rather than
    downloaded again. Of a dry run, what it would download, said."""
    if (named := parse(spec)) is None:
        raise ValueError(f"{spec!r} is not OWNER/NAME or OWNER/NAME:QUANT")
    repo, quant = named
    files = {f.name: f for f in _listed(repo) if f.path.endswith(".gguf")}
    models_found = [name for name in files if not projector(name)]
    chosen = _chosen(models_found, quant)
    if not chosen:
        quants = sorted({q for name in models_found if (q := _quant(name))})
        raise ValueError(f"{repo} has no GGUF of {quant or DEFAULT_QUANT}; it has "
                         f"{', '.join(quants) or 'none it names'}")  # fmt: skip
    wanted = [files[name] for name in chosen]
    if mmproj and (best := _best_projector([n for n in files if projector(n)])):
        wanted.append(files[best])
    folder = directory(models, repo)
    if dry_run:
        for f in wanted:
            say(f"{f.path}: {f.size / 2**30:.2f} GiB")
        return folder / _first(chosen)
    folder.mkdir(parents=True, exist_ok=True)
    for f in wanted:
        _fetched(repo, f, folder / f.name, say)
    return folder / _first(chosen)


def _chosen(names: list[str], quant: str | None) -> list[str]:
    # the model files of a quantization, every part of a split one: a word of their names, as
    # Q4_K_M of model-Q4_K_M.gguf or of Q4_K_M/model-Q4_K_M-00001-of-00002.gguf; a ValueError
    # where it names more than one model, or a split one with parts missing
    word = re.compile(rf"(^|[-_./]){re.escape(quant or DEFAULT_QUANT)}([-./]|$)", re.IGNORECASE)
    found = [name for name in names if word.search(name)]
    if quant is not None and not found and quant in names:  # a file's whole name
        found = [quant]
    models = sorted({f"{model_name(name)}.gguf" for name in found})
    if len(models) > 1:
        raise ValueError(f"{quant or DEFAULT_QUANT} names more than one model: "
                         f"{', '.join(models)}; give the file's name as QUANT")  # fmt: skip
    for name in found:
        if parts := PART.fullmatch(name):
            count = int(parts[3])
            have = {int(p[2]) for n in found if (p := PART.fullmatch(n))}
            if have != set(range(1, count + 1)):
                raise ValueError(f"{name}: a split GGUF of {count} parts lacks some")
            break
    return sorted(found)


def _first(chosen: list[str]) -> str:
    return next((n for n in chosen if (p := PART.fullmatch(n)) and int(p[2]) == 1), chosen[0])


def _quant(name: str) -> str | None:
    # the quantization a model file's name says, as the last word before .gguf or its part
    match = re.search(r"[-_.]((?:UD-)?(?:I?Q\d\w*|[BF]F?\d+|MXFP4))(?:-\d{5}-of-\d{5})?\.gguf$",
                      name, re.IGNORECASE)  # fmt: skip
    return match[1] if match else None


def _best_projector(names: list[str]) -> str | None:
    # of a repository's projectors, that of the first precision it has of _PRECISIONS, else any
    for precision in _PRECISIONS:
        for name in sorted(names):
            if re.search(rf"[-_.]{precision}\.gguf$", name, re.IGNORECASE):
                return name
    return min(names, default=None)


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _token() -> str | None:
    # a Hugging Face token: HF_TOKEN, else huggingface-cli's login
    if token := os.environ.get("HF_TOKEN"):
        return token
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface")
    try:
        return (home / "token").read_text().strip() or None
    except OSError:
        return None


class _Redirects(urllib.request.HTTPRedirectHandler):
    # redirects followed, the token sent only to Hugging Face's own host, not to the storage
    # another host serves files from
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        here = urllib.parse.urlsplit(_endpoint()).netloc
        if new is not None and urllib.parse.urlsplit(newurl).netloc != here:
            new.remove_header("Authorization")
        return new


def _open(url: str, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "leatengine"} | (headers or {}))
    if token := _token():
        request.add_header("Authorization", f"Bearer {token}")
    try:
        return urllib.request.build_opener(_Redirects).open(request, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise ValueError(f"{url}: {e.code}, the repository is gated or private: set HF_TOKEN "
                             "to a token of an account that may read it") from None  # fmt: skip
        if e.code == 404:
            raise ValueError(f"{url}: there is no such repository or file") from None
        raise


def _listed(repo: str) -> list[File]:
    # every file of a repository's main revision, page by page
    url: str | None = f"{_endpoint()}/api/models/{repo}/tree/main?recursive=true"
    files = []
    while url:
        with _open(url) as response:
            entries = json.load(response)
            url = _next(response.headers.get("Link", ""))
        for e in entries:
            if e.get("type") == "file":
                lfs = e.get("lfs") or {}
                files.append(File(e["path"], int(e.get("size", 0)), lfs.get("oid"),
                                  None if lfs else e.get("oid")))  # fmt: skip
    return files


def _next(link: str) -> str | None:
    # the next page's URL a Link header gives, if any
    match = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return match[1] if match else None


def _fetched(repo: str, f: File, path: Path, say: Callable[[str], None]) -> None:
    # a file downloaded to `path`, or checked there if it is, through NAME.part, resumed from
    # what that holds; a ValueError if its hash is not Hugging Face's
    if path.exists() and path.stat().st_size == f.size:
        say(f"{f.name}: there, checking it")
        if _matches(f, _hashed(path, f)):
            return
        say(f"{f.name}: not the file it should be, downloading it again")
        path.unlink()
    partial = path.with_name(path.name + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    if have == f.size and _matches(f, _hashed(partial, f)):  # whole, but for its name
        partial.rename(path)
        say(f"{f.name}: {f.size / 2**30:.2f} GiB, checked")
        return
    if have >= f.size:  # past it, or of other bytes: none of it to keep
        partial.unlink()
        have = 0
    url = f"{_endpoint()}/{repo}/resolve/main/{urllib.parse.quote(f.path)}"
    with _open(url, {"Range": f"bytes={have}-"} if have else None) as response:
        if have and response.status != 206:  # the server sends the whole file again
            have = 0
        hasher = _hasher(f)
        if have:
            _feed(hasher, partial, have)
        with open(partial, "ab" if have else "wb") as out:
            started, done = time.monotonic(), have
            while chunk := response.read(_BLOCK):
                out.write(chunk)
                hasher.update(chunk)
                done += len(chunk)
                _progress(f.name, done, f.size, (done - have) / (time.monotonic() - started))
    print(file=sys.stderr) if sys.stderr.isatty() else None
    if done != f.size or not _matches(f, hasher.hexdigest()):
        partial.unlink()
        raise ValueError(f"{f.name}: downloaded, but not the file Hugging Face lists: removed it")
    partial.rename(path)
    say(f"{f.name}: {f.size / 2**30:.2f} GiB, checked")


def _hasher(f: File) -> Any:
    # SHA-256 of a file LFS holds; else Git's SHA-1 of its blob, its header first
    if f.sha256 is not None:
        return hashlib.sha256()
    hasher = hashlib.sha1()
    hasher.update(f"blob {f.size}\0".encode())
    return hasher


def _feed(hasher: Any, path: Path, size: int) -> None:
    with open(path, "rb") as f:
        while size and (chunk := f.read(min(_BLOCK, size))):
            hasher.update(chunk)
            size -= len(chunk)


def _hashed(path: Path, f: File) -> str:
    hasher = _hasher(f)
    _feed(hasher, path, f.size)
    return str(hasher.hexdigest())


def _matches(f: File, digest: str) -> bool:
    return digest == (f.sha256 or f.git_sha1)


def _progress(name: str, done: int, size: int, rate: float) -> None:
    # how far a download is, on one line of a terminal, rewritten
    if sys.stderr.isatty():
        print(f"\r{name}: {done / 2**30:.2f} of {size / 2**30:.2f} GiB, "
              f"{rate / 2**20:.0f} MiB/s", end="", file=sys.stderr, flush=True)  # fmt: skip
