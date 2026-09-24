"""Canonical Churn acquisition and strictly train-fitted preprocessing.

The SHA-256 of the *entire* immutable archive is the trust anchor.  Member
digests are derived only after verifying that archive, never from a local
receipt that could certify its own arbitrary data.  ``load_churn`` is offline
and requires the archive as well as the extracted directory.  Synthetic data
is available only through an explicitly named in-memory test fixture.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO
from urllib.parse import urlsplit

import numpy as np

CHURN_SOURCE_REVISION = "dfb85cb493bfb7493de0b8c0bd9e2f9e6a56baa1"
CHURN_ARCHIVE_URL = (
    "https://huggingface.co/datasets/Yura52/tabpack-data/resolve/"
    f"{CHURN_SOURCE_REVISION}/tabpack-data.tar.gz"
)
CHURN_ARCHIVE_SHA256 = "89338c628fed24af03084c9348ca0a5c8ca4f12f8b988a576d9d1711cc661558"
CHURN_ARCHIVE_SIZE = 189_856_645
CHURN_ARCHIVE_NAME = "tabpack-data.tar.gz"
CHURN_MEMBER_ROOT = "data/churn/"
SPLITS = ("train", "val", "test")
CHURN_COUNTS = {"train": 6400, "val": 1600, "test": 2000}
# Exact allowed payload names, not a glob. Presence of feature blocks comes
# from the pinned archive itself, never from untrusted extracted local files.
CHURN_MEMBERS = frozenset({
    "info.json", "x_num.npy", "x_cat.npy", "x_bin.npy", "y.npy",
    "splits/default/train.npy", "splits/default/val.npy", "splits/default/test.npy",
})
CHURN_REQUIRED_MEMBERS = CHURN_MEMBERS - {"x_num.npy", "x_cat.npy", "x_bin.npy"}
CHURN_MEMBER_PATHS = tuple(CHURN_MEMBER_ROOT + name for name in sorted(CHURN_MEMBERS))
_CHUNK = 1024 * 1024
_MAX_CHURN_BYTES = 64 * 1024 * 1024
_DOWNLOAD_HOSTS = frozenset({
    "huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs.hf.co",
    "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co",
    "us.aws.cdn.hf.co",
})
_DEVICES = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])$", re.IGNORECASE)


class ChurnDataError(RuntimeError):
    """Fail-closed data error; ``code`` is stable for CLI diagnostics."""

    def __init__(self, code: str, summary: str) -> None:
        self.code = code
        self.summary = summary
        super().__init__(f"{code}: {summary}")


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = 20_000
    max_member_name_bytes: int = 512
    max_member_bytes: int = 512 * 1024 * 1024
    max_total_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024
    max_compression_ratio: int = 200
    max_churn_bytes: int = _MAX_CHURN_BYTES
    deadline_seconds: float = 120.0


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_plain_path(path: Path) -> None:
    """Reject links/reparse points in all existing path components."""
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        try:
            status = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(status.st_mode) or (
            getattr(status, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ChurnDataError("DATA_PATH_UNSAFE", "symlinks and reparse points are forbidden")


def _read_regular(path: Path, maximum: int) -> bytes:
    _check_plain_path(path)
    try:
        with path.open("rb") as stream:
            status = os.fstat(stream.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_size > maximum:
                raise ChurnDataError("DATA_SCHEMA_INVALID", "member is not a bounded regular file")
            result = stream.read(maximum + 1)
    except OSError as exc:
        raise ChurnDataError("DATA_MISSING", f"missing or unreadable member: {path.name}") from exc
    if len(result) > maximum:
        raise ChurnDataError("DATA_SCHEMA_INVALID", "member exceeds its size bound")
    return result


def _validate_https(url: str) -> None:
    try:
        parsed = urlsplit(url)
        allowed = (
            parsed.scheme == "https" and parsed.hostname in _DOWNLOAD_HOSTS
            and parsed.port in (None, 443) and not parsed.username
            and not parsed.password and not parsed.fragment
        )
    except ValueError:
        allowed = False
    if not allowed:
        # Never log signed CDN query strings.
        raise ChurnDataError("SOURCE_NOT_ALLOWLISTED", "unapproved HTTPS delivery URL")


class _PinnedRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _copy_verified(
    source: BinaryIO, target: BinaryIO, size: int, digest: str, *, deadline: float
) -> None:
    observed = hashlib.sha256()
    count = 0
    while True:
        if time.monotonic() > deadline:
            raise ChurnDataError("ACQUISITION_TIMEOUT", "bounded archive operation timed out")
        chunk = source.read(min(_CHUNK, size - count + 1))
        if not chunk:
            break
        count += len(chunk)
        if count > size:
            raise ChurnDataError("DATA_INTEGRITY_MISMATCH", "archive exceeds pinned byte size")
        observed.update(chunk)
        target.write(chunk)
    if count != size or observed.hexdigest() != digest:
        raise ChurnDataError("DATA_INTEGRITY_MISMATCH", "archive size or SHA-256 is not canonical")


def download_churn_archive(
    destination: str | Path, *, timeout: float = 30.0, overall_timeout: float = 600.0
) -> Path:
    """Explicitly download the pinned full bundle, then atomically publish it.

    ``destination`` is the archive filename. No source URL override, prefix
    download, retry, alternate dataset, or automatic model run is performed.
    """
    if not (0 < timeout <= 120 and 0 < overall_timeout <= 1800):
        raise ValueError("download timeouts must be positive and bounded")
    destination = Path(destination)
    _check_plain_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with _verified_archive_snapshot(destination):
            return destination
    _validate_https(CHURN_ARCHIVE_URL)
    request = urllib.request.Request(CHURN_ARCHIVE_URL, headers={"Accept-Encoding": "identity"})
    opener = urllib.request.build_opener(_PinnedRedirectHandler())
    descriptor, temporary = tempfile.mkstemp(prefix=".churn-download-", dir=destination.parent)
    deadline = time.monotonic() + overall_timeout
    try:
        with os.fdopen(descriptor, "wb") as target:
            with opener.open(request, timeout=timeout) as response:
                _validate_https(response.geturl())
                if response.status != 200 or response.headers.get("Content-Range"):
                    raise ChurnDataError("ACQUISITION_FAILED", "a complete HTTP 200 response is required")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ChurnDataError("ACQUISITION_FAILED", "unexpected HTTP content encoding")
                length = response.headers.get("Content-Length")
                if length is not None and length != str(CHURN_ARCHIVE_SIZE):
                    raise ChurnDataError("DATA_INTEGRITY_MISMATCH", "HTTP length differs from pinned size")
                _copy_verified(
                    response, target, CHURN_ARCHIVE_SIZE, CHURN_ARCHIVE_SHA256, deadline=deadline
                )
            target.flush()
            os.fsync(target.fileno())
        _check_plain_path(destination)
        # Atomic and no-clobber on Windows and POSIX: the complete temporary
        # file is linked into place, and its private temporary name is removed.
        os.link(temporary, destination)
    except ChurnDataError:
        raise
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise ChurnDataError("ACQUISITION_FAILED", "pinned archive download failed") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


@contextmanager
def _verified_archive_snapshot(
    path: Path, *, size: int = CHURN_ARCHIVE_SIZE, digest: str = CHURN_ARCHIVE_SHA256
):
    """Parse a private captured snapshot, avoiding a hash-then-reopen race."""
    _check_plain_path(path)
    try:
        with path.open("rb") as source, tempfile.TemporaryFile() as snapshot:
            status = os.fstat(source.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_size != size:
                raise ChurnDataError("DATA_INTEGRITY_MISMATCH", "archive has the wrong byte size")
            _copy_verified(source, snapshot, size, digest, deadline=time.monotonic() + 120)
            snapshot.seek(0)
            yield snapshot
    except OSError as exc:
        raise ChurnDataError("DATA_MISSING", "verified canonical archive is unavailable") from exc


def _safe_tar_name(name: str, *, directory: bool, limit: int = 512) -> str:
    if name.startswith("./"):
        name = name[2:]
    if directory and name.endswith("/"):
        name = name[:-1]
    if (
        not name or len(name.encode("utf-8", "surrogateescape")) > limit
        or name.startswith("/") or "\\" in name
        or any(ord(c) < 32 or ord(c) == 127 for c in name)
    ):
        raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "unsafe archive member name")
    for part in name.split("/"):
        if (
            part in {"", ".", ".."} or part.endswith((".", " "))
            or any(c in part for c in ':<>"|?*') or _DEVICES.fullmatch(part.split(".")[0])
        ):
            raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "unsafe archive path component")
    return name


class _BoundedReader:
    """Count *all* decompressed bytes, including skipped payloads and headers."""
    def __init__(self, source: BinaryIO, maximum: int, deadline: float):
        self.source, self.maximum, self.deadline = source, maximum, deadline
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        if time.monotonic() > self.deadline:
            raise ChurnDataError("ARCHIVE_LIMIT_EXCEEDED", "archive parsing deadline exceeded")
        if size < 0:
            size = _CHUNK
        data = self.source.read(min(size, self.maximum - self.count + 1))
        self.count += len(data)
        if self.count > self.maximum:
            raise ChurnDataError("ARCHIVE_LIMIT_EXCEEDED", "decompressed stream exceeds limit")
        return data


def _scan_archive(
    snapshot: BinaryIO, *, compressed_size: int, limits: ArchiveLimits = ArchiveLimits()
) -> dict[str, bytes]:
    """Read only allowlisted members into bounded memory; never tar.extract."""
    class BoundedTarInfo(tarfile.TarInfo):
        def _reject_sparse(self, *args, **kwargs):
            raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "PAX sparse payloads are forbidden")

        # Reject GNU sparse PAX formats before tarfile reads/allocates their
        # auxiliary extent tables, not merely after yielding the final member.
        _proc_gnusparse_00 = _reject_sparse
        _proc_gnusparse_01 = _reject_sparse
        _proc_gnusparse_10 = _reject_sparse

        @classmethod
        def frombuf(cls, buf, encoding, errors):
            member = super().frombuf(buf, encoding, errors)
            # Bound extensions before tarfile allocates their payloads.
            extension = member.type in (tarfile.XHDTYPE, tarfile.XGLTYPE,
                                         tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK)
            is_churn_member = member.name.startswith(CHURN_MEMBER_ROOT)
            ceiling = 64 * 1024 if extension else (
                limits.max_member_bytes if is_churn_member else limits.max_total_uncompressed_bytes
            )
            if member.size < 0 or member.size > ceiling:
                raise ChurnDataError(
                    "ARCHIVE_LIMIT_EXCEEDED",
                    f"tar header size exceeds limit: {member.name!r} type={member.type!r} size={member.size}",
                )
            if member.type == tarfile.GNUTYPE_SPARSE:
                raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "sparse archive files are forbidden")
            return member

    payloads: dict[str, bytes] = {}
    names: dict[str, bool] = {}
    file_ancestors: set[str] = set()
    total = 0
    churn_bytes = 0
    maximum = min(limits.max_total_uncompressed_bytes, compressed_size * limits.max_compression_ratio)
    try:
        with gzip.GzipFile(fileobj=snapshot, mode="rb") as decompressor:
            bounded = _BoundedReader(decompressor, maximum, time.monotonic() + limits.deadline_seconds)
            with tarfile.open(fileobj=bounded, mode="r|", tarinfo=BoundedTarInfo) as archive:
                for index, member in enumerate(archive):
                    if index >= limits.max_members:
                        raise ChurnDataError("ARCHIVE_LIMIT_EXCEEDED", "archive has too many members")
                    name = _safe_tar_name(member.name, directory=member.isdir(), limit=limits.max_member_name_bytes)
                    key = name.casefold()
                    ancestors = ["/".join(key.split("/")[:i]) for i in range(1, len(key.split("/")))]
                    if key in names or any(names.get(parent) is False for parent in ancestors):
                        raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "duplicate or colliding archive path")
                    if member.isreg() and key in file_ancestors:
                        raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "file/directory path collision")
                    names[key] = member.isdir()
                    file_ancestors.update(ancestors)
                    member_ceiling = (
                        limits.max_member_bytes
                        if name.startswith(CHURN_MEMBER_ROOT)
                        else limits.max_total_uncompressed_bytes
                    )
                    if member.size < 0 or member.size > member_ceiling:
                        raise ChurnDataError("ARCHIVE_LIMIT_EXCEEDED", "member exceeds size limit")
                    if member.sparse is not None or not (member.isreg() or member.isdir()):
                        raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "links, sparse, and special files are forbidden")
                    if member.isdir():
                        if member.size:
                            raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "directory has a payload")
                        continue
                    total += member.size
                    if total > maximum:
                        raise ChurnDataError("ARCHIVE_LIMIT_EXCEEDED", "member total exceeds limit")
                    if name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".xz", ".bz2")):
                        raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "nested archives are forbidden")
                    # The pinned macOS-created bundle contains AppleDouble
                    # metadata alongside the real files (for example
                    # ``._x_num.npy``).  These are metadata sidecars, not
                    # dataset members; validate their safe path/size above,
                    # count them against archive limits, and deliberately do
                    # not publish them or treat them as a Churn schema error.
                    if any(part.startswith("._") for part in name.split("/")):
                        continue
                    if key.startswith(CHURN_MEMBER_ROOT) and not name.startswith(CHURN_MEMBER_ROOT):
                        raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "Churn path has incorrect casing")
                    if not name.startswith(CHURN_MEMBER_ROOT):
                        continue
                    relative = name.removeprefix(CHURN_MEMBER_ROOT)
                    if relative not in CHURN_MEMBERS:
                        raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "unexpected Churn member")
                    churn_bytes += member.size
                    if churn_bytes > limits.max_churn_bytes:
                        raise ChurnDataError("ARCHIVE_LIMIT_EXCEEDED", "Churn payload exceeds memory limit")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "member has no regular payload")
                    with stream:
                        content = stream.read(member.size + 1)
                    if len(content) != member.size:
                        raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "truncated member payload")
                    payloads[relative] = content
            # Validate gzip CRC/trailer and count bytes beyond tar's end marker.
            while bounded.read(_CHUNK):
                pass
    except ChurnDataError:
        raise
    except (tarfile.TarError, OSError, EOFError, ValueError) as exc:
        raise ChurnDataError("ARCHIVE_MEMBER_INVALID", "invalid or truncated tar.gz archive") from exc
    if not CHURN_REQUIRED_MEMBERS.issubset(payloads):
        raise ChurnDataError("DATA_MISSING", "canonical Churn metadata, target, or split members are missing")
    if not ({"x_num.npy", "x_cat.npy", "x_bin.npy"} & payloads.keys()):
        raise ChurnDataError("DATA_MISSING", "Churn archive has no feature block")
    return payloads


def _canonical_payloads(path: Path, limits: ArchiveLimits) -> dict[str, bytes]:
    with _verified_archive_snapshot(path) as snapshot:
        return _scan_archive(snapshot, compressed_size=CHURN_ARCHIVE_SIZE, limits=limits)


def _inventory(payloads: Mapping[str, bytes]) -> tuple[dict[str, Any], ...]:
    roles = {
        "info.json": "schema", "y.npy": "target", "x_num.npy": "features.numeric",
        "x_cat.npy": "features.categorical", "x_bin.npy": "features.binary",
        **{f"splits/default/{part}.npy": f"split.{part}" for part in SPLITS},
    }
    return tuple({"path": name, "role": roles[name], "size_bytes": len(data), "sha256": _digest(data)}
                 for name, data in sorted(payloads.items()))


def extract_churn_archive(
    archive_path: str | Path, destination: str | Path, *, limits: ArchiveLimits = ArchiveLimits()
) -> Path:
    """Publish a new Churn directory after full archive and schema validation.

    ``destination`` is the dataset directory (e.g. ``data/churn``), not the
    tar root. The original archive is retained separately for offline loading.
    """
    destination = Path(destination)
    _check_plain_path(destination)
    if destination.exists():
        raise ChurnDataError("DATA_DESTINATION_EXISTS", "use a new destination for verified extraction")
    payloads = _canonical_payloads(Path(archive_path), limits)
    _raw_from_payloads(payloads, canonical=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".churn-extract-", dir=destination.parent))
    try:
        for name, content in payloads.items():
            target = staging.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
        _check_plain_path(destination)
        if destination.exists():
            raise ChurnDataError("DATA_DESTINATION_EXISTS", "destination appeared during extraction")
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def acquire_churn(root: str | Path, *, archive_path: str | Path | None = None) -> Path:
    """Explicit acquisition convenience API; no network requests from loading."""
    root = Path(root)
    archive = Path(archive_path) if archive_path is not None else root.parent / CHURN_ARCHIVE_NAME
    download_churn_archive(archive)
    return extract_churn_archive(archive, root)


def _npy(data: bytes, name: str) -> np.ndarray:
    """Validate header, dtype and payload length before allocating an array."""
    stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=16384)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=16384)
        else:
            raise ValueError("unsupported NPY version")
        if (
            dtype.hasobject or dtype.fields is not None or dtype.kind not in "fiuUS"
            or dtype.itemsize <= 0 or len(shape) not in (1, 2)
            or any(not isinstance(dim, int) or dim < 0 for dim in shape)
        ):
            raise ValueError("unsafe NPY shape or dtype")
        size = math.prod(shape) * dtype.itemsize
        if size > _MAX_CHURN_BYTES or stream.tell() + size != len(data):
            raise ValueError("NPY payload size does not match its header")
        stream.seek(0)
        result = np.load(stream, allow_pickle=False, max_header_size=16384)
        if not isinstance(result, np.ndarray):
            raise ValueError("expected an NPY array")
        return result
    except (ValueError, TypeError, EOFError, OverflowError) as exc:
        raise ChurnDataError("DATA_SCHEMA_INVALID", f"invalid bounded NPY: {name}") from exc


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FeatureBlock:
    numeric: np.ndarray | None = None
    categorical: np.ndarray | None = None
    row_ids: np.ndarray | None = None
    numeric_names: tuple[str, ...] = ()
    categorical_names: tuple[str, ...] = ()

    def __post_init__(self):
        rows = set()
        for name in ("numeric", "categorical"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, np.ndarray) or value.ndim != 2:
                    raise ChurnDataError("DATA_SCHEMA_INVALID", "features must be rank-two arrays")
                rows.add(len(value))
                object.__setattr__(self, name, _readonly(value))
        if len(rows) != 1 or not self.n_numeric + self.n_categorical or not self.n_rows:
            raise ChurnDataError("DATA_SCHEMA_INVALID", "empty or misaligned feature blocks")
        for kind in ("numeric", "categorical"):
            width = getattr(self, "n_" + kind)
            names = getattr(self, kind + "_names") or tuple(f"{kind}_{i}" for i in range(width))
            if len(names) != width or len(set(names)) != width:
                raise ChurnDataError("DATA_SCHEMA_INVALID", "feature names do not match schema")
            object.__setattr__(self, kind + "_names", tuple(names))
        if self.row_ids is not None:
            if self.row_ids.ndim != 1 or len(self.row_ids) != self.n_rows:
                raise ChurnDataError("SPLIT_INVALID", "row IDs must match feature rows")
            object.__setattr__(self, "row_ids", _readonly(self.row_ids))

    @property
    def n_rows(self) -> int:
        value = self.numeric if self.numeric is not None else self.categorical
        return 0 if value is None else len(value)

    @property
    def n_numeric(self) -> int:
        return 0 if self.numeric is None else self.numeric.shape[1]

    @property
    def n_categorical(self) -> int:
        return 0 if self.categorical is None else self.categorical.shape[1]


FeatureBlockSet = FeatureBlock


@dataclass(frozen=True)
class RawChurn:
    train: FeatureBlock
    val: FeatureBlock
    test: FeatureBlock
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    dataset_id: str
    inventory: tuple[dict[str, Any], ...] = ()
    dataset_sha256: str = ""

    @property
    def source_indices(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({part: getattr(self, part).row_ids for part in SPLITS})

    @property
    def y(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({part: getattr(self, "y_" + part) for part in SPLITS})


def _validate_indices(indices: Mapping[str, np.ndarray], n_rows: int) -> None:
    seen: set[int] = set()
    for part in SPLITS:
        values = indices[part]
        if not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
            raise ChurnDataError("SPLIT_INVALID", "indices must be nonempty integer vectors")
        current = set(values.tolist())
        if len(current) != len(values) or min(current) < 0 or max(current) >= n_rows or current & seen:
            raise ChurnDataError("SPLIT_INVALID", "split indices are repeated, overlapping, or out of range")
        seen.update(current)
    if len(seen) != n_rows:
        raise ChurnDataError("SPLIT_INVALID", "canonical splits do not cover all source rows")


def _raw_from_payloads(payloads: Mapping[str, bytes], *, canonical: bool) -> RawChurn:
    if not CHURN_REQUIRED_MEMBERS.issubset(payloads) or payloads.keys() - CHURN_MEMBERS:
        raise ChurnDataError("DATA_SCHEMA_INVALID", "missing or unexpected source members")
    try:
        info = json.loads(payloads["info.json"])
        if info["task"]["type"] != "binclass":
            raise ValueError("binary task required")
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise ChurnDataError("DATA_SCHEMA_INVALID", "invalid binary-task metadata") from exc
    arrays = {name: _npy(data, name) for name, data in payloads.items() if name.endswith(".npy")}
    y = arrays["y.npy"]
    if y.ndim != 1 or y.dtype != np.dtype("int64") or not np.array_equal(np.unique(y), [0, 1]):
        raise ChurnDataError("DATA_SCHEMA_INVALID", "target must be int64 with classes exactly 0 and 1")
    numeric = arrays.get("x_num.npy")
    categorical = arrays.get("x_cat.npy")
    binary = arrays.get("x_bin.npy")
    for name, value in (("x_num", numeric), ("x_cat", categorical), ("x_bin", binary)):
        if value is None:
            continue
        if value.ndim != 2 or value.shape[0] != len(y) or not value.shape[1]:
            raise ChurnDataError("DATA_SCHEMA_INVALID", f"wrong feature shape: {name}")
        if name == "x_cat":
            if value.dtype != np.dtype("int64") and value.dtype.kind != "U":
                raise ChurnDataError("DATA_SCHEMA_INVALID", "categorical dtype must be int64 or Unicode")
        elif value.dtype != np.dtype("float32") or np.any(np.isinf(value)):
            raise ChurnDataError("DATA_SCHEMA_INVALID", "numeric/binary dtype must be finite-or-NaN float32")
    if binary is not None:
        if not np.all(np.isin(binary, [0, 1]) | np.isnan(binary)):
            raise ChurnDataError("DATA_SCHEMA_INVALID", "binary block contains values outside 0/1/NaN")
        converted = binary.astype(object)
        converted[np.isnan(binary)] = None
        categorical = converted if categorical is None else np.column_stack((categorical, converted))
    indices = {part: arrays[f"splits/default/{part}.npy"] for part in SPLITS}
    if any(value.dtype != np.dtype("int32") for value in indices.values()):
        raise ChurnDataError("SPLIT_INVALID", "canonical split arrays must have int32 dtype")
    _validate_indices(indices, len(y))
    if canonical and any(len(indices[part]) != CHURN_COUNTS[part] for part in SPLITS):
        raise ChurnDataError("SPLIT_INVALID", "canonical split counts must be 6400/1600/2000")
    blocks, targets = {}, {}
    for part, index in indices.items():
        if not np.array_equal(np.unique(y[index]), [0, 1]):
            raise ChurnDataError("SPLIT_INVALID", "every ROC-AUC split must contain both classes")
        blocks[part] = FeatureBlock(
            numeric=None if numeric is None else numeric[index],
            categorical=None if categorical is None else categorical[index], row_ids=index,
        )
        targets["y_" + part] = _readonly(y[index])
    inventory = _inventory(payloads)
    return RawChurn(
        **blocks, **targets, dataset_id="churn" if canonical else "synthetic_test_v1",
        inventory=inventory, dataset_sha256=_digest(_json_bytes(inventory)),
    )


def load_churn(root: str | Path, *, archive_path: str | Path | None = None) -> RawChurn:
    """Verify the canonical archive and exact local member bytes, then load.

    The archive defaults to ``root.parent / 'tabpack-data.tar.gz'``.  Retaining
    it lets this offline loader establish canonical member digests without
    trusting editable acquisition receipts. No network or fallback is used.
    """
    root = Path(root)
    _check_plain_path(root)
    if not root.is_dir():
        raise ChurnDataError("DATA_MISSING", "canonical Churn directory is missing; run explicit acquisition")
    archive = Path(archive_path) if archive_path is not None else root.parent / CHURN_ARCHIVE_NAME
    expected = _canonical_payloads(archive, ArchiveLimits())
    actual_files = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            _check_plain_path(Path(directory) / name)
        for name in filenames:
            actual_files.add((Path(directory) / name).relative_to(root).as_posix())
    if actual_files != expected.keys():
        raise ChurnDataError("DATA_INTEGRITY_MISMATCH", "local Churn inventory differs from verified archive")
    captured = {}
    for name, original in expected.items():
        value = _read_regular(root.joinpath(*name.split("/")), len(original))
        if len(value) != len(original) or _digest(value) != _digest(original):
            raise ChurnDataError("DATA_INTEGRITY_MISMATCH", f"local bytes differ from canonical member: {name}")
        captured[name] = value
    return _raw_from_payloads(captured, canonical=True)


def _missing(value: Any) -> bool:
    return value is None or isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _numeric(block: FeatureBlock) -> np.ndarray:
    if block.numeric is None or block.numeric.dtype.kind not in "fiu":
        raise ChurnDataError("DATA_SCHEMA_INVALID", "numeric block must contain real numbers")
    result = np.asarray(block.numeric, dtype=np.float64)
    if np.any(np.isinf(result)):
        raise ChurnDataError("DATA_SCHEMA_INVALID", "numeric features contain infinity")
    return result


def _categorical(block: FeatureBlock) -> tuple[np.ndarray, tuple[str | None, ...]]:
    values = np.array(block.categorical, dtype=object, copy=True)
    types: list[str | None] = []
    for column in values.T:
        kind = None
        for value in column:
            if _missing(value):
                continue
            current = "str" if isinstance(value, str) else (
                "number" if isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_)) and np.isfinite(value) else None
            )
            if current is None or kind is not None and current != kind:
                raise ChurnDataError("DATA_SCHEMA_INVALID", "categorical columns need homogeneous strings or numbers")
            kind = current
        types.append(kind)
    return values, tuple(types)


def _fill_modes(values: np.ndarray, modes: Sequence[Any]) -> np.ndarray:
    filled = values.copy()
    for index, mode in enumerate(modes):
        for row in range(len(filled)):
            if _missing(filled[row, index]):
                filled[row, index] = mode
    return filled


class FittedPreprocessor:
    """One fitted object; transforming never refits or extends its vocabulary."""
    def __init__(self, *, numeric_imputer, quantiles, encoder, modes, kinds, train, state):
        self._numeric_imputer = numeric_imputer
        self._quantiles = quantiles
        self._encoder = encoder
        self._modes = modes
        self._kinds = kinds
        self.numeric_names = train.numeric_names
        self.categorical_names = train.categorical_names
        self._state_json = _json_bytes(state)

    @property
    def metadata(self) -> dict[str, Any]:
        """A fresh JSON-safe snapshot, including quantiles and category offsets."""
        return json.loads(self._state_json)

    @property
    def fingerprint(self) -> str:
        return _digest(self._state_json)

    @property
    def learned_state_sha256(self) -> str:
        return self.fingerprint

    @property
    def numeric_medians(self) -> tuple[float, ...]:
        return tuple(self.metadata["numeric"]["medians"])

    @property
    def categorical_modes(self) -> tuple[Any, ...]:
        return tuple(self._modes)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["feature_names"])

    @property
    def output_width(self) -> int:
        return len(self.feature_names)

    def transform(self, features: FeatureBlock) -> np.ndarray:
        if features.numeric_names != self.numeric_names or features.categorical_names != self.categorical_names:
            raise ChurnDataError("PREPROCESS_STATE_MISMATCH", "input feature schema differs from training")
        parts = []
        if self._quantiles is not None:
            parts.append(self._quantiles.transform(self._numeric_imputer.transform(_numeric(features))))
        if self._encoder is not None:
            values, kinds = _categorical(features)
            if any(kind is not None and kind != fitted for kind, fitted in zip(kinds, self._kinds)):
                raise ChurnDataError("PREPROCESS_STATE_MISMATCH", "categorical type differs from training")
            parts.append(self._encoder.transform(_fill_modes(values, self._modes)))
        result = np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ChurnDataError("PREPROCESS_NONFINITE", "transformed matrix is non-finite")
        return result


def fit_preprocessor(
    train: FeatureBlock, *, n_quantiles: int = 1000, random_state: int = 0
) -> FittedPreprocessor:
    """Median + normal quantiles; categorical mode + one-hot unknown-ignore.

    Missing numeric entries are NaN. Categorical missing entries are None or
    floating NaN; literal strings such as 'nan' and '' remain real categories.
    All-missing train columns fail. Mode ties use ascending typed value order.
    """
    if type(n_quantiles) is not int or n_quantiles <= 0 or type(random_state) is not int:
        raise ValueError("n_quantiles must be a positive integer; random_state must be an integer")
    try:
        import scipy
        import sklearn
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    except ImportError as exc:
        raise ChurnDataError("PREPROCESS_DEPENDENCY_MISSING", "install the pinned scikit-learn dependencies") from exc
    imputer = quantiles = encoder = None
    modes, kinds = (), ()
    effective_quantiles = min(n_quantiles, train.n_rows)
    state = {
        "schema_version": "preprocess-state/v1", "fit_split": "train", "fit_row_count": train.n_rows,
        "config": {"n_quantiles": effective_quantiles, "subsample": train.n_rows,
                   "output_distribution": "normal", "random_state": random_state,
                   "numeric_imputation": "median", "categorical_imputation": "most_frequent",
                   "unknown_categories": "ignore", "noise": False},
        "numeric": {"medians": [], "quantiles": [], "references": []},
        "categorical": {"modes": [], "categories": [], "block_offsets": [], "types": []},
        "feature_names": list(train.numeric_names),
        "input_schema": {"numeric": list(train.numeric_names), "categorical": list(train.categorical_names)},
        "software": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    }
    if train.n_numeric:
        values = _numeric(train)
        if np.any(np.all(np.isnan(values), axis=0)):
            raise ChurnDataError("ALL_MISSING_TRAIN_COLUMN", "numeric train median is undefined")
        imputer = SimpleImputer(strategy="median")
        filled = imputer.fit_transform(values)
        quantiles = QuantileTransformer(
            n_quantiles=effective_quantiles, output_distribution="normal",
            subsample=train.n_rows, random_state=random_state,
        ).fit(filled)
        state["numeric"] = {
            "medians": imputer.statistics_.tolist(), "quantiles": quantiles.quantiles_.tolist(),
            "references": quantiles.references_.tolist(),
        }
    if train.n_categorical:
        values, kinds = _categorical(train)
        learned_modes = []
        for column in values.T:
            counts = Counter(value for value in column if not _missing(value))
            if not counts:
                raise ChurnDataError("ALL_MISSING_TRAIN_COLUMN", "categorical train mode is undefined")
            largest = max(counts.values())
            learned_modes.append(min(value for value, count in counts.items() if count == largest))
        modes = tuple(value.item() if isinstance(value, np.generic) else value for value in learned_modes)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        encoder.fit(_fill_modes(values, modes))
        offset = train.n_numeric
        for name, categories in zip(train.categorical_names, encoder.categories_):
            state["categorical"]["block_offsets"].append([offset, offset + len(categories)])
            offset += len(categories)
            state["categorical"]["categories"].append(categories.tolist())
            state["feature_names"].extend(f"{name}={value!r}" for value in categories)
        state["categorical"]["modes"] = list(modes)
        state["categorical"]["types"] = list(kinds)
    fitted = FittedPreprocessor(
        numeric_imputer=imputer, quantiles=quantiles, encoder=encoder, modes=modes,
        kinds=kinds, train=train, state=state,
    )
    fitted.transform(train)  # Validate float32 output before publishing the fitted state.
    return fitted


def transform_features(fitted: FittedPreprocessor, features: FeatureBlock) -> np.ndarray:
    return fitted.transform(features)


@dataclass(frozen=True)
class PreparedData:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: FittedPreprocessor
    dataset_id: str
    dataset_sha256: str
    source_indices: Mapping[str, np.ndarray]

    @property
    def x(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({part: getattr(self, part) for part in SPLITS})

    @property
    def y(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({part: getattr(self, "y_" + part) for part in SPLITS})


def prepare_data(raw: RawChurn, *, n_quantiles: int = 1000, random_state: int = 0) -> PreparedData:
    """Shared preprocessing orchestration; labels never enter fit or transform.

    Training/selection callers should receive only train/val views. The test
    matrix is a frozen transform, and must be evaluated after selection freezes.
    """
    total = sum(getattr(raw, part).n_rows for part in SPLITS)
    _validate_indices(raw.source_indices, total)
    for part in SPLITS:
        labels = getattr(raw, "y_" + part)
        if (
            not isinstance(labels, np.ndarray) or labels.ndim != 1
            or len(labels) != getattr(raw, part).n_rows or labels.dtype.kind not in "iu"
            or not np.array_equal(np.unique(labels), [0, 1])
        ):
            raise ChurnDataError("DATA_SCHEMA_INVALID", "targets must be aligned binary integer labels with both classes")
    fitted = fit_preprocessor(raw.train, n_quantiles=n_quantiles, random_state=random_state)
    return PreparedData(
        **{part: fitted.transform(getattr(raw, part)) for part in SPLITS},
        **{"y_" + part: _readonly(getattr(raw, "y_" + part)) for part in SPLITS},
        preprocessor=fitted, dataset_id=raw.dataset_id, dataset_sha256=raw.dataset_sha256,
        source_indices=raw.source_indices,
    )


def make_synthetic_fixture() -> RawChurn:
    """Explicit 32/12/12-row test seam; never called by acquisition or loading."""
    blocks, targets = {}, {}
    identity = []
    for part, start, count in (("train", 0, 32), ("val", 32, 12), ("test", 44, 12)):
        numeric, categories, labels = [], [], []
        for offset in range(count):
            index = start + offset
            label = index % 2
            numeric.append([
                np.nan if index % 7 == 0 else (2 * label - 1) + ((index // 2) % 4) / 16,
                ((3 * index) % 11) / 10,
            ])
            category = (None if index % 9 == 0 else "violet" if part != "train" and offset % 5 == 0
                        else "amber" if label == 0 else "blue")
            categories.append([category])
            labels.append(label)
        blocks[part] = FeatureBlock(
            np.array(numeric, dtype=np.float64), np.array(categories, dtype=object),
            np.arange(start, start + count), ("x0", "x1"), ("cat",),
        )
        targets["y_" + part] = _readonly(np.array(labels, dtype=np.int64))
        identity.append({"split": part, "row_count": count, "start": start})
    return RawChurn(
        **blocks, **targets, dataset_id="synthetic_smoke_v1",
        dataset_sha256=_digest(_json_bytes({"fixture": "synthetic_smoke_v1", "parts": identity})),
    )


synthetic_fixture = make_synthetic_fixture
PreprocessedData = PreparedData
