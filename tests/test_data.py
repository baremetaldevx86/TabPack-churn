"""Offline adversarial acquisition, parsing, and train-only preprocessing tests."""

import dataclasses
import hashlib
import importlib.util
import io
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from tabpack import data

requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="scikit-learn is not installed"
)


def npy(value):
    output = io.BytesIO()
    np.save(output, value)
    return output.getvalue()


def payload_fixture():
    # Deliberately nonmonotonic canonical-style indices exercise row order.
    return {
        "info.json": b'{"task":{"type":"binclass","score":"accuracy"}}',
        "x_num.npy": npy(np.arange(24, dtype=np.float32).reshape(12, 2)),
        "x_cat.npy": npy(np.array([["a"], ["b"]] * 6)),
        "y.npy": npy(np.array([0, 1] * 6, dtype=np.int64)),
        "splits/default/train.npy": npy(np.array([4, 1, 2, 3, 0, 5], dtype=np.int32)),
        "splits/default/val.npy": npy(np.array([6, 7], dtype=np.int32)),
        "splits/default/test.npy": npy(np.array([8, 9, 10, 11], dtype=np.int32)),
    }


def tar_fixture(extras=(), *, payloads=None):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name in ("data", "data/churn", "data/churn/splits", "data/churn/splits/default"):
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
        for name, value in (payload_fixture() if payloads is None else payloads).items():
            member = tarfile.TarInfo("data/churn/" + name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
        for member, value in extras:
            archive.addfile(member, io.BytesIO(value))
    return output.getvalue()


def scan(value, limits=None):
    return data._scan_archive(
        io.BytesIO(value), compressed_size=len(value),
        limits=limits or data.ArchiveLimits(max_compression_ratio=10000),
    )


@pytest.mark.parametrize("url", [
    "http://huggingface.co/file", "https://evil.example/file",
    "https://huggingface.co.evil.example/file", "https://user:secret@huggingface.co/file",
    "https://huggingface.co:80/file", "https://huggingface.co/file#secret",
])
def test_transport_rejects_unapproved_urls(url):
    with pytest.raises(data.ChurnDataError, match="SOURCE_NOT_ALLOWLISTED"):
        data._validate_https(url)


def test_redirect_checks_target_before_following():
    handler = data._PinnedRedirectHandler()
    with pytest.raises(data.ChurnDataError, match="SOURCE_NOT_ALLOWLISTED"):
        handler.redirect_request(
            urllib.request.Request(data.CHURN_ARCHIVE_URL), None, 302, "Found", {},
            "http://evil.example/secret?token=do-not-log",
        )
    data._validate_https("https://cas-bridge.xethub.hf.co/file?signature=opaque")
    data._validate_https("https://us.aws.cdn.hf.co/file?signature=opaque")


@pytest.mark.parametrize("content,size,digest", [
    (b"abc", 4, hashlib.sha256(b"abc").hexdigest()),
    (b"abcd", 3, hashlib.sha256(b"abc").hexdigest()),
    (b"abc", 3, "0" * 64),
])
def test_archive_copy_rejects_partial_oversized_and_wrong_digest(content, size, digest):
    with pytest.raises(data.ChurnDataError, match="DATA_INTEGRITY_MISMATCH"):
        data._copy_verified(io.BytesIO(content), io.BytesIO(), size, digest, deadline=time.monotonic() + 10)


def test_verified_copy_and_snapshot_bind_bytes(tmp_path):
    content = tar_fixture()
    archive = tmp_path / "fixture.tar.gz"
    archive.write_bytes(content)
    with data._verified_archive_snapshot(archive, size=len(content), digest=data._digest(content)) as stream:
        archive.write_bytes(b"modified after verification")
        assert scan(stream.read()) == payload_fixture()
    with pytest.raises(data.ChurnDataError, match="DATA_INTEGRITY_MISMATCH"):
        data.extract_churn_archive(archive, tmp_path / "churn")
    assert not (tmp_path / "churn").exists()


def test_download_cleanup_on_bad_digest(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {}

        def geturl(self):
            return data.CHURN_ARCHIVE_URL

    class Opener:
        def open(self, request, timeout):
            return Response(b"not a canonical archive")

    monkeypatch.setattr(data.urllib.request, "build_opener", lambda *args: Opener())
    target = tmp_path / "bundle.tar.gz"
    with pytest.raises(data.ChurnDataError, match="DATA_INTEGRITY_MISMATCH"):
        data.download_churn_archive(target)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status,headers", [
    (206, {"Content-Range": "bytes 0-1/100"}),
    (200, {"Content-Encoding": "gzip"}),
    (200, {"Content-Length": "2"}),
])
def test_download_rejects_partial_encoded_or_wrong_length(tmp_path, monkeypatch, status, headers):
    class Response(io.BytesIO):
        def geturl(self):
            return data.CHURN_ARCHIVE_URL

    response = Response(b"ab")
    response.status, response.headers = status, headers

    class Opener:
        def open(self, request, timeout):
            return response

    monkeypatch.setattr(data.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(data.ChurnDataError):
        data.download_churn_archive(tmp_path / "bundle.tar.gz")
    assert list(tmp_path.iterdir()) == []


def test_allowlist_skips_other_datasets_and_preserves_churn():
    other = tarfile.TarInfo("data/other/x.npy")
    other.size = 6
    assert scan(tar_fixture([(other, b"opaque")])) == payload_fixture()


@pytest.mark.parametrize("name", [
    "../escape", "/escape", "C:/escape", "\\\\server\\share", "data\\escape",
    "data/churn/../escape", "data/churn/NUL.npy", "data/churn/COM1.txt",
    "data/churn/trailing.", "data/churn/trailing ", "data/churn/file:stream",
    "data/churn/line\nbreak", "data/churn/./x", "data/churn/unapproved.npy",
    "data/churn/y.npy", "data/churn/Y.NPY",
])
def test_tar_rejects_unsafe_extra_duplicate_and_case_collision(name):
    member = tarfile.TarInfo(name)
    member.size = 1
    with pytest.raises(data.ChurnDataError, match="ARCHIVE_MEMBER_INVALID"):
        scan(tar_fixture([(member, b"x")]))


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_tar_rejects_links_and_special_files(kind):
    member = tarfile.TarInfo("data/other/special")
    member.type, member.linkname = kind, "../../escape"
    with pytest.raises(data.ChurnDataError, match="ARCHIVE_MEMBER_INVALID"):
        scan(tar_fixture([(member, b"")]))


def test_tar_rejects_file_directory_collision():
    member = tarfile.TarInfo("data/churn/y.npy/child")
    with pytest.raises(data.ChurnDataError, match="ARCHIVE_MEMBER_INVALID"):
        scan(tar_fixture([(member, b"")]))


@pytest.mark.parametrize("limits", [
    data.ArchiveLimits(max_members=2),
    data.ArchiveLimits(max_member_name_bytes=8),
    data.ArchiveLimits(max_member_bytes=8),
    data.ArchiveLimits(max_total_uncompressed_bytes=2048),
    data.ArchiveLimits(max_compression_ratio=1),
    data.ArchiveLimits(max_churn_bytes=10),
    data.ArchiveLimits(deadline_seconds=-1),
])
def test_tar_limits_include_headers_and_skipped_payloads(limits):
    with pytest.raises(data.ChurnDataError):
        scan(tar_fixture(), limits)


def test_skipped_large_payload_is_bounded():
    member = tarfile.TarInfo("data/other/large.npy")
    member.size = 200_000
    value = tar_fixture([(member, b"0" * member.size)])
    with pytest.raises(data.ChurnDataError, match="ARCHIVE_LIMIT_EXCEEDED"):
        scan(value, data.ArchiveLimits(max_total_uncompressed_bytes=30_000, max_compression_ratio=10000))


def test_nested_archive_and_missing_churn_fail():
    member = tarfile.TarInfo("data/other/bundle.zip")
    with pytest.raises(data.ChurnDataError, match="ARCHIVE_MEMBER_INVALID"):
        scan(tar_fixture([(member, b"")]))
    payloads = payload_fixture()
    del payloads["splits/default/test.npy"]
    with pytest.raises(data.ChurnDataError, match="DATA_MISSING"):
        scan(tar_fixture(payloads=payloads))


def test_truncated_gzip_fails():
    with pytest.raises(data.ChurnDataError, match="ARCHIVE_MEMBER_INVALID"):
        scan(tar_fixture()[:-6])


def test_pax_sparse_is_rejected_before_reading_extent_tables():
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo("data/other/sparse.npy")
        member.size = 512
        member.pax_headers = {"GNU.sparse.major": "1", "GNU.sparse.minor": "0"}
        archive.addfile(member, io.BytesIO(b"9999999999999\n".ljust(512, b"\0")))
    with pytest.raises(data.ChurnDataError, match="PAX sparse payloads are forbidden"):
        scan(output.getvalue())


def test_extended_header_allocation_is_bounded():
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo("data/other/small.npy")
        member.pax_headers = {"comment": "x" * 70_000}
        archive.addfile(member, io.BytesIO())
    with pytest.raises(data.ChurnDataError, match="ARCHIVE_LIMIT_EXCEEDED"):
        scan(output.getvalue())


def test_loader_refuses_unrelated_synthetic_directory_without_network(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("loader attempted acquisition")

    monkeypatch.setattr(data, "download_churn_archive", no_network)
    root = tmp_path / "churn"
    root.mkdir()
    for name, value in payload_fixture().items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    with pytest.raises(data.ChurnDataError, match="DATA_MISSING"):
        data.load_churn(root)
    (tmp_path / data.CHURN_ARCHIVE_NAME).write_bytes(tar_fixture())
    with pytest.raises(data.ChurnDataError, match="DATA_INTEGRITY_MISMATCH"):
        data.load_churn(root)


def test_payload_loader_keeps_order_and_readonly_lineage():
    raw = data._raw_from_payloads(payload_fixture(), canonical=False)
    np.testing.assert_array_equal(raw.train.row_ids, [4, 1, 2, 3, 0, 5])
    np.testing.assert_array_equal(raw.train.numeric[:, 0], [8, 2, 4, 6, 0, 10])
    assert raw.dataset_id == "synthetic_test_v1"
    assert not raw.train.numeric.flags.writeable
    assert not raw.train.row_ids.flags.writeable
    assert not raw.y_train.flags.writeable
    with pytest.raises(data.ChurnDataError, match="SPLIT_INVALID"):
        data._raw_from_payloads(payload_fixture(), canonical=True)


def canonical_shape_fixture():
    """Synthetic shapes only; never satisfies the production archive hash gate."""
    return {
        "info.json": b'{"task":{"type":"binclass","score":"accuracy"}}',
        "x_num.npy": npy(np.arange(20_000, dtype=np.float32).reshape(10_000, 2)),
        "y.npy": npy(np.array([0, 1] * 5000, dtype=np.int64)),
        "splits/default/train.npy": npy(np.arange(0, 6400, dtype=np.int32)),
        "splits/default/val.npy": npy(np.arange(6400, 8000, dtype=np.int32)),
        "splits/default/test.npy": npy(np.arange(8000, 10_000, dtype=np.int32)),
    }


def test_extraction_publication_local_integrity_and_no_clobber(tmp_path, monkeypatch):
    # Stub only the previously tested cryptographic boundary, so tiny offline
    # fixtures exercise actual publication, local integrity and schema logic.
    payloads = canonical_shape_fixture()
    monkeypatch.setattr(data, "_canonical_payloads", lambda path, limits: payloads)
    root = tmp_path / "churn"
    data.extract_churn_archive(tmp_path / "unused.tar.gz", root)
    raw = data.load_churn(root)
    assert [getattr(raw, part).n_rows for part in data.SPLITS] == [6400, 1600, 2000]
    assert raw.inventory == data._inventory(payloads)
    with pytest.raises(data.ChurnDataError, match="DATA_DESTINATION_EXISTS"):
        data.extract_churn_archive(tmp_path / "unused.tar.gz", root)
    path = root / "y.npy"
    changed = bytearray(path.read_bytes())
    changed[-1] ^= 1
    path.write_bytes(changed)
    with pytest.raises(data.ChurnDataError, match="DATA_INTEGRITY_MISMATCH"):
        data.load_churn(root)
    path.write_bytes(payloads["y.npy"])
    (root / "receipt.json").write_text("{}")
    with pytest.raises(data.ChurnDataError, match="DATA_INTEGRITY_MISMATCH"):
        data.load_churn(root)


def test_failed_schema_never_publishes_directory(tmp_path, monkeypatch):
    payloads = canonical_shape_fixture()
    payloads["x_num.npy"] = npy(np.ones((1, 2), dtype=np.float32))
    monkeypatch.setattr(data, "_canonical_payloads", lambda path, limits: payloads)
    with pytest.raises(data.ChurnDataError, match="DATA_SCHEMA_INVALID"):
        data.extract_churn_archive(tmp_path / "unused.tar.gz", tmp_path / "churn")
    assert list(tmp_path.iterdir()) == []


def test_dataset_digest_covers_holdout_and_split_bytes():
    payloads = payload_fixture()
    first = data._raw_from_payloads(payloads, canonical=False)
    numeric = np.arange(24, dtype=np.float32).reshape(12, 2)
    numeric[-1] += 1
    payloads["x_num.npy"] = npy(numeric)
    second = data._raw_from_payloads(payloads, canonical=False)
    assert first.dataset_sha256 != second.dataset_sha256
    payloads["splits/default/test.npy"] = npy(np.array([11, 10, 9, 8], dtype=np.int32))
    third = data._raw_from_payloads(payloads, canonical=False)
    assert second.dataset_sha256 != third.dataset_sha256


def test_all_filesystem_path_components_reject_symlinks(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "y.npy").write_bytes(b"arbitrary")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit unprivileged symlink creation")
    with pytest.raises(data.ChurnDataError, match="DATA_PATH_UNSAFE"):
        data._read_regular(link / "y.npy", 100)


@pytest.mark.parametrize("indices", [
    np.array([4, 1, 2, 3, 0, 0], dtype=np.int32),
    np.array([-1, 1, 2, 3, 0, 5], dtype=np.int32),
    np.array([99, 1, 2, 3, 0, 5], dtype=np.int32),
    np.array([4, 1, 2, 3, 6, 5], dtype=np.int32),
    np.array([4, 1, 2, 3, 0], dtype=np.int32),
    np.array([4, 1, 2, 3, 0, 5], dtype=np.float32),
])
def test_invalid_splits_rejected(indices):
    payloads = payload_fixture()
    payloads["splits/default/train.npy"] = npy(indices)
    with pytest.raises(data.ChurnDataError, match="SPLIT_INVALID"):
        data._raw_from_payloads(payloads, canonical=False)


@pytest.mark.parametrize("name,value", [
    ("x_num.npy", np.full((12, 2), np.inf, dtype=np.float32)),
    ("x_num.npy", np.zeros((11, 2), dtype=np.float32)),
    ("x_num.npy", np.zeros((12, 2), dtype=np.float64)),
    ("x_cat.npy", np.array([[object()]] * 12, dtype=object)),
    ("x_bin.npy", np.full((12, 1), 2, dtype=np.float32)),
    ("y.npy", np.full(12, 2, dtype=np.int64)),
    ("y.npy", np.zeros(12, dtype=np.int64)),
])
def test_bad_schema_and_pickle_are_rejected(name, value):
    payloads = payload_fixture()
    payloads[name] = npy(value)
    with pytest.raises(data.ChurnDataError, match="DATA_SCHEMA_INVALID"):
        data._raw_from_payloads(payloads, canonical=False)


def test_npy_large_claim_or_trailing_bytes_rejected_before_allocation():
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, {"descr": "<f4", "fortran_order": False, "shape": (10**12, 100)}
    )
    with pytest.raises(data.ChurnDataError, match="DATA_SCHEMA_INVALID"):
        data._npy(stream.getvalue(), "large.npy")
    with pytest.raises(data.ChurnDataError, match="DATA_SCHEMA_INVALID"):
        data._npy(npy(np.zeros(1, dtype=np.float32)) + b"extra", "trailing.npy")


def leakage_fixture():
    blocks = {
        "train": data.FeatureBlock(
            np.array([[20, 10], [30, 20], [40, 30], [50, 40], [60, 50], [np.nan, 60.]]),
            np.array([["basic", "north"], ["basic", "north"], ["basic", "south"],
                      ["plus", "north"], ["plus", "south"], [None, None]], dtype=object),
            np.arange(6), ("age", "income"), ("plan", "region"),
        ),
        "val": data.FeatureBlock(
            np.array([[10000, -10000], [10001, -10001], [np.nan, np.nan]]),
            np.array([["validation_only", "north"], ["basic", "south"], [None, None]], dtype=object),
            np.arange(6, 9), ("age", "income"), ("plan", "region"),
        ),
        "test": data.FeatureBlock(
            np.array([[20000, -20000], [20001, -20001], [np.nan, np.nan]]),
            np.array([["test_only", "south"], ["plus", "north"], [None, None]], dtype=object),
            np.arange(9, 12), ("age", "income"), ("plan", "region"),
        ),
    }
    return data.RawChurn(**blocks, y_train=np.array([0, 1] * 3), y_val=np.array([1, 0, 1]),
                         y_test=np.array([0, 1, 0]), dataset_id="synthetic_leakage_v1")


@requires_sklearn
def test_train_only_medians_modes_unknown_blocks_and_finite_outputs():
    raw = leakage_fixture()
    prepared = data.prepare_data(raw)
    fitted = prepared.preprocessor
    assert fitted.numeric_medians == (40., 35.)
    assert fitted.categorical_modes == ("basic", "north")
    assert fitted.output_width == 6
    first_start, first_end = fitted.metadata["categorical"]["block_offsets"][0]
    np.testing.assert_array_equal(prepared.val[0, first_start:first_end], 0)
    np.testing.assert_array_equal(prepared.test[0, first_start:first_end], 0)
    for matrix in prepared.x.values():
        assert matrix.shape[1] == 6 and matrix.dtype == np.float32
        assert np.all(np.isfinite(matrix)) and matrix.flags.c_contiguous
    probe = data.FeatureBlock(np.array([[40., 35.]]), np.array([["basic", "north"]]),
                              numeric_names=("age", "income"), categorical_names=("plan", "region"))
    np.testing.assert_array_equal(fitted.transform(probe)[0], prepared.val[-1])
    np.testing.assert_array_equal(prepared.val[-1], prepared.test[-1])


@requires_sklearn
def test_orchestration_fits_train_only_and_ignores_holdout_labels(monkeypatch):
    raw = leakage_fixture()
    observed = []
    original_fit = data.fit_preprocessor

    def fit_spy(train, **kwargs):
        observed.append(train.row_ids.tolist())
        return original_fit(train, **kwargs)

    monkeypatch.setattr(data, "fit_preprocessor", fit_spy)
    first = data.prepare_data(raw)
    changed = dataclasses.replace(
        raw,
        val=dataclasses.replace(raw.val, numeric=np.full((3, 2), 35.),
                                categorical=np.full((3, 2), "holdout_replacement", dtype=object)),
        test=dataclasses.replace(raw.test, numeric=np.full((3, 2), 40.),
                                 categorical=np.full((3, 2), "different_holdout", dtype=object)),
        y_train=1 - raw.y_train, y_val=1 - raw.y_val, y_test=1 - raw.y_test,
    )
    second = data.prepare_data(changed)
    assert observed == [list(range(6)), list(range(6))]
    assert first.preprocessor.metadata == second.preprocessor.metadata
    assert first.preprocessor.fingerprint == second.preprocessor.fingerprint
    np.testing.assert_array_equal(first.train, second.train)
    before = first.preprocessor.metadata
    first.preprocessor.transform(raw.test)
    first.preprocessor.transform(raw.val)
    assert before == first.preprocessor.metadata
    assert "validation_only" not in json.dumps(before)
    assert "test_only" not in json.dumps(before)


@requires_sklearn
def test_independent_sklearn_oracle():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

    raw = leakage_fixture()
    median = SimpleImputer(strategy="median").fit(raw.train.numeric)
    quantiles = QuantileTransformer(n_quantiles=6, output_distribution="normal", subsample=6,
                                    random_state=0).fit(median.transform(raw.train.numeric))
    categorical = raw.train.categorical.copy()
    categorical[-1] = ["basic", "north"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32).fit(categorical)
    expected = np.column_stack((quantiles.transform(median.transform(raw.train.numeric)),
                                encoder.transform(categorical))).astype(np.float32)
    actual = data.prepare_data(raw)
    np.testing.assert_allclose(actual.train, expected, rtol=0, atol=0)


@requires_sklearn
@pytest.mark.parametrize("kind", ["numeric", "categorical"])
def test_all_missing_training_column_fails_even_if_holdout_is_observed(kind):
    raw = leakage_fixture()
    block = dataclasses.replace(raw.train, **{kind: np.full((6, 2), np.nan if kind == "numeric" else None)})
    with pytest.raises(data.ChurnDataError, match="ALL_MISSING_TRAIN_COLUMN"):
        data.prepare_data(dataclasses.replace(raw, train=block))


@requires_sklearn
def test_numeric_only_categorical_only_constant_and_mode_tie():
    numeric = data.FeatureBlock(numeric=np.array([[1.], [1.], [np.nan], [1.]]))
    transformed = data.fit_preprocessor(numeric).transform(numeric)
    assert transformed.shape == (4, 1) and np.all(np.isfinite(transformed))
    categorical = data.FeatureBlock(categorical=np.array([[10], [2], [None]], dtype=object))
    fitted = data.fit_preprocessor(categorical)
    assert fitted.categorical_modes == (2,)
    assert fitted.transform(categorical).shape == (3, 2)
    literal = data.FeatureBlock(categorical=np.array([["nan"], [""], [None]], dtype=object))
    assert data.fit_preprocessor(literal).categorical_modes == ("",)


@requires_sklearn
def test_fingerprint_includes_quantiles_not_only_medians():
    first = data.FeatureBlock(numeric=np.array([[0.], [2.], [4.]]))
    second = data.FeatureBlock(numeric=np.array([[-20.], [2.], [40.]]))
    assert data.fit_preprocessor(first).numeric_medians == data.fit_preprocessor(second).numeric_medians
    assert data.fit_preprocessor(first).fingerprint != data.fit_preprocessor(second).fingerprint


@requires_sklearn
def test_feature_schema_mismatch_and_lineage_contamination_fail():
    raw = leakage_fixture()
    fitted = data.fit_preprocessor(raw.train)
    with pytest.raises(data.ChurnDataError, match="PREPROCESS_STATE_MISMATCH"):
        fitted.transform(dataclasses.replace(raw.val, numeric_names=("income", "age")))
    with pytest.raises(data.ChurnDataError, match="SPLIT_INVALID"):
        data.prepare_data(dataclasses.replace(raw, val=dataclasses.replace(raw.val, row_ids=np.array([0, 7, 8]))))


def test_explicit_fixture_has_distinct_identity_and_disjoint_lineage():
    raw = data.make_synthetic_fixture()
    assert raw.dataset_id == "synthetic_smoke_v1"
    assert [getattr(raw, part).n_rows for part in data.SPLITS] == [32, 12, 12]
    assert len(set(np.concatenate(tuple(raw.source_indices.values())))) == 56
    assert "violet" not in raw.train.categorical
    assert data.make_synthetic_fixture().dataset_sha256 == raw.dataset_sha256


def test_source_document_matches_pinned_constants():
    source = json.loads((Path(__file__).parents[1] / "docs" / "data-source.json").read_text())
    assert source["archive"]["url"] == data.CHURN_ARCHIVE_URL
    assert source["archive"]["sha256"] == data.CHURN_ARCHIVE_SHA256
    assert source["archive"]["size_bytes"] == data.CHURN_ARCHIVE_SIZE
    assert set(source["allowed_members"]) == set(data.CHURN_MEMBER_PATHS)
