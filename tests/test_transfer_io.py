from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

import mefikit as mf


def mixed_source() -> mf.UMesh:
    """2D source with QUAD4 and TRI3 blocks, in a deliberately mixed layout."""
    src = mf.UMesh(
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 2.0], [1.0, 2.0]]
        )
    )
    src.add_regular_block("QUAD4", np.array([[0, 1, 2, 3]], dtype=np.uint))
    src.add_regular_block("TRI3", np.array([[2, 3, 4], [4, 5, 2]], dtype=np.uint))
    src.set_field("T", {"QUAD4": np.array([[1.0]]), "TRI3": np.array([[2.0], [3.0]])})
    return src


def target_grid() -> mf.UMesh:
    return mf.build_cmesh(np.linspace(0.1, 0.9, 3), np.linspace(0.1, 1.9, 3))


def transfer_factories():
    return [
        mf.transfer.ConstantPiecewise,
        lambda s, t, **kw: mf.transfer.MovingLeastSquares(s, t, k=4, **kw),
        lambda s, t, **kw: mf.transfer.InverseDistance(s, t, k=4, **kw),
        mf.transfer.ConservativeP0,
    ]


@pytest.mark.parametrize("make", transfer_factories())
def test_roundtrip_all_methods(tmp_path, make):
    src = mixed_source()
    tgt = target_grid()
    trsf = make(src, tgt, def_val=np.nan)

    path = str(tmp_path / "operator.npz")
    trsf.save(path)
    loaded = mf.transfer.load(path, src, tgt)

    assert type(loaded) is type(trsf)
    before = trsf.eval("T")
    after = loaded.eval("T")
    assert before.keys() == after.keys()
    for etype in before:
        np.testing.assert_allclose(
            np.asarray(before[etype]), np.asarray(after[etype]), equal_nan=True
        )


def test_downcast_roundtrip(tmp_path):
    src = mf.build_cmesh(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    src.set_field("T", {"HEX8": np.array([[5.0]])})
    tgt = mf.UMesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    )
    tgt.add_regular_block("QUAD4", np.array([[0, 1, 2, 3]], dtype=np.uint))

    trsf = mf.transfer.ConservativeP0(src, tgt)
    path = str(tmp_path / "downcast.npz")
    trsf.save(path)
    loaded = mf.transfer.load(path, src, tgt)

    np.testing.assert_allclose(
        np.asarray(loaded.eval("T")["QUAD4"]).ravel(),
        np.asarray(trsf.eval("T")["QUAD4"]).ravel(),
    )


def test_saved_matrix_is_scipy_compatible(tmp_path):
    sp = pytest.importorskip("scipy.sparse")
    src = mixed_source()
    tgt = target_grid()
    trsf = mf.transfer.ConservativeP0(src, tgt)
    path = str(tmp_path / "operator.npz")
    trsf.save(path)

    matrix = sp.load_npz(path)
    assert matrix.shape == (tgt.num_elements(), src.num_elements())
    assert matrix.nnz > 0
    assert np.all(matrix.indices < src.num_elements())

    with np.load(path) as archive:
        meta = json.loads(archive["mefikit_meta"].tobytes())
    # Blocks are written name-sorted, not in ElementType declaration order.
    assert [block["element_type"] for block in meta["source_blocks"]] == [
        "QUAD4",
        "TRI3",
    ]
    assert [block["element_type"] for block in meta["target_blocks"]] == ["QUAD4"]


def test_uncompressed_roundtrip(tmp_path):
    src = mixed_source()
    tgt = target_grid()
    trsf = mf.transfer.InverseDistance(src, tgt, k=3, def_val=-7.0)
    path = str(tmp_path / "plain.npz")
    trsf.save(path, compressed=False)

    with zipfile.ZipFile(path) as archive:
        assert all(
            item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()
        )

    loaded = mf.transfer.load(path, src, tgt)
    np.testing.assert_allclose(
        np.asarray(loaded.eval("T")["QUAD4"]),
        np.asarray(trsf.eval("T")["QUAD4"]),
        equal_nan=True,
    )


def test_default_value_is_restored(tmp_path):
    src = mf.build_cmesh(np.array([0.0, 0.5]), np.array([0.0, 0.5]))
    tgt = mf.build_cmesh(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    src.set_field("T", {"QUAD4": np.ones((1, 1))})
    trsf = mf.transfer.ConservativeP0(src, tgt, def_val=np.nan)

    path = str(tmp_path / "default.npz")
    trsf.save(path)
    loaded = mf.transfer.load(path, src, tgt)

    values = np.asarray(loaded.eval("T")["QUAD4"]).ravel()
    assert np.isnan(values).any()
    assert np.any(np.isclose(values, 1.0))


def test_load_rejects_wrong_source_or_target_mesh(tmp_path):
    src = mixed_source()
    tgt = target_grid()
    trsf = mf.transfer.ConservativeP0(src, tgt)
    path = str(tmp_path / "operator.npz")
    trsf.save(path)

    bigger = mf.build_cmesh(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    with pytest.raises(ValueError, match="block"):
        mf.transfer.load(path, bigger, tgt)
    with pytest.raises(ValueError, match="block"):
        mf.transfer.load(path, src, bigger)


def test_load_rejects_plain_scipy_npz(tmp_path):
    sp = pytest.importorskip("scipy.sparse")
    path = str(tmp_path / "plain.npz")
    sp.save_npz(path, sp.csr_matrix((2, 2)))
    with pytest.raises(ValueError, match="missing array"):
        mf.transfer.load(path, mixed_source())


def _rewrite_archive(source: str, destination: str, update) -> None:
    with np.load(source) as archive:
        arrays = {name: archive[name] for name in archive.files}
    update(arrays)
    np.savez_compressed(destination, **arrays)


def test_load_rejects_out_of_range_source_index(tmp_path):
    src = mixed_source()
    tgt = target_grid()
    path = str(tmp_path / "operator.npz")
    mf.transfer.InverseDistance(src, tgt, k=2).save(path)

    def corrupt(arrays):
        arrays["indices"] = arrays["indices"].copy()
        arrays["indices"][0] = 999

    bad = str(tmp_path / "bad-index.npz")
    _rewrite_archive(path, bad, corrupt)
    with pytest.raises(ValueError, match="out of range"):
        mf.transfer.load(bad, src, tgt)


def test_load_rejects_unsupported_version(tmp_path):
    src = mixed_source()
    tgt = target_grid()
    path = str(tmp_path / "operator.npz")
    mf.transfer.ConstantPiecewise(src, tgt).save(path)

    def bump_version(arrays):
        meta = json.loads(arrays["mefikit_meta"].tobytes())
        meta["version"] = 99
        arrays["mefikit_meta"] = np.frombuffer(
            json.dumps(meta).encode(), dtype=np.uint8
        )

    bad = str(tmp_path / "bad-version.npz")
    _rewrite_archive(path, bad, bump_version)
    with pytest.raises(ValueError, match="version"):
        mf.transfer.load(bad, src, tgt)
