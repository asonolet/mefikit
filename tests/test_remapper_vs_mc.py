from __future__ import annotations

import medcoupling as mc
import numpy as np

import mefikit as mf

N_SUBSET = 256


def mc_field(
    mmesh: mc.MEDCouplingUMesh,
    vals: np.ndarray,
    nature: int = mc.IntensiveConservation,
    name="Measure",
):
    f = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    a = mc.DataArrayDouble(np.ascontiguousarray(vals.ravel()))
    a.setName(name)
    f.setArray(a)
    f.setMesh(mmesh)
    f.setNature(nature)
    return f


def test_remap():
    """mefikit ConservativeP0 transfer matches medcoupling P0P0 on real meshes.

    The remap runs on a deterministic cell subset of the reference meshes
    because medcoupling's PDE-based `prepare` is quadratic in the number of
    cells (a few seconds for 2000-cell files); building both sides from the
    very same subset keeps the geometry comparison exact.
    """
    mesh_files = ["mesh_27.med", "mesh_36.med"]

    for i, mfn_src in enumerate(mesh_files):
        src_path = "tests/data/" + mfn_src
        # mefikit's own reader on the reference file stays covered at low cost
        assert mf.UMesh.read(src_path).num_elements() == 2000
        for mfn_tgt in mesh_files[i:]:
            mc_src = mc.ReadMeshFromFile(src_path, 0)
            mc_tgt = mc.ReadMeshFromFile("tests/data/" + mfn_tgt, 0)

            cells = list(range(min(N_SUBSET, mc_src.getNumberOfCells())))
            s = mc_src.buildPartOfMySelf(cells)
            t = mc_tgt.buildPartOfMySelf(cells)

            mf_src = mf.UMesh.from_mc(s).reorient()
            mf_tgt = mf.UMesh.from_mc(t).reorient()

            mf_src.fields["Measure"] = mf.M
            mes_npy = mf_src.fields["Measure"].numpy()

            vt = mc.MEDCouplingRemapper()
            vt.prepare(s, t, "P0P0")
            mcf_remapped = vt.transferField(mc_field(s, mes_npy), 0.0)
            tgt_npy = mcf_remapped.getArray().toNumPyArray()

            trsf = mf.transfer.ConservativeP0(mf_src, mf_tgt)
            trsf.apply_update(mf_src, "Measure", mf_tgt, def_val=0.0)
            mff_remapped = mf_tgt.fields["Measure"].numpy()

            assert tgt_npy.shape == mff_remapped.shape
            assert np.allclose(tgt_npy, mff_remapped)


if __name__ == "__main__":
    test_remap()
