from __future__ import annotations

import os
import time

import medcoupling as mc
import numpy as np

import mefikit as mf


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


def mc_remap(mc_src, mc_tgt, field_npy) -> tuple[float, float, np.ndarray]:
    mcf = mc_field(mc_src, field_npy)

    t0 = time.time()
    vt = mc.MEDCouplingRemapper()
    vt.prepare(mc_src, mc_tgt, "P0P0")
    t1 = time.time()
    mcf_tgt = vt.transferField(mcf, 0.0)
    t2 = time.time()

    tgt_npy = mcf_tgt.getArray().toNumPyArray()
    return t1 - t0, t2 - t1, tgt_npy


def mf_remap(mf_src, mf_tgt) -> tuple[float, float, np.ndarray]:
    t0 = time.time()
    trsf = mf.transfer.ConservativeP0(mf_src, mf_tgt)
    t1 = time.time()
    trsf.apply_update(mf_src, "Measure", mf_tgt, def_val=0.0)
    t2 = time.time()
    return t1 - t0, t2 - t1, mf_tgt.fields["Measure"].numpy()


# --- remap timing + validation ---------------------------------------------
def test_remap():
    """Test medcoupling remapper vs mefikit transfer and bench."""
    mesh_files = os.listdir("tests/data")
    mc_preps = []
    mc_applies = []
    mf_preps = []
    mf_applies = []

    for i, mfn_src in enumerate(mesh_files):
        for mfn_tgt in mesh_files[i:]:
            print("Testing", mfn_src, mfn_tgt)

            mf_src = mf.UMesh.read("tests/data/" + mfn_src)
            mf_tgt = mf.UMesh.read("tests/data/" + mfn_tgt)

            mf_src = mf_src.reorient()
            mf_tgt = mf_tgt.reorient()

            mc_src = mc.ReadMeshFromFile("tests/data/" + mfn_src, 0)
            mc_tgt = mc.ReadMeshFromFile("tests/data/" + mfn_tgt, 0)

            mf_src.fields["Measure"] = mf.M
            mes_npy = mf_src.fields["Measure"].numpy()

            mc_prep, mc_apply, mcf_remapped = mc_remap(mc_src, mc_tgt, mes_npy)
            print("MEDCoupling prepare: ", mc_prep)
            print("MEDCoupling apply  : ", mc_apply)
            mf_prep, mf_apply, mff_remapped = mf_remap(mf_src, mf_tgt)
            print("Mefikit prepare    : ", mf_prep)
            print("Mefikit apply      : ", mf_apply)
            mc_preps.append(mc_prep)
            mc_applies.append(mc_apply)
            mf_preps.append(mf_prep)
            mf_applies.append(mf_apply)

            allclose = np.allclose(mcf_remapped, mff_remapped)
            if not allclose:
                mf_tgt.fields["mc_remapped"] = mcf_remapped
                mf_tgt.fields["diff"] = mf.Field("Measure") - mf.Field("mc_remapped")
                diff_cells = np.abs(mcf_remapped - mff_remapped) > 1e-9
                print(np.where(diff_cells))
                print("mc : ", mcf_remapped[diff_cells])
                print("mf : ", mff_remapped[diff_cells])
                print("mes: ", mes_npy[diff_cells])
                mc_tgt.write(f"diff_{mfn_src[:-4]}_{mfn_tgt[:-4]}.med")
                cell = mc_tgt.buildPartOfMySelf([80])
                cell.zipCoords()
                cell.write(f"cell80_{mfn_tgt[:-4]}.med")
                # mc.WriteMesh(f"diff_{mfn_src[:-4]}_{mfn_tgt[:-4]}.med", mc_tgt, True)
                # mc.WriteField(f"diff_{mfn_src[:-4]}_{mfn_tgt[:-4]}.med", )
                break
        break
    print("MEDCoupling prepare: ", np.mean(mc_preps))
    print("MEDCoupling apply  : ", np.mean(mc_applies))
    print("Mefikit prepare    : ", np.mean(mf_preps))
    print("Mefikit apply      : ", np.mean(mf_applies))


if __name__ == "__main__":
    test_remap()
