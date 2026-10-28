import numpy as np

import mefikit as mf


def make_triangle_mesh(n: int = 6) -> mf.UMesh:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    m = mf.UMesh(coords)
    m.add_regular_block("TRI3", np.array([[0, 2, 1], [1, 2, 3]], dtype=np.uint))
    md = m.duplicate(mf.Transform.translation([1.0, 0.0]), n - 1)
    return md.merge_nodes()


def make_wedge_mesh(z, n: int = 6) -> mf.UMesh:
    return make_triangle_mesh(n).extrude(z)


def test_wedge_measure():
    w = make_wedge_mesh([0.0, 1.0])
    w2 = w.translate([0.5, 0.0])

    tr = mf.transfer.ConservativeP0(w, w, def_val=0.0)
    w.fields["Mt"] = (tr(mf.M) / mf.M - 1.0) * 100.0
    w.to_pyvista().plot(show_edges=True)

    tr2 = mf.transfer.ConservativeP0(w, w2, def_val=0.0)
    w2.fields["Mt"] = tr2(mf.M)
    w2.to_pyvista().plot(show_edges=True)


if __name__ == "__main__":
    test_wedge_measure()
