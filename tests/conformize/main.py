# /// script
# requires-python = ">=3.12"
# dependencies = ["mefikit[io]"]
# ///

import mefikit as mf


def main() -> None:
    m = mf.UMesh.read("./Mesh_non_conforme_simplify.med")
    m3 = m.select(mf.sel.types(["PHED"])).to_mesh()
    # m3.to_pyvista().shrink(0.9).plot(show_edges=True)
    print(m3.blocks()["PHED"])
    print(mf.is_conform(m3))
    m3b = mf.conformize(m3)

    m3b.to_pyvista().shrink(0.9).plot(show_edges=True)
    # print(m3b.blocks()["PHED"])
    # m3b.write("conformized_mesh_non_conforme.med")


if __name__ == "__main__":
    main()
