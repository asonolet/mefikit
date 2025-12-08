from mefikit import UMesh

type_order = [
    "VERTEX",
    "SEG2",
    "SEG3",
    "TRI3",
    "TRI6",
    "QUAD4",
    "QUAD8",
    "TET4",
    "TET10",
    "HEX8",
    "HEX20",
]

meshio_to_mefikit_type = {
    "vertex": "VERTEX",
    "line": "SEG2",
    "line3": "SEG3",
    "triangle": "TRI3",
    "triangle6": "TRI6",
    "quad": "QUAD4",
    "quad8": "QUAD8",
    "tetra": "TET4",
    "tetra10": "TET10",
    "hexahedron": "HEX8",
    "hexahedron20": "HEX20",
}
mefikit_to_meshio_type = {v: k for k, v in meshio_to_mefikit_type.items()}


def to_meshio(umesh: UMesh):
    import meshio

    blocks = umesh.blocks()  # type: dict

    cells = {mefikit_to_meshio_type[t]: b for (t, b) in blocks.items()}

    return meshio.Mesh(
        umesh.coords(),
        cells,
    )


UMesh.to_meshio = to_meshio
