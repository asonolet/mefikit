import numpy as np

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


mf_types_dim = {
    "VERTEX": 0,
    "SEG2": 1,
    "SEG3": 1,
    "TRI3": 2,
    "TRI6": 2,
    "QUAD4": 2,
    "QUAD8": 2,
    "TET4": 3,
    "TET10": 3,
    "HEX8": 3,
    "HEX20": 3,
}


mf_types_num_node = {
    "VERTEX": 1,
    "SEG2": 2,
    "SEG3": 3,
    "TRI3": 3,
    "TRI6": 6,
    "QUAD4": 4,
    "QUAD8": 8,
    "TET4": 4,
    "TET10": 10,
    "HEX8": 8,
    "HEX20": 20,
}


def to_meshio(umesh: UMesh):
    import meshio

    blocks = umesh.blocks()  # type: dict

    cells = {mefikit_to_meshio_type[t]: b for (t, b) in blocks.items()}

    return meshio.Mesh(
        umesh.coords(),
        cells,
    )


UMesh.to_meshio = to_meshio


def to_mc(umesh: UMesh, lev=None):
    import medcoupling as mc

    mf_types_mc_id = {
        "VERTEX": 0,
        "SEG2": 1,
        "SEG3": 2,
        "TRI3": 3,
        "TRI6": 6,
        "QUAD4": 4,
        "QUAD8": 8,
        "TET4": 14,
        "TET10": 20,
        "HEX8": 18,
        "HEX20": 30,
    }

    def _mf_reg_to_mc_connectivity(et: str, conn: np.ndarray):
        num_nodes = conn.shape[1]
        n_elem = conn.shape[0]

        new_connectivity = np.insert(
            conn.flatten(), np.arange(n_elem) * num_nodes, mf_types_mc_id[et]
        )
        offsets = np.arange(n_elem, dtype=int) * (num_nodes + 1)
        return new_connectivity, offsets

    blocks = umesh.blocks()
    coords = umesh.coords()

    mc_conn = np.array([], dtype=int)
    mc_offset = np.array([], dtype=int)

    if lev is None:
        lev = max(mf_types_dim[et] for et in blocks)

    for et in type_order:
        if et not in blocks or mf_types_dim[et] != lev:
            continue

        conn, offset = _mf_reg_to_mc_connectivity(et, blocks[et])
        mc_conn = np.hstack((mc_conn, conn), dtype=int)
        # offsets the offset
        if len(mc_offset) > 0:
            offset += mc_offset[-1]
            offset = offset[1:]
        mc_offset = np.r_[mc_offset, offset]
    res = mc.MEDCouplingUMesh()
    res.setCoords(mc.DataArrayDouble(coords))
    res.setConnectivity(mc.DataArrayInt(mc_conn), mc.DataArrayInt(mc_offset))
    res.setName("mf_UMesh")
    return res


UMesh.to_mc = to_mc


def to_pyvista(umesh: UMesh):
    import pyvista as pv

    blocks = umesh.blocks()
    coords = umesh.coords()

    pv_conn = np.array([], dtype=int)
    pv_et_types = np.array([], dtype=int)

    mf_types_to_pv = {
        "VERTEX": pv.CellType.VERTEX,
        "SEG2": pv.CellType.WEDGE,
        "SEG3": pv.CellType.QUADRATIC_EDGE,
        "TRI3": pv.CellType.TRIANGLE,
        "TRI6": pv.CellType.QUADRATIC_TRIANGLE,
        "QUAD4": pv.CellType.QUAD,
        "QUAD8": pv.CellType.QUADRATIC_QUAD,
        "TET4": pv.CellType.TETRA,
        "TET10": pv.CellType.QUADRATIC_TETRA,
        "HEX8": pv.CellType.HEXAHEDRON,
        "HEX20": pv.CellType.QUADRATIC_HEXAHEDRON,
    }

    def _mf_reg_to_pv_connectivity(et: str, conn: np.ndarray):
        num_nodes = conn.shape[1]
        n_elem = conn.shape[0]

        new_connectivity = np.insert(
            conn.flatten(), np.arange(n_elem) * num_nodes, num_nodes
        )
        elems_type = np.array([mf_types_to_pv[et]] * n_elem)
        return new_connectivity, elems_type

    for et in type_order:
        if et not in blocks:
            continue

        conn, et_types = _mf_reg_to_pv_connectivity(et, blocks[et])
        pv_conn = np.hstack((pv_conn, conn), dtype=int)
        pv_et_types = np.hstack((pv_et_types, et_types), dtype=int)

    if coords.shape[1] == 1:
        pv_coords = np.hstack((coords, np.zeros((coords.shape[0], 2))))
    elif coords.shape[1] == 2:
        pv_coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
    else:
        pv_coords = coords

    res = pv.UnstructuredGrid(pv_conn, pv_et_types, pv_coords)
    return res


UMesh.to_pyvista = to_pyvista
