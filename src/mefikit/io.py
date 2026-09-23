from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

type_order = [
    "VERTEX",
    "SEG2",
    "SEG3",
    "TRI3",
    "TRI6",
    "QUAD4",
    "QUAD8",
    "PGON",
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
    "PGON": 2,
    "TET4": 3,
    "TET10": 3,
    "HEX8": 3,
    "HEX20": 3,
    "PHED": 3,
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


def install_conversions():
    import medcoupling as mc
    import meshio
    import pyvista as pv

    from mefikit import UMesh

    def to_meshio(self: UMesh) -> meshio.Mesh:
        blocks: dict = self.blocks()

        cells = {mefikit_to_meshio_type[t]: b for (t, b) in blocks.items()}

        return meshio.Mesh(
            self.coords(),
            cells,
        )

    def to_mc(self: UMesh, lev=None) -> mc.MEDCouplingUMesh:
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

        # Node permutations to convert VTK ordering to MED ordering.
        _mf_permutations = {
            "TET4": [0, 1, 3, 2],
            "TET10": [0, 1, 3, 2, 4, 8, 7, 6, 5, 9],
            "HEX8": [4, 5, 6, 7, 0, 1, 2, 3],
        }

        def _mf_reg_to_mc_connectivity(et: str, conn: npt.NDArray[np.uintp]):
            num_nodes = conn.shape[1]
            n_elem = conn.shape[0]

            if et in _mf_permutations:
                conn = conn[:, _mf_permutations[et]]

            new_connectivity = np.insert(
                conn.flatten(), np.arange(n_elem) * num_nodes, mf_types_mc_id[et]
            )
            # MEDCoupling nodal-connectivity index has one entry per cell plus a
            # trailing end offset, i.e. n_elem + 1 elements. Omitting the last
            # entry silently drops the final cell.
            offsets = np.arange(n_elem + 1, dtype=int) * (num_nodes + 1)
            return new_connectivity, offsets

        blocks = self.blocks()
        coords = self.coords()

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
        et = next(iter(blocks.keys()))
        res = mc.MEDCouplingUMesh("mf_UMesh", mf_types_dim[et])
        res.setCoords(mc.DataArrayDouble(coords))
        res.setConnectivity(mc.DataArrayInt(mc_conn), mc.DataArrayInt(mc_offset))
        return res

    def to_pyvista(
        self: UMesh,
        dim: str | int | None = None,
        with_fields: Sequence[str] | bool = True,
    ) -> pv.UnstructuredGrid:
        blocks = self.blocks()
        coords = self.coords()
        if not with_fields:
            fields = {}
        else:
            fields = {n: ref.values() for n, ref in self.fields.items()}
            if not isinstance(with_fields, bool):
                fields = {n: f for n, f in fields.items() if n in with_fields}

        mf_types_to_pv = {
            "VERTEX": pv.CellType.VERTEX,
            "SEG2": pv.CellType.LINE,
            "SEG3": pv.CellType.QUADRATIC_EDGE,
            "TRI3": pv.CellType.TRIANGLE,
            "TRI6": pv.CellType.QUADRATIC_TRIANGLE,
            "QUAD4": pv.CellType.QUAD,
            "QUAD8": pv.CellType.QUADRATIC_QUAD,
            "PGON": pv.CellType.POLYGON,
            "TET4": pv.CellType.TETRA,
            "TET10": pv.CellType.QUADRATIC_TETRA,
            "HEX8": pv.CellType.HEXAHEDRON,
            "HEX20": pv.CellType.QUADRATIC_HEXAHEDRON,
            "PHED": pv.CellType.POLYHEDRON,
        }

        if dim is None:
            dim = max(mf_types_dim[et] for et in blocks)

        def _mf_reg_to_pv_connectivity(et: str, conn: np.ndarray):
            num_nodes = conn.shape[1]
            n_elem = conn.shape[0]

            new_connectivity = np.insert(
                conn.flatten(), np.arange(n_elem) * num_nodes, num_nodes
            )
            elems_type = np.full(n_elem, mf_types_to_pv[et], dtype=np.intp)
            return new_connectivity, elems_type

        def _mf_poly_to_pv_connectivity(et: str, conn: np.ndarray, offsets: np.ndarray):
            n_elem = offsets.shape[0]
            offsets = offsets.astype(np.intp)
            starts = np.concatenate(([0], offsets[:-1]))
            num_nodes = offsets - starts
            new_connectivity = np.insert(
                conn.reshape(-1).astype(np.intp), starts, num_nodes
            )
            elems_type = np.full(n_elem, mf_types_to_pv[et], dtype=np.intp)
            return new_connectivity, elems_type

        def _mf_phed_to_pv_connectivity(
            conn: np.ndarray, offsets: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            # Polyhedral connectivity is a flat node id array split into cells
            # by `offsets` (cumulative ends, no leading zero). Within each cell
            # the faces are separated by the `usize::MAX` sentinel. Each cell is
            # emitted as [n_entries, n_faces, nv_0, ids..., nv_1, ids..., ...]
            # which is the pyvista/VTK polyhedron layout.
            n_cells = offsets.shape[0]
            if n_cells == 0:
                return np.array([], dtype=int), np.array([], dtype=int)

            data = np.asarray(conn, dtype=np.uintp)
            offsets = offsets.astype(np.intp)
            max_id = np.iinfo(np.uintp).max
            sent = data == max_id
            node_rank = np.cumsum(~sent, dtype=np.intp) - 1
            nodes = data[~sent].astype(np.int64)

            # `offsets` are cumulative cell ends, so `cell_starts` slice each
            # cell inside `data`. A face starts right after a sentinel or at a
            # cell start (cells may carry a trailing sentinel).
            n_nodes_per_cell = np.diff(np.r_[-1, node_rank[offsets - 1]])
            cell_starts = np.concatenate(([0], offsets[:-1]))

            is_face_start = np.zeros(sent.size, dtype=bool)
            is_face_start[1:] = sent[:-1]
            is_face_start[cell_starts] = True
            face_pos = np.flatnonzero(is_face_start)
            n_faces_per_cell = np.bincount(
                np.searchsorted(offsets, face_pos, side="right"), minlength=n_cells
            )
            cell_spec_len = 1 + n_faces_per_cell + n_nodes_per_cell

            # Face node counts are the distances between consecutive face starts
            # in the (ordered) node stream.
            face_start_node = node_rank[face_pos]
            face_n_nodes = np.diff(np.r_[face_start_node, nodes.size])

            cell_first_node = node_rank[cell_starts]

            # Three token kinds land at node positions: the cell entry length and
            # face count at each cell start, then each face's node count at the
            # face start. Tokens sharing a position keep plan order
            # (entry length < face count < face node count).
            insert_pos = np.concatenate(
                [cell_first_node, cell_first_node, face_start_node]
            )
            insert_val = np.concatenate([cell_spec_len, n_faces_per_cell, face_n_nodes])
            insert_prio = np.concatenate(
                [
                    np.zeros_like(cell_first_node),
                    np.ones_like(cell_first_node),
                    np.full_like(face_start_node, 2),
                ]
            )

            order = np.lexsort((insert_prio, insert_pos))
            insert_pos = insert_pos[order]
            insert_val = insert_val[order]

            n_insert = insert_pos.size
            insert_count = np.bincount(insert_pos, minlength=nodes.size)
            insert_cum = np.cumsum(insert_count)
            cells = np.empty(nodes.size + n_insert, dtype=np.intp)
            cells[np.arange(nodes.size) + insert_cum] = nodes
            insert_base = insert_pos + insert_cum[insert_pos] - insert_count[insert_pos]
            group_first = np.concatenate([[0], np.flatnonzero(np.diff(insert_pos)) + 1])
            in_group = np.arange(n_insert) - np.repeat(
                group_first, np.diff(np.r_[group_first, n_insert])
            )
            cells[insert_base + in_group] = insert_val

            elems_type = np.full(n_cells, mf_types_to_pv["PHED"], dtype=np.intp)
            return cells, elems_type

        conns = []
        et_typess = []
        fields_dict = {f: [] for f in fields}
        for et in [*type_order, "PHED"]:
            if et not in blocks or (dim != "all" and mf_types_dim[et] != dim):
                continue

            if et == "PHED":
                conn, offsets = blocks[et]
                conn, et_types = _mf_phed_to_pv_connectivity(conn, offsets)
            elif et == "PGON":
                conn, offsets = blocks[et]
                conn, et_types = _mf_poly_to_pv_connectivity(et, conn, offsets)
            else:
                conn, et_types = _mf_reg_to_pv_connectivity(et, blocks[et])
            conns.append(conn)
            et_typess.append(et_types)
            for f, v in fields.items():
                fields_dict[f].append(v[et])

        pv_conn = conns[0] if len(conns) == 1 else np.hstack(conns, dtype=int)
        pv_et_types = (
            et_typess[0] if len(et_typess) == 1 else np.hstack(et_typess, dtype=int)
        )

        pv_fields_dict = {f: np.hstack(fields_dict[f], dtype=float) for f in fields}

        if coords.shape[1] == 1:
            pv_coords = np.hstack((coords, np.zeros((coords.shape[0], 2))))
        elif coords.shape[1] == 2:
            pv_coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
        else:
            pv_coords = coords

        res = pv.UnstructuredGrid(pv_conn, pv_et_types, pv_coords)
        for f, v in pv_fields_dict.items():
            res.cell_data[f] = v
        return res

    UMesh.to_meshio = to_meshio
    UMesh.to_mc = to_mc
    UMesh.to_pyvista = to_pyvista
