from __future__ import annotations

import warnings
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
        }

        if dim is None:
            dim = max(mf_types_dim[et] for et in blocks)

        def _mf_reg_to_pv_connectivity(et: str, conn: np.ndarray):
            num_nodes = conn.shape[1]
            n_elem = conn.shape[0]

            new_connectivity = np.insert(
                conn.flatten(), np.arange(n_elem) * num_nodes, num_nodes
            )
            elems_type = np.array([mf_types_to_pv[et]] * n_elem)
            return new_connectivity, elems_type

        def _mf_poly_to_pv_connectivity(et: str, conn: np.ndarray, offsets: np.ndarray):
            n_elem = offsets.shape[0]
            offsets = offsets.astype(int)

            num_nodes = np.r_[offsets[0], offsets[1:] - offsets[:-1]]
            pos = np.r_[0, offsets[:-1]]

            new_connectivity = np.insert(conn.flatten(), pos, num_nodes)
            elems_type = np.array([mf_types_to_pv[et]] * n_elem)
            return new_connectivity, elems_type

        conns = []
        et_typess = []
        fields_dict = {f: [] for f in fields}
        for et in type_order:
            if et not in blocks or (dim != "all" and mf_types_dim[et] != dim):
                continue

            if et == "PGON":
                conn, offsets = blocks[et]
                conn, et_types = _mf_poly_to_pv_connectivity(et, conn, offsets)
            else:
                conn, et_types = _mf_reg_to_pv_connectivity(et, blocks[et])
            conns.append(conn)
            et_typess.append(et_types)
            for f, v in fields.items():
                fields_dict[f].append(v[et])

        pv_conn = np.hstack(conns, dtype=int)
        pv_et_types = np.hstack(et_typess, dtype=int)

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

    def from_mc(cls, mesh, fields=None):
        if not isinstance(mesh, (mc.MEDFileUMesh, mc.MEDCouplingUMesh)):
            raise TypeError(
                "Expected a MEDCouplingUMesh or MEDFileUMesh, got "
                f"{type(mesh).__name__}"
            )

        if isinstance(fields, mc.MEDCouplingFieldDouble):
            fields = (fields,)

        mf_types_num_node = {
            "VERTEX": 1,
            "SEG2": 2,
            "SEG3": 3,
            "SEG4": 4,
            "TRI3": 3,
            "TRI6": 6,
            "TRI7": 7,
            "QUAD4": 4,
            "QUAD8": 8,
            "QUAD9": 9,
            "TET4": 4,
            "TET10": 10,
            "HEX8": 8,
        }

        _mf_permutations = {
            "TET4": (0, 1, 3, 2),
            "TET10": (0, 1, 3, 2, 4, 8, 7, 6, 5, 9),
            "HEX8": (4, 5, 6, 7, 0, 1, 2, 3),
        }

        _mc_codes = {
            0: "VERTEX",
            1: "SEG2",
            2: "SEG3",
            10: "SEG4",
            3: "TRI3",
            6: "TRI6",
            7: "TRI7",
            4: "QUAD4",
            8: "QUAD8",
            9: "QUAD9",
            14: "TET4",
            20: "TET10",
            18: "HEX8",
            5: "PGON",
            31: "PHED",
        }
        known_codes = np.array(sorted(_mc_codes), dtype=np.int64)
        known_types = [_mc_codes[int(c)] for c in known_codes]
        poly_types = {"PGON", "PHED"}

        def _type_labels(codes):
            idx = np.searchsorted(known_codes, codes)
            np.clip(idx, 0, len(known_codes) - 1, out=idx)
            hit = known_codes[idx] == codes
            labels = np.full(len(codes), -1, dtype=np.intp)
            labels[hit] = idx[hit]
            return labels

        def _extract_level(umesh):
            conn = umesh.getNodalConnectivity().toNumPyArray()
            conn_i = umesh.getNodalConnectivityIndex().toNumPyArray()
            n = umesh.getNumberOfCells()
            codes = conn[conn_i[:n]]
            labels = _type_labels(codes)
            unknown = codes[labels < 0]
            if len(unknown):
                codes_list = ", ".join(f"{c}" for c in np.unique(unknown))
                warnings.warn(
                    f"{len(unknown)} cells of unsupported MEDCoupling cell type "
                    f"(code(s) {codes_list}) skipped"
                )
            starts = conn_i[:n] + 1
            ends = conn_i[1 : n + 1]
            global_ids = {}
            for t, et in enumerate(known_types):
                sel = labels == t
                if np.any(sel):
                    global_ids[et] = np.nonzero(sel)[0]
            return conn, starts, ends, labels, global_ids

        def _gather(conn, starts, counts):
            n = len(counts)
            offsets = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(counts, out=offsets[1:])
            within = np.arange(offsets[-1]) - np.repeat(offsets[:-1], counts)
            entries = conn[np.repeat(starts, counts) + within]
            return offsets, within, entries

        def _build_blocks(res, conn, starts, ends, labels):
            for t, et in enumerate(known_types):
                sel = labels == t
                n_cells = np.count_nonzero(sel)
                if n_cells == 0:
                    continue
                counts = ends[sel] - starts[sel]
                if et in poly_types:
                    offsets, within, entries = _gather(conn, starts[sel], counts)
                    if et == "PGON":
                        data = entries.astype(np.uintp)
                        poly_offsets = offsets[1:].astype(np.uintp)
                    else:
                        base = offsets[:-1] + np.arange(n_cells)
                        out = np.full(offsets[-1] + n_cells, -1, dtype=np.int64)
                        out[base.repeat(counts) + within] = entries
                        data = out.astype(np.uintp)
                        poly_offsets = (offsets[1:] + np.arange(1, n_cells + 1)).astype(
                            np.uintp
                        )
                    res.add_poly_block(et, data, poly_offsets)
                else:
                    nn = mf_types_num_node[et]
                    block = conn[starts[sel][:, None] + np.arange(nn)]
                    if et in _mf_permutations:
                        block = block[:, _mf_permutations[et]]
                    res.add_regular_block(et, block.astype(np.uintp))

        if isinstance(mesh, mc.MEDFileUMesh):
            levels = [lev for lev in mesh.getNonEmptyLevelsExt() if lev != 1]
            level_meshes = [mesh.getMeshAtLevel(lev, False) for lev in levels]
        else:
            levels = [0]
            level_meshes = [mesh]

        ref_lev = 0 if 0 in levels else levels[0]
        coords = level_meshes[levels.index(ref_lev)].getCoords().toNumPyArray()
        space_dim = mesh.getSpaceDimension()

        level_data = [None] * len(levels)
        level_ids = [None] * len(levels)
        for j, (lev, umesh) in enumerate(zip(levels, level_meshes)):
            level_data[j] = _extract_level(umesh)
            level_ids[j] = level_data[j][4]

        res = cls(coords.reshape(-1, space_dim))

        for lev, (conn, starts, ends, labels, _) in zip(levels, level_data):
            _build_blocks(res, conn, starts, ends, labels)

        if isinstance(mesh, mc.MEDFileUMesh):
            handled = set()
            groups_by_name = {}
            for lev, ids in zip(levels, level_ids):
                for grp in mesh.getGroupsOnSpecifiedLev(lev):
                    handled.add(grp)
                    grp_ids = mesh.getGroupArr(lev, grp, False).toNumPyArray()
                    per_et = {}
                    for et, gids in ids.items():
                        local = np.nonzero(np.isin(gids, grp_ids))[0]
                        if len(local):
                            per_et[et] = local.astype(np.uintp)
                    if per_et:
                        groups_by_name.setdefault(grp, {}).update(per_et)
            for grp, per_et in groups_by_name.items():
                res.groups[grp] = per_et
            for grp in mesh.getGroupsOnSpecifiedLev(1):
                if grp not in handled:
                    warnings.warn(
                        f"Node group '{grp}' has no element at a cell level; "
                        "skipping it"
                    )

        if fields:
            for field in fields:
                if field.getTypeOfField() != mc.ON_CELLS:
                    raise ValueError(
                        "Only ON_CELLS fields are supported "
                        f"(field '{field.getName()}' is "
                        f"{field.getTypeOfField()})"
                    )
                fmesh = field.getMesh()
                lev = None
                for j, umesh in enumerate(level_meshes):
                    if fmesh is umesh or (
                        hasattr(fmesh, "isEqual") and fmesh.isEqual(umesh, 1e-12)
                    ):
                        lev = j
                        break
                if lev is None:
                    raise ValueError(
                        f"Field '{field.getName()}' is not attached to an "
                        "imported mesh level"
                    )
                values = field.getArray().toNumPyArray()
                n_cells = field.getNumberOfTuples()
                n_comp = field.getNumberOfComponents()
                values = values.reshape(n_cells, n_comp)
                per_et = {}
                for et, gids in level_ids[lev].items():
                    per_et[et] = values[gids]
                res.set_field(field.getName(), per_et)

        return res

    UMesh.from_mc = classmethod(from_mc)
