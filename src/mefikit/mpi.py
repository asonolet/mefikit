"""Distributed (MPI) conservative P0 transfer.

This module is the Python-only entry point for parallel field transfers. The
geometry work stays in the Rust :class:`mefikit.transfer.ConservativeP0`
operator; MPI communication is orchestrated with :mod:`mpi4py`, so the Rust
crate does not depend on an MPI implementation at build time.

The parallel algorithm follows the same decomposition as MEDCoupling's
``InterpKernelDEC``:

* each rank owns a local piece of the source mesh and/or a local piece of the
  target mesh;
* a broad phase exchanges only the source cells whose bounding box intersects a
  target rank's bounding box;
* each target rank builds one local :class:`ConservativeP0` operator for its
  local target cells against the received source pieces;
* applying a field is one sparse matrix-vector product after a single
  ``Alltoallv`` that fetches the source cell values requested by each target
  rank.

The operator is prepared once and can be reused for any number of fields (for
example in a time loop), as long as the source and target partitions do not
change.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from . import io as _io
from . import mefipy as _mf

_TAG_BBOX = 4101
_TAG_MESH = 4102
_TAG_REQ = 4103

_MAX_UINT = np.iinfo(np.uintp).max


# ---------------------------------------------------------------------------
# Small mesh helpers
# ---------------------------------------------------------------------------


def _block_len(conn: Any) -> int:
    """Number of cells in a connectivity array (regular or poly)."""
    if isinstance(conn, tuple):
        # Poly connectivity stores one cumulative end offset per cell.
        return len(conn[1])
    return int(np.asarray(conn).shape[0])


def _cell_bounds(
    conn: Any, coords: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Axis-aligned bounding box of every cell in one connectivity block."""
    if isinstance(conn, tuple):
        data = np.asarray(conn[0])
        offsets = np.asarray(conn[1])
        n = len(offsets)
        dim = coords.shape[1]
        lo = np.empty((n, dim), dtype=np.float64)
        hi = np.empty((n, dim), dtype=np.float64)
        start = 0
        for i in range(n):
            stop = offsets[i]
            nodes = data[start:stop]
            nodes = nodes[nodes != _MAX_UINT]
            pts = coords[nodes]
            lo[i] = pts.min(axis=0)
            hi[i] = pts.max(axis=0)
            start = stop
        return lo, hi

    nodes = np.asarray(conn, dtype=np.intp)
    if nodes.shape[0] == 0:
        dim = coords.shape[1]
        return np.empty((0, dim)), np.empty((0, dim))
    pts = coords[nodes]
    return pts.min(axis=1), pts.max(axis=1)


class _MeshInfo:
    """Cached per-block metadata and bounds for one local mesh piece."""

    def __init__(self, mesh: Any) -> None:
        self.coords = np.asarray(mesh.coords(), dtype=np.float64)
        blocks = mesh.blocks()
        self.block_order: list[str] = list(blocks)
        self.lengths = {et: _block_len(conn) for et, conn in blocks.items()}
        self.dims = {et: _io.mf_types_dim[et] for et in self.block_order}
        self.bounds = {
            et: _cell_bounds(conn, self.coords) for et, conn in blocks.items()
        }
        dims = [self.dims[et] for et in self.block_order if self.lengths[et] > 0]
        self.top_dim = max(dims) if dims else None

    def bbox(self, dim: int) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """Bounding box of the cells of topological dimension ``dim``."""
        lo = None
        hi = None
        for et in self.block_order:
            if self.dims[et] != dim or self.lengths[et] == 0:
                continue
            blo = self.bounds[et][0].min(axis=0)
            bhi = self.bounds[et][1].max(axis=0)
            lo = blo if lo is None else np.minimum(lo, blo)
            hi = bhi if hi is None else np.maximum(hi, bhi)
        if lo is None or hi is None:
            return None
        return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def _bbox_intersects(
    a: tuple[tuple[float, ...], tuple[float, ...]] | None,
    b: tuple[tuple[float, ...], tuple[float, ...]] | None,
) -> bool:
    """Whether two axis-aligned bounding boxes intersect (closed boxes)."""
    if a is None or b is None:
        return False
    (a_min, a_max), (b_min, b_max) = a, b
    if len(a_min) != len(b_min):
        return False
    scale = max(
        1.0,
        *(abs(v) for v in (*a_min, *a_max, *b_min, *b_max)),
    )
    eps = 1e-12 * scale
    for x0, x1, y0, y1 in zip(a_min, a_max, b_min, b_max, strict=True):
        if x1 < y0 - eps or y1 < x0 - eps:
            return False
    return True


def _select_cells_in_bbox(
    info: _MeshInfo | None,
    dim: int,
    bbox: tuple[tuple[float, ...], tuple[float, ...]] | None,
) -> dict[str, npt.NDArray[np.intp]]:
    """Indices of the cells of dimension ``dim`` whose bounds meet ``bbox``."""
    if info is None or bbox is None:
        return {}
    b_min, b_max = bbox
    scale = max(1.0, *(abs(v) for v in (*b_min, *b_max)))
    eps = 1e-12 * scale
    selected: dict[str, npt.NDArray[np.intp]] = {}
    for et in info.block_order:
        if info.dims[et] != dim or info.lengths[et] == 0:
            continue
        lo, hi = info.bounds[et]
        n = lo.shape[1]
        if n != len(b_min):
            continue
        b0 = np.asarray(b_min[:n]) - eps
        b1 = np.asarray(b_max[:n]) + eps
        mask = np.all(hi >= b0, axis=1) & np.all(lo <= b1, axis=1)
        ids = np.flatnonzero(mask).astype(np.intp)
        if ids.size:
            selected[et] = ids
    return selected


def _source_offsets(info: _MeshInfo, dim: int) -> dict[str, int]:
    """First flat source index of each source-dimension element block."""
    offsets: dict[str, int] = {}
    offset = 0
    for et in info.block_order:
        if info.dims[et] != dim:
            continue
        offsets[et] = offset
        offset += info.lengths[et]
    return offsets


def _extract_submesh(
    mesh: Any, ids_by_et: dict[str, npt.NDArray[np.intp]]
) -> Any | None:
    """Extract selected cells and compact the coordinate array."""
    if not ids_by_et:
        return None
    selected = {
        et: np.ascontiguousarray(ids, dtype=np.uintp)
        for et, ids in ids_by_et.items()
        if ids.size
    }
    if not selected:
        return None
    sub = mesh.select(selected).to_mesh(with_fields=False)
    return sub.remove_unused_coords()


def _pack_mesh(mesh: Any) -> dict[str, Any] | None:
    if mesh is None:
        return None
    blocks: dict[str, Any] = {}
    for et, conn in mesh.blocks().items():
        if isinstance(conn, tuple):
            blocks[et] = (
                np.ascontiguousarray(conn[0], dtype=np.uintp),
                np.ascontiguousarray(conn[1], dtype=np.uintp),
            )
        else:
            blocks[et] = np.ascontiguousarray(conn, dtype=np.uintp)
    return {
        "coords": np.ascontiguousarray(mesh.coords(), dtype=np.float64),
        "blocks": blocks,
    }


def _unpack_mesh(payload: dict[str, Any] | None) -> Any | None:
    if payload is None:
        return None
    mesh = _mf.UMesh(payload["coords"])
    for et, conn in payload["blocks"].items():
        if isinstance(conn, tuple):
            mesh.add_poly_block(et, conn[0], conn[1])
        else:
            mesh.add_regular_block(et, conn)
    return mesh


def _pack_piece(
    mesh: Any | None,
    ids_by_et: dict[str, npt.NDArray[np.intp]],
    offsets: dict[str, int],
) -> tuple[dict[str, Any] | None, dict[str, npt.NDArray[np.intp]], dict[str, int]]:
    if mesh is None or not ids_by_et:
        return None, {}, offsets
    sub = _extract_submesh(mesh, ids_by_et)
    return _pack_mesh(sub), ids_by_et, offsets


# ---------------------------------------------------------------------------
# Distributed operator
# ---------------------------------------------------------------------------


class DistributedConservativeP0:
    """Conservative P0 transfer over a distributed source and target mesh.

    Each rank passes the local piece of the source mesh and/or the local piece
    of the target mesh (``None`` is accepted for a rank that owns no cell on
    one side). The union of the local pieces is the global source mesh, and
    likewise for the target mesh.

    The constructor performs the parallel precompute: a bounding-box broad
    phase, exchange of the candidate source cells, and construction of one
    local Rust ``ConservativeP0`` operator per target rank. The operator is
    then reused by :meth:`eval` / :meth:`apply_update`.

    Parameters
    ----------
    src_mesh:
        Local source mesh piece, or ``None``.
    tgt_mesh:
        Local target mesh piece, or ``None``.
    comm:
        MPI communicator (defaults to ``MPI.COMM_WORLD``). All ranks of the
        communicator must call the constructor.
    def_val:
        Value assigned to target cells covered by no source cell.

    Notes
    -----
    This follows the same field-nature convention as the sequential
    :class:`mefikit.transfer.ConservativeP0`: intensive fields are normalized
    by the target cell measure, while ``extensive=True`` keeps the raw
    measure-weighted sum. Downcasts (3D -> 2D/1D/0D, 2D -> 1D/0D, 1D -> 0D) are
    supported with the same geometry rules as the sequential operator.
    """

    def __init__(
        self,
        src_mesh: Any | None,
        tgt_mesh: Any | None,
        comm: Any | None = None,
        def_val: float = 0.0,
    ) -> None:
        try:
            from mpi4py import MPI
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "DistributedConservativeP0 requires mpi4py; install it with "
                "`pip install mefikit[mpi]`."
            ) from exc

        self._mpi = MPI
        self._comm = MPI.COMM_WORLD if comm is None else comm
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        self._def_val = float(def_val)
        self._src_mesh = src_mesh
        self._tgt_mesh = tgt_mesh
        self._src_info = _MeshInfo(src_mesh) if src_mesh is not None else None
        self._tgt_info = _MeshInfo(tgt_mesh) if tgt_mesh is not None else None

        self._space_dim = self._validated_space_dim()
        src_dim = self._global_top_dim(self._src_info)
        tgt_dim = self._global_top_dim(self._tgt_info)
        if src_dim is None:
            raise ValueError("the distributed transfer has no source mesh on any rank")
        if tgt_dim is None:
            raise ValueError("the distributed transfer has no target mesh on any rank")
        if src_dim < tgt_dim:
            raise ValueError(
                "Conservative P0 cannot transfer from lower-dimensional source "
                f"cells ({src_dim}D) to higher-dimensional target cells ({tgt_dim}D)"
            )
        self._src_dim = src_dim
        self._tgt_dim = tgt_dim

        self._src_offsets = (
            _source_offsets(self._src_info, self._src_dim)
            if self._src_info is not None
            else {}
        )

        src_bbox = (
            self._src_info.bbox(self._src_dim) if self._src_info is not None else None
        )
        tgt_bbox = (
            self._tgt_info.bbox(self._tgt_dim) if self._tgt_info is not None else None
        )
        all_boxes = self._comm.allgather((src_bbox, tgt_bbox))
        self._prepare(all_boxes, tgt_bbox)

    # -- setup ---------------------------------------------------------

    def _validated_space_dim(self) -> int:
        local: int | None = None
        for info in (self._src_info, self._tgt_info):
            if info is None or info.coords.shape[1] == 0:
                continue
            dim = int(info.coords.shape[1])
            if local is not None and local != dim:
                raise ValueError(
                    "the local source and target meshes must share the same "
                    f"space dimension, got {local}D and {dim}D"
                )
            local = dim
        dims = self._comm.allgather(local)
        active = [d for d in dims if d is not None]
        if not active:
            raise ValueError("the distributed transfer has no mesh on any rank")
        if any(d != active[0] for d in active):
            raise ValueError(
                "source and target meshes must share the same space dimension, "
                f"got {sorted(set(active))}"
            )
        if active[0] not in (2, 3):
            raise ValueError(
                "Conservative P0 is only supported in 2D and 3D space, got "
                f"{active[0]}D"
            )
        return active[0]

    def _global_top_dim(self, info: _MeshInfo | None) -> int | None:
        local = info.top_dim if info is not None else None
        dims = self._comm.allgather(local)
        active = [d for d in dims if d is not None]
        return max(active) if active else None

    def _prepare(
        self,
        all_boxes: list[
            tuple[
                tuple[tuple[float, ...], tuple[float, ...]] | None,
                tuple[tuple[float, ...], tuple[float, ...]] | None,
            ]
        ],
        tgt_bbox: tuple[tuple[float, ...], tuple[float, ...]] | None,
    ) -> None:
        comm = self._comm
        rank = self._rank
        size = self._size

        owners = (
            [r for r in range(size) if _bbox_intersects(all_boxes[r][0], tgt_bbox)]
            if tgt_bbox is not None
            else []
        )

        # Boolean all-to-all: each source rank learns which target ranks need
        # cells from it, without an O(P^2) object exchange.
        want = np.zeros(size, dtype=np.uint8)
        want[owners] = 1
        have = np.empty(size, dtype=np.uint8)
        comm.Alltoall(want, have)
        requesters = [int(i) for i in np.flatnonzero(have)]

        pending = []

        # Target -> source: send the local target bounding box.
        for owner in owners:
            if owner != rank:
                pending.append(comm.isend(tgt_bbox, dest=owner, tag=_TAG_BBOX))

        # Source side: filter the local source cells against every target bbox
        # and send back the corresponding compact submesh.
        local_payload: tuple[
            dict[str, Any] | None,
            dict[str, npt.NDArray[np.intp]],
            dict[str, int],
        ] = (None, {}, self._src_offsets)
        remote_payloads: dict[
            int,
            tuple[
                dict[str, Any] | None,
                dict[str, npt.NDArray[np.intp]],
                dict[str, int],
            ],
        ] = {}

        for target in requesters:
            if target == rank:
                box = tgt_bbox
            else:
                box = comm.recv(source=target, tag=_TAG_BBOX)
            ids = _select_cells_in_bbox(self._src_info, self._src_dim, box)
            payload = _pack_piece(self._src_mesh, ids, self._src_offsets)
            if target == rank:
                local_payload = payload
            else:
                pending.append(comm.isend(payload, dest=target, tag=_TAG_MESH))

        # Target side: receive one source submesh per candidate owner.
        for owner in owners:
            if owner != rank:
                remote_payloads[owner] = comm.recv(source=owner, tag=_TAG_MESH)

        for request in pending:
            request.Wait()

        payloads = {rank: local_payload}
        payloads.update(remote_payloads)

        pieces = []
        for owner in owners:
            packed, ids, offsets = payloads.get(owner, (None, {}, {}))
            if packed is None or not ids:
                continue
            mesh = _unpack_mesh(packed)
            if mesh is None:  # pragma: no cover - defensive
                continue
            pieces.append((owner, mesh, ids, offsets))

        self._op = None
        self._n_combined = 0
        empty = np.empty(0, dtype=np.intp)
        self._requests_to_owner = [empty.copy() for _ in range(size)]
        self._positions_to_owner = [empty.copy() for _ in range(size)]

        target_mesh = self._tgt_mesh
        if pieces and target_mesh is not None:
            combined = (
                pieces[0][1]
                if len(pieces) == 1
                else _mf.aggregate([piece[1] for piece in pieces])
            )
            self._op = _mf.transfer.ConservativeP0(combined, target_mesh)
            self._build_request_plan(combined, pieces)

        # Target -> source: send the local source indices needed by each source
        # owner, so that applying a field only needs one Alltoallv.
        pending = []
        for owner in owners:
            if owner != rank:
                pending.append(
                    comm.isend(
                        np.ascontiguousarray(
                            self._requests_to_owner[owner], dtype=np.uintp
                        ),
                        dest=owner,
                        tag=_TAG_REQ,
                    )
                )

        self._requests_from_target = [empty.copy() for _ in range(size)]
        for target in requesters:
            if target == rank:
                self._requests_from_target[target] = self._requests_to_owner[target]
            else:
                received = comm.recv(source=target, tag=_TAG_REQ)
                self._requests_from_target[target] = np.asarray(received, dtype=np.intp)

        for request in pending:
            request.Wait()

        self._send_counts = np.asarray(
            [len(v) for v in self._requests_from_target], dtype=np.int64
        )
        self._recv_counts = np.asarray(
            [len(v) for v in self._requests_to_owner], dtype=np.int64
        )

    def _build_request_plan(
        self,
        combined: Any,
        pieces: list[tuple[int, Any, dict[str, npt.NDArray[np.intp]], dict[str, int]]],
    ) -> None:
        """Map every combined source cell to its owner and local field index."""
        et_order = combined.block_types()
        et_offset: dict[str, int] = {}
        offset = 0
        for et in et_order:
            et_offset[et] = offset
            offset += sum(len(piece[2].get(et, ())) for piece in pieces)
        self._n_combined = offset

        positions: dict[tuple[int, str], npt.NDArray[np.intp]] = {}
        for et in et_order:
            running = et_offset[et]
            for owner, _, ids, _ in pieces:
                count = len(ids.get(et, ()))
                if count:
                    positions[(owner, et)] = running + np.arange(count, dtype=np.intp)
                running += count

        for owner, _, ids, offsets in pieces:
            requests = []
            cell_positions = []
            for et in et_order:
                local_ids = ids.get(et)
                if local_ids is None or local_ids.size == 0:
                    continue
                requests.append(offsets[et] + local_ids)
                cell_positions.append(positions[(owner, et)])
            if requests:
                self._requests_to_owner[owner] = np.concatenate(requests).astype(
                    np.intp, copy=False
                )
                self._positions_to_owner[owner] = np.concatenate(cell_positions).astype(
                    np.intp, copy=False
                )

    # -- evaluation ----------------------------------------------------

    def _local_source_values(
        self, mesh: Any, expr: Any
    ) -> tuple[npt.NDArray[np.float64] | None, tuple[int, ...] | None]:
        if mesh is None:
            return None, None
        if mesh is self._src_mesh and self._src_info is not None:
            info = self._src_info
        else:
            info = _MeshInfo(mesh)
        if not isinstance(expr, _mf.Field):
            expr = _mf.Field(expr)
        values = mesh.eval(expr, dim=self._src_dim)
        parts: list[npt.NDArray[np.float64]] = []
        trailing: tuple[int, ...] | None = None
        for et in info.block_order:
            if info.dims[et] != self._src_dim:
                continue
            if et not in values:
                raise ValueError(
                    f"the source field expression has no values for block {et}"
                )
            arr = np.asarray(values[et], dtype=np.float64)
            if arr.ndim == 0:
                arr = np.full((info.lengths[et],), float(arr), dtype=np.float64)
            if arr.shape[0] != info.lengths[et]:
                raise ValueError(
                    f"the source field has {arr.shape[0]} values for block {et}, "
                    f"expected {info.lengths[et]}"
                )
            shape = arr.shape[1:]
            if trailing is None:
                trailing = shape
            elif trailing != shape:
                raise ValueError(
                    "all source element blocks must carry the same number of "
                    f"components, got {trailing} and {shape}"
                )
            parts.append(arr.reshape(arr.shape[0], -1))
        if not parts:
            return np.empty((0, 1), dtype=np.float64), None
        return np.concatenate(parts, axis=0), trailing

    def _evaluate(
        self,
        src_mesh: Any | None,
        expr: Any,
        extensive: bool,
        def_val: float,
    ) -> dict[str, npt.NDArray[np.float64]]:
        flat, trailing = self._local_source_values(src_mesh, expr)

        reports = self._comm.allgather(trailing)
        active = [s for s in reports if s is not None]
        if not active:
            raise ValueError(
                "the source field expression produced no source cell on any rank"
            )
        trailing_shape = active[0]
        if any(s != trailing_shape for s in active):
            raise ValueError(
                "all source ranks must evaluate the same field with the same "
                f"trailing shape, got {sorted(set(active))}"
            )
        n_compo = int(np.prod(trailing_shape)) if trailing_shape else 1

        if flat is None:
            flat = np.empty((0, n_compo), dtype=np.float64)
        elif flat.size == 0:
            flat = flat.reshape(0, n_compo)

        size = self._size
        combined = np.zeros((self._n_combined, n_compo), dtype=np.float64)

        # The local source piece never needs to travel through MPI.
        local_requests = self._requests_to_owner[self._rank]
        if local_requests.size:
            combined[self._positions_to_owner[self._rank]] = flat[local_requests]

        send_counts = (self._send_counts * n_compo).astype(np.int64, copy=True)
        recv_counts = (self._recv_counts * n_compo).astype(np.int64, copy=True)
        send_counts[self._rank] = 0
        recv_counts[self._rank] = 0
        send_displs = np.zeros(size, dtype=np.int64)
        recv_displs = np.zeros(size, dtype=np.int64)
        if size > 1:
            send_displs[1:] = np.cumsum(send_counts)[:-1]
            recv_displs[1:] = np.cumsum(recv_counts)[:-1]

        send_parts = [
            flat[indices].reshape(-1)
            for target, indices in enumerate(self._requests_from_target)
            if target != self._rank and indices.size
        ]
        sendbuf = (
            np.concatenate(send_parts).astype(np.float64, copy=False)
            if send_parts
            else np.empty(0, dtype=np.float64)
        )
        recvbuf = np.empty(int(recv_counts.sum()), dtype=np.float64)
        sendbuf = np.ascontiguousarray(sendbuf)
        send_counts_list = send_counts.tolist()
        recv_counts_list = recv_counts.tolist()
        send_displs_list = send_displs.tolist()
        recv_displs_list = recv_displs.tolist()
        try:
            # mpi4py >= 4 takes buffer-spec sequences.
            self._comm.Alltoallv(
                [sendbuf, send_counts_list, send_displs_list, self._mpi.DOUBLE],
                [recvbuf, recv_counts_list, recv_displs_list, self._mpi.DOUBLE],
            )
        except TypeError:  # pragma: no cover - mpi4py 3 compatibility
            self._comm.Alltoallv(  # type: ignore[call-arg]
                sendbuf,
                send_counts_list,
                send_displs_list,
                recvbuf,
                recv_counts_list,
                recv_displs_list,
            )

        for owner in range(size):
            if owner == self._rank:
                continue
            count = int(self._recv_counts[owner])
            if count == 0:
                continue
            lo = int(recv_displs[owner])
            hi = lo + count * n_compo
            combined[self._positions_to_owner[owner]] = recvbuf[lo:hi].reshape(
                count, n_compo
            )

        combined_values: npt.NDArray[np.float64]
        if trailing_shape:
            combined_values = combined.reshape((self._n_combined, *trailing_shape))
        else:
            combined_values = combined.reshape(self._n_combined)

        if self._op is None:
            return self._default_field(trailing_shape, def_val)

        result = self._op._apply_values(  # type: ignore[attr-defined]
            combined_values, extensive=extensive, def_val=def_val
        )
        return {et: np.asarray(arr) for et, arr in result.items()}

    def _default_field(
        self, trailing_shape: tuple[int, ...], def_val: float
    ) -> dict[str, npt.NDArray[np.float64]]:
        if self._tgt_info is None or self._tgt_dim is None:
            return {}
        result = {}
        for et in self._tgt_info.block_order:
            if self._tgt_info.dims[et] != self._tgt_dim:
                continue
            n = self._tgt_info.lengths[et]
            if n == 0:
                continue
            result[et] = np.full((n, *trailing_shape), def_val, dtype=np.float64)
        return result

    # -- public API ----------------------------------------------------

    def eval(
        self,
        expr: Any,
        extensive: bool = False,
        def_val: float | None = None,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Transfer a field to the local target cells.

        ``expr`` accepts the same field expressions as the sequential
        operators (a field name, a :class:`mefikit.Field`, or an array), and is
        evaluated on each rank's local source mesh.

        Returns the per-element-type arrays of the local target field. Uncovered
        target cells keep ``def_val`` (the constructor value unless overridden).
        """
        return self._evaluate(
            self._src_mesh,
            expr,
            extensive,
            self._def_val if def_val is None else def_val,
        )

    def __call__(
        self,
        expr: Any,
        extensive: bool = False,
        def_val: float | None = None,
    ) -> dict[str, npt.NDArray[np.float64]]:
        return self.eval(expr, extensive=extensive, def_val=def_val)

    def apply_update(
        self,
        src_mesh: Any | None,
        field_name: str,
        tgt_mesh: Any | None,
        tgt_field_name: str | None = None,
        def_val: float | None = None,
        extensive: bool = False,
    ) -> None:
        """Transfer ``field_name`` and store the result on the local target mesh.

        ``src_mesh`` and ``tgt_mesh`` must describe the same local pieces as the
        constructor (they may be the same objects). Pass ``None`` on a rank that
        owns no cell on that side.
        """
        source = self._src_mesh if src_mesh is None else src_mesh
        target = self._tgt_mesh if tgt_mesh is None else tgt_mesh
        values = self._evaluate(
            source,
            field_name,
            extensive,
            self._def_val if def_val is None else def_val,
        )
        if target is not None and values:
            target.set_field(tgt_field_name or field_name, values)

    def __repr__(self) -> str:
        return (
            f"DistributedConservativeP0(rank={self._rank}/{self._size}, "
            f"src_dim={self._src_dim}, tgt_dim={self._tgt_dim}, "
            f"n_src={self._n_combined})"
        )


def distributed(
    src_mesh: Any | None,
    expr: Any,
    tgt_mesh: Any | None,
    comm: Any | None = None,
    def_val: float = 0.0,
    extensive: bool = False,
) -> dict[str, npt.NDArray[np.float64]]:
    """One-shot distributed conservative P0 transfer.

    This is a convenience wrapper around
    :class:`DistributedConservativeP0`; build the class explicitly to reuse the
    prepared operator for several fields.
    """
    operator = DistributedConservativeP0(src_mesh, tgt_mesh, comm=comm, def_val=def_val)
    return operator.eval(expr, extensive=extensive)


__all__ = ["DistributedConservativeP0", "distributed"]
