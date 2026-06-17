from collections.abc import Sequence
from typing import TypeAlias

import medcoupling as mc
import meshio as mio
import numpy as np
import numpy.typing as npt
import pyvista as pv

# --- Helper type aliases ---

Array1F: TypeAlias = npt.NDArray[np.float64]
Array2F: TypeAlias = npt.NDArray[np.float64]
Array1U: TypeAlias = npt.NDArray[np.uintp]
Array2U: TypeAlias = npt.NDArray[np.uintp]
ArrayDynF: TypeAlias = npt.NDArray[np.float64]

# Connectivity:
# - Regular: (n_elem, n_nodes_per_elem)
# - Poly: (data, offsets)
Connectivity: TypeAlias = Array2U | tuple[Array1U, Array1U]

class PySelection: ...
class PyField: ...

def build_cmesh(*args: Sequence[Sequence[float]]) -> UMesh: ...

class UMesh:
    def __init__(self, coords: Array2F) -> None: ...

    # --- basic accessors ---

    def coords(self) -> Array2F: ...
    def block_types(self) -> list[str]: ...
    def blocks(self) -> dict[str, Connectivity]: ...
    def fields(self) -> dict[str, dict[str, ArrayDynF]]: ...

    # --- serialization ---

    def to_json(self) -> str: ...
    def to_json_pretty(self) -> str: ...
    @staticmethod
    def read(path: str) -> UMesh: ...
    def write(self, path: str) -> None: ...

    # --- mesh construction ---

    def add_regular_block(
        self,
        et: str,
        block: Array2U,
        fields: dict[str, ArrayDynF] | None = ...,
    ) -> None: ...

    # --- topology operations ---

    def descend(
        self,
        src_dim: int | None = ...,
        target_dim: int | None = ...,
    ) -> UMesh: ...
    def descend_update(
        self,
        src_dim: int | None = ...,
        target_dim: int | None = ...,
    ) -> UMesh | None: ...
    def boundaries(
        self,
        src_dim: int | None = ...,
        target_dim: int | None = ...,
    ) -> UMesh: ...
    def boundaries_update(
        self,
        src_dim: int | None = ...,
        target_dim: int | None = ...,
    ) -> UMesh | None: ...
    def connected_components(
        self,
        src_dim: int | None = ...,
        link_dim: int | None = ...,
        with_fields: bool = ...,
    ) -> list[UMesh]: ...

    # --- measures ---

    def measure(self) -> dict[str, Array1F]: ...
    def measure_update(self) -> None: ...

    # --- geometric ops ---

    def crack(self, cut_mesh: UMesh) -> UMesh: ...
    def snap(self, reference: UMesh, eps: float = ...) -> UMesh: ...
    def merge_nodes(self, eps: float = ...) -> UMesh: ...
    def extrude(self, along: list[float]) -> UMesh: ...
    def extrude_parallel(self, along: Array2F) -> UMesh: ...
    def extrude_curv(self, along: Array2F) -> UMesh: ...

    # --- selection & evaluation ---

    def select(self, expr: PySelection, with_fields: bool = ...) -> UMesh: ...
    def eval(self, expr: PyField) -> dict[str, ArrayDynF]: ...
    def eval_update(self, name: str, expr: PyField) -> None: ...

    # --- misc ---

    def __str__(self) -> str: ...

    # --- conversions ---

    def to_pyvista(
        self, dim: str | int | None = None, with_fields: Sequence[str] | bool = True
    ) -> pv.UnstructuredGrid: ...
    def to_mc(self, lev: int | None = None) -> mc.MEDCouplingUMesh: ...
    def to_meshio(self) -> mio.Mesh: ...
