from typing import Any, TypeAlias, final

import numpy as np
import numpy.typing as npt

from . import DistanceWeighting, Field, UMesh

_Array: TypeAlias = npt.NDArray[np.float64]

__all__ = [
    "ConservativeP0",
    "ConstantPiecewise",
    "DistributedConservativeP0",
    "InverseDistance",
    "MovingLeastSquares",
    "distributed",
    "load",
]

@final
class ConstantPiecewise:
    def __new__(
        cls,
        src_mesh: UMesh,
        tgt_mesh: UMesh,
        def_val: float = ...,
    ) -> ConstantPiecewise: ...
    def save(self, path: str, compressed: bool = ...) -> None: ...
    def __call__(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> Field: ...
    def eval(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> dict[str, _Array]: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
        extensive: bool = ...,
    ) -> None: ...

@final
class MovingLeastSquares:
    def __new__(
        cls,
        src_mesh: UMesh,
        tgt_mesh: UMesh,
        k: int = ...,
        weighting: DistanceWeighting = ...,
        def_val: float = ...,
    ) -> MovingLeastSquares: ...
    def save(self, path: str, compressed: bool = ...) -> None: ...
    def __call__(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> Field: ...
    def eval(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> dict[str, _Array]: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
        extensive: bool = ...,
    ) -> None: ...

@final
class InverseDistance:
    def __new__(
        cls,
        src_mesh: UMesh,
        tgt_mesh: UMesh,
        k: int = ...,
        exponent: float = ...,
        def_val: float = ...,
    ) -> InverseDistance: ...
    def save(self, path: str, compressed: bool = ...) -> None: ...
    def __call__(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> Field: ...
    def eval(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> dict[str, _Array]: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
        extensive: bool = ...,
    ) -> None: ...

@final
class ConservativeP0:
    def __new__(
        cls,
        src_mesh: UMesh,
        tgt_mesh: UMesh,
        def_val: float = ...,
    ) -> ConservativeP0: ...
    def save(self, path: str, compressed: bool = ...) -> None: ...
    def __call__(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> Field: ...
    def eval(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> dict[str, _Array]: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
        extensive: bool = ...,
    ) -> None: ...
    def _apply_values(
        self,
        values: _Array,
        extensive: bool = ...,
        def_val: float = ...,
    ) -> dict[str, _Array]: ...

@final
class DistributedConservativeP0:
    def __new__(
        cls,
        src_mesh: UMesh | None,
        tgt_mesh: UMesh | None,
        comm: Any | None = ...,
        def_val: float = ...,
    ) -> DistributedConservativeP0: ...
    def eval(
        self,
        expr: Field | str | _Array,
        extensive: bool = ...,
        def_val: float | None = ...,
    ) -> dict[str, _Array]: ...
    def __call__(
        self,
        expr: Field | str | _Array,
        extensive: bool = ...,
        def_val: float | None = ...,
    ) -> dict[str, _Array]: ...
    def apply_update(
        self,
        src_mesh: UMesh | None,
        field_name: str,
        tgt_mesh: UMesh | None,
        tgt_field_name: str | None = ...,
        def_val: float | None = ...,
        extensive: bool = ...,
    ) -> None: ...

def distributed(
    src_mesh: UMesh | None,
    expr: Field | str | _Array,
    tgt_mesh: UMesh | None,
    comm: Any | None = ...,
    def_val: float = ...,
    extensive: bool = ...,
) -> dict[str, _Array]: ...

_Transfer: TypeAlias = (
    ConstantPiecewise | MovingLeastSquares | InverseDistance | ConservativeP0
)

def load(
    path: str,
    src_mesh: UMesh,
    tgt_mesh: UMesh | None = ...,
) -> _Transfer: ...
