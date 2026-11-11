from typing import TypeAlias, final

import numpy as np
import numpy.typing as npt

from . import DistanceWeighting, Field, UMesh

_Array: TypeAlias = npt.NDArray[np.float64]

__all__ = [
    "ConservativeP0",
    "ConstantPiecewise",
    "InverseDistance",
    "MovingLeastSquares",
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

_Transfer: TypeAlias = (
    ConstantPiecewise | MovingLeastSquares | InverseDistance | ConservativeP0
)

def load(
    path: str,
    src_mesh: UMesh,
    tgt_mesh: UMesh | None = ...,
) -> _Transfer: ...
