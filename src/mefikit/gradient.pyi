from typing import TypeAlias, final

import numpy as np
import numpy.typing as npt

from . import DistanceWeighting, Field, UMesh

_Array: TypeAlias = npt.NDArray[np.float64]

__all__ = ["Gradient"]

@final
class Gradient:
    def __new__(
        cls,
        src_mesh: UMesh,
        tgt_mesh: UMesh | None = ...,
        k: int = ...,
        weighting: DistanceWeighting = ...,
        def_val: float = ...,
    ) -> Gradient: ...
    @staticmethod
    def at_points(
        src_mesh: UMesh,
        points: _Array,
        k: int = ...,
        weighting: DistanceWeighting = ...,
        def_val: float = ...,
    ) -> Gradient: ...
    def __call__(
        self,
        expr: Field,
    ) -> Field: ...
    def eval(
        self,
        expr: Field,
    ) -> dict[str, _Array]: ...
    def eval_points(
        self,
        expr: Field,
    ) -> _Array: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
    ) -> None: ...
