from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from . import DistanceWeighting, Field, UMesh

Array2F: TypeAlias = npt.NDArray[np.float64]

class Gradient:
    def __init__(
        self,
        src_mesh: UMesh,
        tgt_mesh: UMesh | None = ...,
        k: int = ...,
        weighting: DistanceWeighting = ...,
        def_val: float = ...,
    ) -> None: ...
    @staticmethod
    def at_points(
        src_mesh: UMesh,
        points: Array2F,
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
    ) -> dict[str, object]: ...
    def eval_points(
        self,
        expr: Field,
    ) -> Array2F: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
    ) -> None: ...
