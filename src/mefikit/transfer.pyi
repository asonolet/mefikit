from typing import Callable

from . import UMesh

class ConstantPiecewise:
    def __init__(self, src_mesh: UMesh, tgt_mesh: UMesh) -> None: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
    ) -> None: ...

class DistanceWeighting:
    # `DistanceWeighting.None()` also exists at runtime, but `None` is a Python
    # keyword and cannot be spelled in a stub file.
    InverseDistance: Callable[[float], DistanceWeighting]
    Gaussian: Callable[[], DistanceWeighting]

class MovingLeastSquares:
    def __init__(
        self,
        src_mesh: UMesh,
        tgt_mesh: UMesh,
        k: int = ...,
        weighting: DistanceWeighting = ...,
    ) -> None: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
    ) -> None: ...

class InverseDistance:
    def __init__(
        self, src_mesh: UMesh, tgt_mesh: UMesh, k: int = ..., exponent: float = ...
    ) -> None: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
    ) -> None: ...

class ConservativeP0:
    def __init__(self, src_mesh: UMesh, tgt_mesh: UMesh) -> None: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
        extensive: bool = ...,
    ) -> None: ...
