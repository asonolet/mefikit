from typing import Callable

from . import Field, UMesh

class ConstantPiecewise:
    def __init__(
        self,
        src_mesh: UMesh,
        tgt_mesh: UMesh,
        def_val: float = ...,
    ) -> None: ...
    def __call__(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> Field: ...
    def eval(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> dict[str, object]: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
    ) -> None: ...

class DistanceWeighting:
    Constant: Callable[[], DistanceWeighting]
    InverseDistance: Callable[[float], DistanceWeighting]
    Gaussian: Callable[[], DistanceWeighting]

class MovingLeastSquares:
    def __init__(
        self,
        src_mesh: UMesh,
        tgt_mesh: UMesh,
        k: int = ...,
        weighting: DistanceWeighting = ...,
        def_val: float = ...,
    ) -> None: ...
    def __call__(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> Field: ...
    def eval(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> dict[str, object]: ...
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
        self,
        src_mesh: UMesh,
        tgt_mesh: UMesh,
        k: int = ...,
        exponent: float = ...,
        def_val: float = ...,
    ) -> None: ...
    def __call__(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> Field: ...
    def eval(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> dict[str, object]: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
    ) -> None: ...

class ConservativeP0:
    def __init__(
        self,
        src_mesh: UMesh,
        tgt_mesh: UMesh,
        def_val: float = ...,
    ) -> None: ...
    def __call__(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> Field: ...
    def eval(
        self,
        expr: Field,
        extensive: bool = False,
    ) -> dict[str, object]: ...
    def apply_update(
        self,
        src_mesh: UMesh,
        field_name: str,
        tgt_mesh: UMesh,
        tgt_field_name: str | None = ...,
        def_val: float = ...,
        extensive: bool = ...,
    ) -> None: ...
