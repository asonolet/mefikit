import importlib.util

from . import data as data
from . import io
from .mefipy import (
    C,
    ConformanceReport,
    DistanceWeighting,
    Field,
    FieldRef,
    FieldsMapping,
    GroupRef,
    GroupsMapping,
    M,
    Normal,
    Nx,
    Ny,
    Nz,
    OverlayOperation,
    Selection,
    SelectionResult,
    SurfaceOverlay,
    Transform,
    UMesh,
    X,
    Y,
    Z,
    aggregate,
    build_cmesh,
    concat,
    conformize,
    gradient,
    is_conform,
    sel,
    stitch,
    transfer,
)

ConstantPiecewise = transfer.ConstantPiecewise
MovingLeastSquares = transfer.MovingLeastSquares
InverseDistance = transfer.InverseDistance
ConservativeP0 = transfer.ConservativeP0

Gradient = gradient.Gradient


def has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


if has("meshio") and has("medcoupling") and has("pyvista"):
    io.install_conversions()
del io

__all__ = (
    "C",
    "ConformanceReport",
    "ConservativeP0",
    "ConstantPiecewise",
    "DistanceWeighting",
    "Field",
    "FieldRef",
    "FieldsMapping",
    "Gradient",
    "GroupRef",
    "GroupsMapping",
    "InverseDistance",
    "M",
    "MovingLeastSquares",
    "Normal",
    "Nx",
    "Ny",
    "Nz",
    "OverlayOperation",
    "Selection",
    "SelectionResult",
    "SurfaceOverlay",
    "Transform",
    "UMesh",
    "X",
    "Y",
    "Z",
    "aggregate",
    "build_cmesh",
    "concat",
    "conformize",
    "data",
    "gradient",
    "is_conform",
    "sel",
    "stitch",
    "transfer",
)
