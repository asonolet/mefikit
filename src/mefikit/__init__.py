import importlib.util

from . import data as data
from . import io
from .mefipy import (
    C,
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
    UMesh,
    X,
    Y,
    Z,
    build_cmesh,
    sel,
    transfer,
)

ConstantPiecewise = transfer.ConstantPiecewise
MovingLeastSquares = transfer.MovingLeastSquares
InverseDistance = transfer.InverseDistance
ConservativeP0 = transfer.ConservativeP0
DistanceWeighting = transfer.DistanceWeighting


def has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


if has("meshio") and has("medcoupling") and has("pyvista"):
    io.install_conversions()
del io

__all__ = (
    "C",
    "ConservativeP0",
    "ConstantPiecewise",
    "DistanceWeighting",
    "Field",
    "FieldRef",
    "FieldsMapping",
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
    "UMesh",
    "X",
    "Y",
    "Z",
    "build_cmesh",
    "data",
    "sel",
    "transfer",
)
