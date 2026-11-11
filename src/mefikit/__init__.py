import importlib.util
import sys
from enum import IntFlag

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
    TimeSeries,
    TimeSeriesMapping,
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

# The PyO3 submodules are attributes of the extension module. Register them
# under the package name as well so `import mefikit.transfer` (and the matching
# `.pyi` stubs) works like a regular package submodule.
sys.modules[f"{__name__}.gradient"] = gradient
sys.modules[f"{__name__}.sel"] = sel
sys.modules[f"{__name__}.transfer"] = transfer
del sys


class ConformanceChecks(IntFlag):
    """Which checks :func:`is_conform` runs.

    ``FAST`` runs the structural and per-cell checks; ``FULL`` adds the expensive
    cell-pair searches.
    """

    NONE = 0
    TOPOLOGY = 1
    BUTTERFLY = 2
    INTERSECTIONS = 4
    FACE_OVERLAP = 8
    FAST = TOPOLOGY | BUTTERFLY
    FULL = TOPOLOGY | BUTTERFLY | INTERSECTIONS | FACE_OVERLAP


def has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


if has("meshio") and has("medcoupling") and has("pyvista"):
    io.install_conversions()
del io

__all__ = (
    "C",
    "ConformanceChecks",
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
    "TimeSeries",
    "TimeSeriesMapping",
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
