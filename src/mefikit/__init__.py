import importlib.util

from . import data as data
from . import io
from .mefipy import Field, OverlayOperation, UMesh, build_cmesh, sel, transfer


def has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


if has("meshio") and has("medcoupling") and has("pyvista"):
    io.install_conversions()
del io

__all__ = (
    "Field",
    "OverlayOperation",
    "UMesh",
    "build_cmesh",
    "data",
    "sel",
    "transfer",
)
