from .mefikit import *
from . import io

__doc__ = mefikit.__doc__
if hasattr(mefikit, "__all__"):
    __all__ = mefikit.__all__
else:
    __all__ = ()

del mefikit
