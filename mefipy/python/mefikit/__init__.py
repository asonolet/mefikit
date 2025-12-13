# ruff: disable[F403,F405,F821,E402]
from .mefikit import *

__doc__ = mefikit.__doc__
if hasattr(mefikit, "__all__"):
    __all__ = mefikit.__all__
else:
    __all__ = ()
del mefikit

from . import io as io
from . import data
# ruff: enable[F403,F405,F821,E402]


__all__ = (*__all__, "data")
