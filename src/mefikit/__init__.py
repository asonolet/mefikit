# ruff: disable[F403,F405,F821,E402]
from .mefipy import *

__doc__ = mefipy.__doc__
if hasattr(mefipy, "__all__"):
    __all__ = mefipy.__all__
else:
    __all__ = ()
del mefipy

from . import io as io
from . import data
# ruff: enable[F403,F405,F821,E402]


__all__ = (*__all__, "data")
