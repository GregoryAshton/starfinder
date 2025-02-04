from .logger import *
from .starfield import StarField

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = 'unknown'
