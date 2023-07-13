"""
A PyTorch-powered python toolbox to train deep neural networks to perform motor tasks.
"""

from importlib import metadata

__name__ = "motornet"
__version__ = metadata.version("motornet")

from . import muscle
from . import skeleton
from . import environment
from . import effector
from . import plotor