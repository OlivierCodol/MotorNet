"""
This module contains four submodules, with each submodule implementing a series of subclasses based off typical
tensorflow objects. In other words, this module can be conceptualized as an extension of tensorflow for the purpose of
bridging `motornet.plants` classes and tensorflow itself.
"""

from . import callbacks
from . import layers
from . import losses
from . import models
from .models import MotorNetModel
