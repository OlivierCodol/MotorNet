"""
A TensorFlow-powered python toolbox to train deep neural networks to perform motor tasks.

.. note::
    The API referenced in the documentation is outdated and corresponds to version 0.1.5 of MotorNet, which relies on
    TensorFlow. However, it is still referenced here for users who may wish to consult it.
"""

__name__ = "motornet_tf"
__version__ = "0.1.5"
__author__ = 'Olivier Codol'

from . import nets
from . import plants
from . import utils
from . import tasks
