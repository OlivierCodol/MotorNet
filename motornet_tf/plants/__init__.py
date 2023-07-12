"""This package contains the classes and subclasses used to create the plants that a model will be trained to control.
Plants are implemented as subclasses of the :class:`motornet.plants.plants.Plant` object. They all contain a single
:class:`motornet.plants.skeletons.Skeleton` and :class:`motornet.plants.muscles.Muscle` subclass defining the
biomechanical properties of that `Plant` object. The `Plant` object itself implements the numerical integration
routines, some of the geometry state calculation routines (in conjunction with the `Skeleton` object), applies moment
arms to produce generalized forces, and generally handles communication between the `Skeleton` and `Muscle` objects.

.. note::
    While `Plant` objects are technically implemented in the :class:`motornet.plants.plants` module, it is possible
    (and recommended) to call them using the :class:`motornet.plants.Plant` path, which is more concise and strictly
    equivalent.
"""

from . import muscles
from . import skeletons
from .plants import Plant
from .plants import RigidTendonArm26
from .plants import CompliantTendonArm26
from .plants import ReluPointMass24
