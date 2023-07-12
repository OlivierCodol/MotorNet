"""This package contains various functions and scripts that can be useful to the general user for building, training,
and analysing models from `MotorNet`.
"""

from . import plotor
from . import net_utils


class Alias:
    """
    Object to create a transparent alias for attributes.
    Modified from a solution proposed here:
    https://adamj.eu/tech/2021/10/13/how-to-create-a-transparent-attribute-alias-in-python/

    Args:
        source_name: `String`, the name of the attribute being aliased.
        alias_name: `String`, the desired alias for the attribute.
    """
    def __init__(self, source_name: str, alias_name: str):
        self.source_name = source_name
        self.alias_name = alias_name

    def __get__(self, obj, objtype=None):
        if obj is None:
            # Class lookup, return descriptor
            return self
        if hasattr(obj, self.source_name):
            return getattr(obj, self.source_name)
        else:
            s = str(obj.__class__)
            raise AttributeError("'" + s[s.rfind('.')+1:-2] + "' object has no attribute '" + self.alias_name + "'")

    def __set__(self, obj, value):
        setattr(obj, self.source_name, value)

    def __delete__(self, obj):
        delattr(obj, self.source_name)
