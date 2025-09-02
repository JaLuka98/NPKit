"""
Roots module: A simple example for a python library.
"""

# Explicitly import the `newtons_method` function from `simple.py`
from .simple import newtons_method

# Expose the `newtons_method` function as part of the module's public API (nicer import)
__all__ = ["newtons_method"]
