"""
The package containing the implementations of all supported architectures. Not necessary for most user code.
"""

from .AetherNet import AetherNetArch
from .ESRGAN import ESRGANArch
from .SwinIR import SwinIRArch

__all__ = [
    "AetherNetArch",
    "ESRGANArch",
    "SwinIRArch",
]

__docformat__ = "google"
