# The MIT License (MIT)
# Copyright © 2025 Sparket

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Read version from pyproject.toml via importlib.metadata
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("sparket-subnet")
except Exception:
    __version__ = "0.0.0"

# Spec version formula: base + (1000 * major) + (10 * minor) + (1 * patch)
# Base offset required to meet existing subnet weights_version hyperparameter (>= 2400)
_SPEC_VERSION_BASE = 2401

version_split = __version__.split(".")
__spec_version__ = _SPEC_VERSION_BASE + (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# Import all submodules is disabled in tests to avoid heavy deps; set env to enable.
import os as _os

if _os.environ.get("SPARKET_EAGER_IMPORTS", "0") == "1":
    from . import protocol  # type: ignore
    from . import base  # type: ignore
    from . import validator  # type: ignore

