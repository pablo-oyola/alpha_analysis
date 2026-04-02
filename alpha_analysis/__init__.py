"""alpha_analysis package.

Expose a compact public API while avoiding eager imports of heavy optional
dependencies such as ASCOT/a5py and DESC.
"""

from importlib import import_module
import logging

__version__ = "0.0.0"

logger = logging.getLogger("alpha_analysis")

_EXPORTS = {
    "desc_field": ("._load", "desc_field"),
    "desc_LCFS": ("._load", "desc_LCFS"),
    "get_symmetry": ("._load", "get_symmetry"),
    "convert_flux_to_cylindrical": ("._load", "convert_flux_to_cylindrical"),
    "Poincare": ("._run_poincare", "Poincare"),
    "distrz2distrho": (".utils", "distrz2distrho"),
    "RunItem": ("._create_run", "RunItem"),
    "duplicate_run_with_new_options": ("._create_run", "duplicate_run_with_new_options"),
    "transform2Epitch": ("._dist5d_epitch", "transform2Epitch"),
    "transform2E": ("._dist5d_epitch", "transform2E"),
    "get_ascot_info": ("._git", "get_ascot_info"),
    "get_logger": ("._logger", "get_logger"),
    "ResultItem": ("._result_item", "ResultItem"),
}

__all__ = ["__version__", "logger", *_EXPORTS]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
