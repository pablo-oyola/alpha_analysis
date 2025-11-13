"""alpha_analysis package

Expose package version and a compact public API.
"""
__version__ = "0.0.0"

import logging
from ._load import desc_field, desc_LCFS
from ._run_poincare import Poincare

logger = logging.getLogger('alpha_analysis')
