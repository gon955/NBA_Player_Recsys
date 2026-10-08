"""Tests for the root-level data pipeline (lineup_features.py and friends).

Kept out of tests/ on purpose: tests/conftest.py puts backend/ first on
sys.path so the app's `import helper` gets backend/helper.py, while the
pipeline needs the *root* helper.py (canonical_team, era_of, TOTAL_MARKERS).
The CI image also only copies backend/ and tests/, so these run locally:

    pytest tests_pipeline --no-cov
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
