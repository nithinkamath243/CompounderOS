"""OpenTrim launcher.

Legacy monolith moved to opentrim_v4_legacy.py.
"""

import os
import sys

try:
    from opentrim_app.main import run
except ModuleNotFoundError:
    # Fallback when launched from outside the project directory.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from opentrim_app.main import run


if __name__ == "__main__":
    run()
