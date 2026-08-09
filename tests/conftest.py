"""
Shared pytest configuration.

Loads the project's .env before test collection so that the integration
tests' `skipif(not os.getenv("OPENAI_API_KEY"))` guard sees a locally
configured key and actually runs. In CI there is no .env, so this is a no-op
and the integration tests skip as intended.
"""

import os
import sys

from dotenv import load_dotenv

# Make the project root importable (app, rag, agent_pipeline, security, ...).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
