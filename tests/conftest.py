"""Test isolation.

Force a clean, hermetic environment BEFORE the app (and its Settings) are
imported, so a developer's local ``.env`` (which may select the real GPU
backend or enable auth) never leaks into the test run. Environment variables
take precedence over ``.env`` in pydantic-settings, so setting them here wins.
"""
import os

# ---- TTS isolation ---------------------------------------------------------
os.environ["TTS_BACKEND"] = "mock"      # never load a real model in tests
os.environ["TTS_BACKENDS"] = ""         # no extra backends enabled
os.environ["TTS_API_KEYS"] = ""         # auth disabled (override .env)
os.environ["TTS_MODEL_ID"] = ""         # -> mock's default model
os.environ["TTS_MODELS"] = ""
os.environ["TTS_SAMPLE_RATE"] = "24000"
os.environ["TTS_EMIT_MARKS"] = "1"     # WS marks enabled (default; override .env)

# ---- SST isolation ---------------------------------------------------------
os.environ["SST_BACKEND"] = "mock"      # never load a real model in tests
os.environ["SST_BACKENDS"] = ""         # no extra backends enabled
os.environ["SST_API_KEYS"] = ""         # auth disabled (override .env)
os.environ["SST_MODEL_ID"] = ""         # -> mock's default model
os.environ["SST_MODELS"] = ""
