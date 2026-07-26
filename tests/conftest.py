"""Test isolation.

Force a clean, hermetic environment BEFORE the app (and its Settings) are
imported, so a developer's local ``.env`` (which may select the real GPU
backend or enable auth) never leaks into the test run. Environment variables
take precedence over ``.env`` in pydantic-settings, so setting them here wins.
"""
import os

os.environ["TTS_BACKEND"] = "mock"      # never load a real model in tests
os.environ["TTS_BACKENDS"] = ""         # no extra backends enabled
os.environ["TTS_API_KEYS"] = ""         # auth disabled
os.environ["TTS_MODEL_ID"] = ""         # -> mock's default model
os.environ["TTS_MODELS"] = ""
os.environ["TTS_SAMPLE_RATE"] = "24000"
