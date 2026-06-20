"""Fully-local dictation for macOS: hold a hotkey, speak, get cleaned text at your cursor."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sotto")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+dev"
