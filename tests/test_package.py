"""Package smoke tests."""

from klip_ppo import __version__


def test_version_is_defined() -> None:
    """The package exposes a version string for import smoke tests."""
    assert __version__ == "0.0.0"
