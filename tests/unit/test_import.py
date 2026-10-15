import pyfgls


def test_version_is_exposed() -> None:
    assert isinstance(pyfgls.__version__, str)
    assert pyfgls.__version__.startswith("0.")


def test_public_api_is_declared() -> None:
    assert "__version__" in pyfgls.__all__
