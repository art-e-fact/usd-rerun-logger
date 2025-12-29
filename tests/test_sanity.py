import importlib


def test_import_package() -> None:
    """Minimal smoke test ensuring the installed package can be imported."""

    module = importlib.import_module("usd_rerun_logger")
    assert module is not None
