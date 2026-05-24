"""Smoke tests for the yosegi CLI surface."""

from typer.testing import CliRunner

from yosegi import __version__
from yosegi.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("acquire", "stitch", "run"):
        assert command in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_stitch_stub_exits_cleanly() -> None:
    result = runner.invoke(app, ["stitch", "--input", "x", "--output", "y"])
    assert result.exit_code == 1
    assert "not implemented" in result.output.lower()


def test_acquire_rejects_invalid_grid() -> None:
    # --rows 0 is rejected by the option's min=1 before any scope connection.
    result = runner.invoke(app, ["acquire", "-o", "x", "--rows", "0", "--step-x", "1", "--step-y", "1"])
    assert result.exit_code == 2
    assert "rows" in result.output.lower()
