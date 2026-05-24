"""Command-line interface for yosegi-scope.

Three commands:

* ``acquire`` — fetch overlapping tiles from an OpenFlexure microscope.
* ``stitch``  — align and merge a folder of tiles into one composite.
* ``run``     — acquire then stitch in a single pass.
"""

from __future__ import annotations

from pathlib import Path

import typer

from yosegi import __version__

app = typer.Typer(
    name="yosegi",
    help="Assemble OpenFlexure microscopy mosaics from overlapping tiles.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"yosegi {__version__}")
        raise typer.Exit()


def _abort_if_unimplemented(exc: NotImplementedError) -> None:
    """Report a stubbed step as a clean error instead of a traceback."""
    typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """yosegi-scope: OpenFlexure mosaic assembly."""


@app.command()
def acquire(
    output: Path = typer.Option(..., "--output", "-o", help="Directory to save captured tiles."),
    host: str | None = typer.Option(
        None, "--host", help="Microscope hostname or IP. Omitted = mDNS auto-discovery."
    ),
    rows: int = typer.Option(3, "--rows", help="Number of grid rows to scan."),
    cols: int = typer.Option(3, "--cols", help="Number of grid columns to scan."),
    overlap: float = typer.Option(0.2, "--overlap", help="Fractional tile overlap (0-1)."),
) -> None:
    """Scan a sample and fetch overlapping tiles from the microscope."""
    from yosegi.acquire import fetch_tiles

    try:
        tiles = fetch_tiles(host=host, out_dir=output, rows=rows, cols=cols, overlap=overlap)
    except NotImplementedError as exc:
        _abort_if_unimplemented(exc)
    typer.echo(f"Captured {len(tiles)} tiles to {output}")


@app.command()
def stitch(
    input: Path = typer.Option(..., "--input", "-i", help="Directory of captured tiles."),
    output: Path = typer.Option(..., "--output", "-o", help="Path for the composite image."),
) -> None:
    """Align and merge tiles into a single seamless composite."""
    from yosegi.stitch import stitch_tiles

    try:
        result = stitch_tiles(in_dir=input, out_file=output)
    except NotImplementedError as exc:
        _abort_if_unimplemented(exc)
    typer.echo(f"Wrote {result.width}x{result.height} mosaic from {result.tile_count} tiles to {result.path}")


@app.command()
def run(
    output: Path = typer.Option(..., "--output", "-o", help="Path for the composite image."),
    host: str | None = typer.Option(None, "--host", help="Microscope hostname or IP."),
    rows: int = typer.Option(3, "--rows", help="Number of grid rows to scan."),
    cols: int = typer.Option(3, "--cols", help="Number of grid columns to scan."),
    overlap: float = typer.Option(0.2, "--overlap", help="Fractional tile overlap (0-1)."),
) -> None:
    """Acquire tiles from the microscope, then stitch them into a mosaic."""
    from yosegi.acquire import fetch_tiles
    from yosegi.stitch import stitch_tiles

    tile_dir = output.parent / f"{output.stem}_tiles"
    try:
        fetch_tiles(host=host, out_dir=tile_dir, rows=rows, cols=cols, overlap=overlap)
        result = stitch_tiles(in_dir=tile_dir, out_file=output)
    except NotImplementedError as exc:
        _abort_if_unimplemented(exc)
    typer.echo(f"Wrote {result.width}x{result.height} mosaic from {result.tile_count} tiles to {result.path}")


if __name__ == "__main__":
    app()
