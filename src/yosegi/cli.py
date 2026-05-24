"""Command-line interface for yosegi-scope.

Three commands:

* ``acquire`` — fetch overlapping tiles from an OpenFlexure microscope.
* ``stitch``  — align and merge a folder of tiles into one composite.
* ``run``     — acquire then stitch in a single pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

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


def _abort(exc: Exception) -> NoReturn:
    """Report a step failure as a clean one-line error instead of a traceback."""
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
    rows: int = typer.Option(3, "--rows", min=1, help="Number of grid rows to scan."),
    cols: int = typer.Option(3, "--cols", min=1, help="Number of grid columns to scan."),
    step_x: int = typer.Option(2000, "--step-x", help="Stage steps to move in X between tiles."),
    step_y: int = typer.Option(2000, "--step-y", help="Stage steps to move in Y between tiles."),
    autofocus: bool = typer.Option(
        True, "--autofocus/--no-autofocus", help="Autofocus at each tile before capture."
    ),
    overlap: float = typer.Option(0.2, "--overlap", help="Fractional tile overlap (metadata only)."),
) -> None:
    """Scan a sample and fetch overlapping tiles from the microscope."""
    from yosegi.acquire import AcquisitionError, fetch_tiles

    try:
        tiles = fetch_tiles(
            host=host,
            out_dir=output,
            rows=rows,
            cols=cols,
            step_x=step_x,
            step_y=step_y,
            autofocus=autofocus,
            overlap=overlap,
        )
    except AcquisitionError as exc:
        _abort(exc)
    typer.echo(f"Captured {len(tiles)} tiles to {output}")


@app.command()
def stitch(
    input: Path = typer.Option(..., "--input", "-i", help="Directory of captured tiles."),
    output: Path = typer.Option(..., "--output", "-o", help="Path for the composite image."),
    correlate: bool = typer.Option(
        True, "--correlate/--no-correlate", help="Refine placement with phase correlation (vs stage-only)."
    ),
    high_pass_sigma: float = typer.Option(
        10.0, "--high-pass-sigma", help="High-pass filter sigma for correlation; lower for faint samples."
    ),
    minimum_overlap: float = typer.Option(
        0.2, "--minimum-overlap", help="Minimum fractional overlap for a tile pair to be correlated."
    ),
) -> None:
    """Merge tiles into a composite with openflexure-stitching."""
    from yosegi.stitch import StitchError, stitch_tiles

    try:
        result = stitch_tiles(
            in_dir=input, out_file=output, correlate=correlate,
            high_pass_sigma=high_pass_sigma, minimum_overlap=minimum_overlap,
        )
    except StitchError as exc:
        _abort(exc)
    typer.echo(f"Wrote {result.width}x{result.height} mosaic from {result.tile_count} tiles to {result.path}")


@app.command()
def run(
    output: Path = typer.Option(..., "--output", "-o", help="Path for the composite image."),
    host: str | None = typer.Option(None, "--host", help="Microscope hostname or IP."),
    rows: int = typer.Option(3, "--rows", min=1, help="Number of grid rows to scan."),
    cols: int = typer.Option(3, "--cols", min=1, help="Number of grid columns to scan."),
    step_x: int = typer.Option(2000, "--step-x", help="Stage steps to move in X between tiles."),
    step_y: int = typer.Option(2000, "--step-y", help="Stage steps to move in Y between tiles."),
    autofocus: bool = typer.Option(
        True, "--autofocus/--no-autofocus", help="Autofocus at each tile before capture."
    ),
    overlap: float = typer.Option(0.2, "--overlap", help="Fractional tile overlap (metadata only)."),
    correlate: bool = typer.Option(
        True, "--correlate/--no-correlate", help="Refine placement with phase correlation (vs stage-only)."
    ),
    high_pass_sigma: float = typer.Option(
        10.0, "--high-pass-sigma", help="High-pass filter sigma for correlation; lower for faint samples."
    ),
    minimum_overlap: float = typer.Option(
        0.2, "--minimum-overlap", help="Minimum fractional overlap for a tile pair to be correlated."
    ),
) -> None:
    """Acquire tiles from the microscope, then stitch them into a mosaic."""
    from yosegi.acquire import AcquisitionError, fetch_tiles
    from yosegi.stitch import StitchError, stitch_tiles

    tile_dir = output.parent / f"{output.stem}_tiles"
    try:
        fetch_tiles(
            host=host,
            out_dir=tile_dir,
            rows=rows,
            cols=cols,
            step_x=step_x,
            step_y=step_y,
            autofocus=autofocus,
            overlap=overlap,
        )
        result = stitch_tiles(
            in_dir=tile_dir, out_file=output, correlate=correlate,
            high_pass_sigma=high_pass_sigma, minimum_overlap=minimum_overlap,
        )
    except (AcquisitionError, StitchError) as exc:
        _abort(exc)
    typer.echo(f"Wrote {result.width}x{result.height} mosaic from {result.tile_count} tiles to {result.path}")


if __name__ == "__main__":
    app()
