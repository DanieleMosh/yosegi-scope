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
    auto: bool = typer.Option(
        False,
        "--auto/--no-auto",
        help="Automatically detect the sample boundary and plan the scan from a coarse overview pass "
             "(ignores --rows/--cols/--step-x/--step-y).",
    ),
    rows: int = typer.Option(
        3, "--rows", min=1, help="Number of grid rows to scan (ignored with --auto)."
    ),
    cols: int = typer.Option(
        3, "--cols", min=1, help="Number of grid columns to scan (ignored with --auto)."
    ),
    step_x: int = typer.Option(
        2000, "--step-x", help="Stage steps to move in X between tiles (ignored with --auto)."
    ),
    step_y: int = typer.Option(
        2000, "--step-y", help="Stage steps to move in Y between tiles (ignored with --auto)."
    ),
    overview_rows: int = typer.Option(
        5, "--overview-rows", min=1, help="Rows in the coarse overview raster (--auto only)."
    ),
    overview_cols: int = typer.Option(
        5, "--overview-cols", min=1, help="Cols in the coarse overview raster (--auto only)."
    ),
    overview_step_x: int = typer.Option(
        8000, "--overview-step-x", help="Stage steps between overview tiles in X (--auto only)."
    ),
    overview_step_y: int = typer.Option(
        8000, "--overview-step-y", help="Stage steps between overview tiles in Y (--auto only)."
    ),
    min_area_frac: float = typer.Option(
        0.005,
        "--min-area-frac",
        help="Minimum tissue area (fraction of overview image) for detection (--auto only).",
    ),
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
    """Acquire tiles from the microscope, then stitch them into a mosaic.

    With ``--auto`` the scope first runs a coarse overview pass, detects the
    sample boundary, plans a high-resolution scan that covers it, then runs
    that scan and stitches the final mosaic -- no ``--rows``/``--cols`` needed.
    """
    from yosegi.acquire import AcquisitionError, connect, fetch_tiles
    from yosegi.stitch import StitchError, stitch_tiles
    from yosegi.survey import SurveyError, run_auto_survey

    try:
        if auto:
            scope = connect(host)
            result = run_auto_survey(
                client=scope,
                out_file=output,
                overview_rows=overview_rows,
                overview_cols=overview_cols,
                overview_step_x=overview_step_x,
                overview_step_y=overview_step_y,
                overlap=overlap,
                autofocus=autofocus,
                correlate=correlate,
                high_pass_sigma=high_pass_sigma,
                minimum_overlap=minimum_overlap,
                min_area_frac=min_area_frac,
            )
        else:
            tile_dir = output.parent / f"{output.stem}_tiles"
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
    except (AcquisitionError, StitchError, SurveyError) as exc:
        _abort(exc)
    typer.echo(f"Wrote {result.width}x{result.height} mosaic from {result.tile_count} tiles to {result.path}")


if __name__ == "__main__":
    app()
