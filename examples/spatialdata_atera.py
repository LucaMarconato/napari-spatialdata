"""Open the Atera datasets, with and without the spatial index (RFC 3 / RFC 4), in napari-spatialdata.

Usage:

    python examples/spatialdata_atera.py                       # list the datasets
    python examples/spatialdata_atera.py kidney-tiled          # indexed kidney: transcripts (Parquet + Zarr) and cells
    python examples/spatialdata_atera.py breast-parquet --max-points 500000

Indexed points load only what is in view, sampled down to ``--max-points``, and reload when the camera moves;
indexed cells load by view when at most ``--max-shapes`` are in view (zoom in to see them).
"""
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   # the experimental spatial index (RFC 3 points, RFC 4 shapes): LucaMarconato/spatialdata#13
#   "spatialdata @ git+https://github.com/LucaMarconato/spatialdata@tiled-points-spatial-index",
#   # loading indexed points/shapes by view: LucaMarconato/napari-spatialdata#1
#   "napari-spatialdata @ git+https://github.com/LucaMarconato/napari-spatialdata@tiled-points-spatial-index",
#   "napari[pyqt6]",
#   "rich",
# ]
#
# [tool.uv]
# # the fork has no release tags, so the branch builds as spatialdata 0.1.devN; override napari-spatialdata's
# # "spatialdata>=0.7.2" with the branch
# override-dependencies = [
#   "spatialdata @ git+https://github.com/LucaMarconato/spatialdata@tiled-points-spatial-index",
# ]
#
# # To use the local checkouts instead (editable, picks up uncommitted changes), uncomment:
# # [tool.uv.sources]
# # spatialdata = { path = "/Users/macbook/embl/projects/basel/spatialdata-tiled-index", editable = true }
# # napari-spatialdata = { path = "/Users/macbook/embl/projects/basel/napari-spatialdata-tiled-index", editable = true }
# ///

from __future__ import annotations

import argparse
from pathlib import Path

import spatialdata as sd

from napari_spatialdata import Interactive
from napari_spatialdata.constants import config

DATA = Path("/Volumes/Scratch/tiled-index-data")
BREAST = Path("/Volumes/Data/atera/atera_breast_cancer_io/data.zarr")

# name -> (description, [(store, selection or None for everything)])
DATASETS: dict[str, tuple[str, list[tuple[Path, tuple[str, ...] | None]]]] = {
    "kidney-tiled": (
        "kidney (spatialdata-db): indexed transcripts (Parquet and Zarr), indexed cells, table with X_csc",
        [(DATA / "kidney.zarr", None)],
    ),
    # "kidney-original": (
    #     "kidney (spatialdata-db) as downloaded, no index (points are subsampled in memory: slow to open)",
    #     [(DATA / "spatialdata_db_atera_kidney.zarr", ("points", "shapes", "tables"))],
    # ),
    "breast-parquet": (
        "breast (raw Atera): indexed transcripts, Parquet (points format 0.3)",
        [(DATA / "breast_parquet.zarr", None)],
    ),
    "breast-zarr": (
        "breast (raw Atera): indexed transcripts, Zarr (points format 0.3-experimental)",
        [(DATA / "breast_zarr.zarr", None)],
    ),
    "breast-parquet-with-image": (
        "breast: indexed transcripts (Parquet) + morphology image, cells and table of the original (external drive)",
        [(DATA / "breast_parquet.zarr", None), (BREAST, ("images", "shapes", "tables"))],
    ),
    # "breast-original": (
    #     "breast as converted by spatialdata-io, no index (external drive; points subsampled in memory: slow to open)",
    #     [(BREAST, None)],
    # ),
}


def _read(store: Path, selection: tuple[str, ...] | None) -> sd.SpatialData:
    if not store.exists():
        raise FileNotFoundError(f"{store} does not exist (is the external drive mounted?)")
    sdata = sd.read_zarr(store, selection=selection) if selection else sd.read_zarr(store)
    for table in sdata.tables.values():
        # spatialdata-db tables have an obs index named "barcode" that is also an obs column, which napari-spatialdata
        # cannot join (a pre-existing issue, unrelated to the spatial index)
        if table.obs.index.name is not None and table.obs.index.name in table.obs.columns:
            table.obs.index.name = None
    return sdata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", choices=list(DATASETS))
    ap.add_argument("--max-points", type=int, default=config.POINT_THRESHOLD, help="points loaded per view")
    ap.add_argument("--max-shapes", type=int, default=config.TILED_SHAPES_MAX, help="cells loaded per view")
    args = ap.parse_args()
    if args.dataset is None:
        for name, (description, _) in DATASETS.items():
            print(f"{name:28s} {description}")
        return

    config.POINT_THRESHOLD = args.max_points
    config.TILED_SHAPES_MAX = args.max_shapes
    sdatas = [_read(store, selection) for store, selection in DATASETS[args.dataset][1]]
    for s in sdatas:
        print(s)
    Interactive(sdatas if len(sdatas) > 1 else sdatas[0]).run()


if __name__ == "__main__":
    main()
