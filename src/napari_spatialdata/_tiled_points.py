"""Load points with a spatial index (RFC 3) lazily: only the points in view, at a level that fits a budget."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from napari import Viewer
    from napari.layers import Points
    from spatialdata import SpatialData
    from spatialdata._io.spatial_index import PointsSpatialIndex


def get_points_spatial_index(sdata: SpatialData, name: str) -> PointsSpatialIndex | None:
    """The spatial index of a points element of a disk-backed SpatialData object, if it has one."""
    if sdata.path is None:
        return None
    path = Path(sdata.path) / "points" / name
    if not (path / "zarr.json").exists():
        return None
    try:
        import zarr
        from spatialdata._io.spatial_index import PointsSpatialIndex
    except ImportError:  # spatialdata without the experimental spatial index
        return None
    attrs = zarr.open_group(path, mode="r").attrs.asdict()
    if "spatial_index" not in attrs:
        return None
    return PointsSpatialIndex(path)


def _canvas_size(viewer: Viewer) -> tuple[int, int]:
    """Canvas size (height, width) in screen pixels."""
    size = getattr(viewer, "_canvas_size", None)  # older napari
    if size is None:
        size = viewer.window._qt_viewer.canvas.size
    return int(size[0]), int(size[1])


def view_box_in_data(viewer: Viewer, layer: Points) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box (min, max) of the canvas in the data coordinates of the layer (layer axis order)."""
    zoom = float(viewer.camera.zoom)
    h, w = _canvas_size(viewer)
    center = np.asarray(viewer.camera.center, dtype=np.float64)[-2:]
    half = np.array([h, w], dtype=np.float64) / zoom / 2.0
    ndim_world = viewer.dims.ndim
    corners = []
    for dy in (-1, 1):
        for dx in (-1, 1):
            world = np.zeros(ndim_world)
            world[-2:] = center + half * np.array([dy, dx])
            corners.append(np.asarray(layer.world_to_data(world))[-2:])
    c = np.asarray(corners)
    return c.min(axis=0), c.max(axis=0)


def color_points(values: pd.Series, cmap: str = "viridis") -> np.ndarray:
    """RGBA colors of the points from one column, independent of which points are loaded.

    Categorical: the palette of napari-spatialdata (``tab20`` sampled over all the categories), indexed by the code
    of each point in the global category list, so a category keeps its color at every reload. Numeric: ``cmap`` over
    the values of the loaded points.
    """
    import matplotlib.pyplot as plt

    if isinstance(values.dtype, pd.CategoricalDtype):
        n = max(len(values.cat.categories), 1)
        palette = plt.get_cmap("tab20")(np.linspace(0, 1, n))
        codes = values.cat.codes.to_numpy()
        out = palette[np.clip(codes, 0, n - 1)]
        out[codes < 0] = (0.5, 0.5, 0.5, 1.0)
        return out
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    lo, hi = np.nanmin(v) if len(v) else 0.0, np.nanmax(v) if len(v) else 1.0
    norm = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return plt.get_cmap(cmap)(np.nan_to_num(norm))


class TiledPointsLoader:
    """Keep a napari Points layer filled with the points in view, read through the spatial index.

    The number of points in view is counted from the tile offsets (no data read); if it is above ``max_points``,
    the budget ``b = max_points / count`` selects a uniform sample (RFC 3, "Reading with a budget").
    """

    def __init__(
        self,
        index: PointsSpatialIndex,
        layer_axes: list[str],
        max_points: int,
        feature_key: str | None = None,
        features: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> None:
        self.index = index
        self.layer_axes = layer_axes  # e.g. ["y", "x"]
        self.max_points = max_points
        self.feature_key = feature_key
        self.features = features
        self.columns = columns
        self.last_query: dict[str, Any] = {}
        self._timer: Any = None
        self._viewer: Viewer | None = None
        self._layer: Points | None = None

    def point_size(self) -> float:
        """About the mean spacing of the points of a typical non-empty tile of level 0, in data units."""
        lv0 = self.index.metadata.levels[0]
        axes = [i for i, a in enumerate(self.index.metadata.axes) if a in self.layer_axes]
        tile_area = float(np.prod([lv0.tile_size[i] for i in axes]))
        try:
            o = self.index._offsets(None, 0)[:]
            counts = np.diff(o.astype(np.int64))
            # band 0 holds 1 - P(1) of the points of each tile
            frac = 1.0 - (self.index.metadata.levels[1].P if self.index.metadata.n_levels > 1 else 0.0)
            median = float(np.median(counts[counts > 0])) / max(frac, 1e-9)
        except Exception:  # noqa: BLE001
            median = self.index.n_rows / max(lv0.n_tiles, 1)
        return float(np.sqrt(tile_area / max(median, 1.0)))

    # --- data -----------------------------------------------------------------------------------------------------
    def _index_box(self, lo: np.ndarray | None, hi: np.ndarray | None) -> tuple[Any, Any]:
        if lo is None or hi is None:
            return None, None
        # the box in the order of the axes of the index; axes not shown (e.g. z of a 2-D view) span the whole grid
        md = self.index.metadata
        lv0 = md.levels[0]
        full_lo = np.asarray(lv0.origin, dtype=np.float64)
        full_hi = full_lo + np.asarray(lv0.tile_size) * np.asarray(lv0.grid_shape)
        pos = {a: i for i, a in enumerate(self.layer_axes)}
        bmin = np.array([lo[pos[a]] if a in pos else full_lo[i] for i, a in enumerate(md.axes)])
        bmax = np.array([hi[pos[a]] if a in pos else full_hi[i] for i, a in enumerate(md.axes)])
        return bmin, bmax

    def count(self, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> int:
        bmin, bmax = self._index_box(lo, hi)
        if self.features and self.feature_key in self.index.metadata.feature_keys:
            cats = self.index.categories(self.feature_key)
            lut = {c: i for i, c in enumerate(cats)}
            codes = [lut[f] for f in self.features if f in lut]
            s, e = self.index.feature_ranges(self.feature_key, codes, bmin, bmax, 0)
        else:
            if hasattr(self.index, "estimate_count"):
                # the points expected inside the view (the tiles of the coarse bands are larger than the view)
                return int(round(self.index.estimate_count(bmin, bmax, 0)))
            s, e = self.index.tile_ranges(bmin, bmax, 0)
        return int(np.sum(e - s))

    def load(self, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> pd.DataFrame:
        """Points in the box [lo, hi] (layer axis order; None = everything), at most about max_points."""
        n = self.count(lo, hi)
        budget = min(1.0, self.max_points / max(n, 1))
        bmin, bmax = self._index_box(lo, hi)
        df = self.index.query(
            bmin,
            bmax,
            budget=budget if budget < 1.0 else None,
            features=self.features,
            feature_key=self.feature_key,
            columns=self.columns,
        )
        # the feature column with the global category list, the same at every reload: Parquet row groups carry only
        # the categories they use, so without this the categories (and the colors derived from their order) would
        # change with the view
        fk = self.feature_key
        if fk is not None and fk in df.columns and fk in self.index.metadata.feature_keys:
            df[fk] = pd.Categorical(df[fk].astype(str), categories=self._global_categories())
        self.last_query = {"n_in_view": n, "budget": budget, "n_loaded": len(df), "box": (lo, hi)}
        return df

    def _global_categories(self) -> list[str]:
        if not hasattr(self, "_cats"):
            assert self.feature_key is not None
            self._cats = self.index.categories(self.feature_key)
        return self._cats

    # --- napari ---------------------------------------------------------------------------------------------------
    def update_layer(self, layer: Points, df: pd.DataFrame) -> None:
        size = layer.current_size
        # not a user edit: do not let the edit caches of napari-spatialdata record it
        # not a user edit: the edit caches of napari-spatialdata skip it (blocking the data event would also hide
        # the change from napari's own views, which then slice stale arrays: IndexError)
        layer.metadata["_tiled_updating"] = True
        try:
            layer.data = df[self.layer_axes].to_numpy()
        finally:
            layer.metadata["_tiled_updating"] = False
        layer.size = size
        layer.metadata["_n_indices"] = len(df)
        layer.metadata["indices"] = df.index.to_list()
        other = df.drop(columns=[c for c in ("x", "y") if c in df.columns])
        layer.metadata["_columns_df"] = other if other.shape[1] else None
        layer.metadata["_tiled_last_query"] = dict(self.last_query)
        # the colors belong to the points just replaced: compute them again for the new points (the coloring chosen
        # in the widgets), or reset them, instead of letting napari keep the old colors by position
        color_by = layer.metadata.get("_tiled_color_by")
        if color_by is not None and len(df) and color_by in df.columns:
            layer.face_color = color_points(df[color_by], layer.metadata.get("_tiled_cmap", "viridis"))
            return
        recolor = layer.metadata.get("_tiled_recolor")
        if callable(recolor) and len(df):
            try:
                recolor()
                return
            except Exception as e:  # noqa: BLE001 - keep the viewer alive
                logger.exception(f"Coloring the tiled points failed: {e}")
        if len(df):
            layer.face_color = layer.current_face_color

    def refresh(self) -> None:
        """Load the points in view now (synchronously)."""
        if self._viewer is None or self._layer is None:
            return
        lo, hi = view_box_in_data(self._viewer, self._layer)
        try:
            df = self.load(lo, hi)
        except Exception as e:  # noqa: BLE001 - keep the viewer alive
            logger.error(f"Loading tiled points failed: {e}")
            return
        self.update_layer(self._layer, df)

    def _refresh_async(self) -> None:
        """Load the points in view in a worker thread, then update the layer; at most one load at a time."""
        if self._viewer is None or self._layer is None:
            return
        if self._busy:
            self._pending = True
            return
        from napari.qt.threading import create_worker

        lo, hi = view_box_in_data(self._viewer, self._layer)
        self._busy = True
        self._pending = False
        worker = create_worker(self.load, lo, hi)

        def done(df: pd.DataFrame) -> None:
            if self._layer is not None:
                self.update_layer(self._layer, df)

        def finished() -> None:
            self._busy = False
            if self._pending:
                self._refresh_async()

        worker.returned.connect(done)
        worker.errored.connect(lambda e: logger.error(f"Loading tiled points failed: {e}"))
        worker.finished.connect(finished)
        worker.start()
        self._worker = worker

    def connect(self, viewer: Viewer, layer: Points, interval_ms: int = 60) -> None:
        """Reload the points in view while the camera moves.

        The reloads are throttled (at most one every ``interval_ms``, and one at a time, in a worker thread), not
        debounced: the points follow the zoom and the pan continuously. With the budget ``b = max_points / count``,
        a small change of the view changes ``b`` a little: the points with p below the old budget stay, and only the
        ones between the two budgets appear or disappear (RFC 3, "Reading with a budget").
        """
        from qtpy.QtCore import QTimer

        self._viewer = viewer
        self._layer = layer
        self._busy = False
        self._pending = False
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._refresh_async)

        def schedule(*_: Any) -> None:
            if self._layer is not None and self._layer in viewer.layers and self._layer.visible:
                if not self._timer.isActive():  # throttle: keep the scheduled reload, do not postpone it
                    self._timer.start()

        viewer.camera.events.center.connect(schedule)
        viewer.camera.events.zoom.connect(schedule)
        layer.events.visible.connect(schedule)
        self._schedule = schedule

    def disconnect(self) -> None:
        if self._viewer is not None:
            self._viewer.camera.events.center.disconnect(self._schedule)
            self._viewer.camera.events.zoom.disconnect(self._schedule)
        self._viewer = None
        self._layer = None

    def set_features(self, features: list[str] | None) -> None:
        """Show only these features (None: all) and reload."""
        self.features = features
        self.refresh()


# --- shapes (RFC 4) -----------------------------------------------------------------------------------------------
def get_shapes_spatial_index(sdata: SpatialData, name: str) -> Any:
    """The spatial index (RFC 4) of a shapes element of a disk-backed SpatialData object, if it has one."""
    if sdata.path is None:
        return None
    path = Path(sdata.path) / "shapes" / name
    if not (path / "zarr.json").exists():
        return None
    try:
        import zarr
        from spatialdata._io.spatial_index import ShapesSpatialIndex
    except ImportError:
        return None
    attrs = zarr.open_group(path, mode="r").attrs.asdict()
    if "spatial_index" not in attrs:
        return None
    table = attrs["spatial_index"].get("spatialdata_extra", {}).get("table", {}).get("name")
    return ShapesSpatialIndex(path, Path(sdata.path) / "tables" / table if table else None)


def _polygons_yx(gdf: Any) -> tuple[list[np.ndarray], np.ndarray]:
    """Exterior of every part of every shape, as (y, x) vertices, and the position of its shape in ``gdf``."""
    out, owner = [], []
    for i, g in enumerate(gdf.geometry):
        parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
        for part in parts:
            xy = np.asarray(part.exterior.coords)[:-1]
            out.append(xy[:, ::-1])
            owner.append(i)
    return out, np.asarray(owner, dtype=np.int64)


def warn_if_slow_triangulation() -> None:
    """Warn when napari triangulates the polygons without bermuda (about 30 times slower for thousands of cells)."""
    import importlib.util

    if importlib.util.find_spec("bermuda") is None:
        logger.warning(
            "bermuda is not installed: drawing the cells in view is slow (seconds for a few thousand cells). "
            "Install it with `pip install bermuda` (or `napari-spatialdata[bermuda]`)."
        )


class TiledShapesLoader:
    """Keep a napari Shapes layer filled with the shapes in view, read through the spatial index (RFC 4).

    RFC 4 has no levels: when more than ``max_shapes`` are in view the layer is emptied (zoom in to see them; the
    centroids can be shown as indexed points instead).
    """

    def __init__(self, index: Any, max_shapes: int) -> None:
        self.index = index
        self.max_shapes = max_shapes
        self.gene: str | None = None
        self.last_query: dict[str, Any] = {}
        self._viewer: Viewer | None = None
        self._layer: Any = None
        self._timer: Any = None

    def load(self, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> Any:
        """Shapes in the box [lo, hi] (layer order (y, x); None: everything), or None if there are too many."""
        import geopandas as gpd

        if lo is None or hi is None:
            lv = self.index.metadata.levels[0]
            o = np.asarray(lv.origin)
            bmin, bmax = o, o + np.asarray(lv.tile_size) * np.asarray(lv.grid_shape)
        else:
            bmin, bmax = np.asarray(lo)[::-1], np.asarray(hi)[::-1]
        n = self.index.count(bmin, bmax)
        self.last_query = {"n_in_view": n, "box": (lo, hi), "loaded": n <= self.max_shapes}
        if n > self.max_shapes:
            return gpd.GeoDataFrame(geometry=[])
        return self.index.query(bmin, bmax)

    def update_layer(self, layer: Any, gdf: Any) -> None:
        # not a user edit: do not let the edit caches of napari-spatialdata record it
        polygons, owner = _polygons_yx(gdf) if len(gdf) else ([], np.empty(0, dtype=np.int64))
        layer.metadata["_tiled_updating"] = True
        try:
            layer.data = []
            if polygons:
                layer.add_polygons(polygons)
        finally:
            layer.metadata["_tiled_updating"] = False
        # one napari polygon per part; each part carries the index and the row of its shape
        rows = gdf["__row__"].to_numpy()[owner] if len(gdf) else np.empty(0, dtype=np.int64)
        layer.metadata["_n_indices"] = len(polygons)
        layer.metadata["indices"] = list(np.asarray(gdf.index)[owner]) if len(gdf) else []
        layer.metadata["_tiled_rows"] = rows
        layer.metadata["_tiled_last_query"] = dict(self.last_query)
        if self.gene is not None and len(rows):
            self._color(layer, rows)

    def _color(self, layer: Any, rows: np.ndarray) -> None:
        from napari.utils.colormaps import ensure_colormap

        v = self.index.gene_values(self.gene, rows).astype(np.float64)
        vmax = float(v.max()) if len(v) and v.max() > 0 else 1.0
        layer.face_color = ensure_colormap("viridis").map(v / vmax)

    def color_by_gene(self, gene: str | None) -> None:
        """Color the shapes in view by the expression of a gene, read from the CSC layer of the table."""
        self.gene = gene
        if self._layer is not None:
            rows = self._layer.metadata.get("_tiled_rows", np.empty(0, dtype=np.int64))
            if gene is not None and len(rows):
                self._color(self._layer, rows)

    def refresh(self) -> None:
        if self._viewer is None or self._layer is None:
            return
        lo, hi = view_box_in_data(self._viewer, self._layer)
        try:
            gdf = self.load(lo, hi)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Loading tiled shapes failed: {e}")
            return
        self.update_layer(self._layer, gdf)

    def connect(self, viewer: Viewer, layer: Any, delay_ms: int = 100) -> None:
        from qtpy.QtCore import QTimer

        self._viewer = viewer
        self._layer = layer
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self.refresh)

        def schedule(*_: Any) -> None:
            if self._layer is not None and self._layer in viewer.layers and self._layer.visible:
                if not self._timer.isActive():  # throttle, as for the points
                    self._timer.start()

        viewer.camera.events.center.connect(schedule)
        viewer.camera.events.zoom.connect(schedule)
        self._schedule = schedule

    def disconnect(self) -> None:
        if self._viewer is not None:
            self._viewer.camera.events.center.disconnect(self._schedule)
            self._viewer.camera.events.zoom.disconnect(self._schedule)
        self._viewer = None
        self._layer = None
