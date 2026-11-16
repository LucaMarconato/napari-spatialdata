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

    # --- data -----------------------------------------------------------------------------------------------------
    def _index_box(self, lo: np.ndarray | None, hi: np.ndarray | None) -> tuple[Any, Any]:
        if lo is None or hi is None:
            return None, None
        ax = list(self.index.metadata.axes)
        pos = {a: i for i, a in enumerate(self.layer_axes)}
        return np.array([lo[pos[a]] for a in ax]), np.array([hi[pos[a]] for a in ax])

    def count(self, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> int:
        bmin, bmax = self._index_box(lo, hi)
        if self.features and self.feature_key in self.index.metadata.feature_keys:
            cats = self.index.categories(self.feature_key)
            lut = {c: i for i, c in enumerate(cats)}
            codes = [lut[f] for f in self.features if f in lut]
            s, e = self.index.feature_ranges(self.feature_key, codes, bmin, bmax, 0)
        else:
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
        self.last_query = {"n_in_view": n, "budget": budget, "n_loaded": len(df), "box": (lo, hi)}
        return df

    # --- napari ---------------------------------------------------------------------------------------------------
    def update_layer(self, layer: Points, df: pd.DataFrame) -> None:
        size = layer.current_size
        # not a user edit: do not let the edit caches of napari-spatialdata record it
        with layer.events.data.blocker():
            layer.data = df[self.layer_axes].to_numpy()
        layer.size = size
        layer.metadata["_n_indices"] = len(df)
        layer.metadata["indices"] = df.index.to_list()
        other = df.drop(columns=[c for c in ("x", "y") if c in df.columns])
        layer.metadata["_columns_df"] = other if other.shape[1] else None
        layer.metadata["_tiled_last_query"] = dict(self.last_query)

    def refresh(self) -> None:
        if self._viewer is None or self._layer is None:
            return
        lo, hi = view_box_in_data(self._viewer, self._layer)
        try:
            df = self.load(lo, hi)
        except Exception as e:  # noqa: BLE001 - keep the viewer alive
            logger.error(f"Loading tiled points failed: {e}")
            return
        self.update_layer(self._layer, df)

    def connect(self, viewer: Viewer, layer: Points, delay_ms: int = 150) -> None:
        """Reload the points in view after the camera moves (debounced)."""
        from qtpy.QtCore import QTimer

        self._viewer = viewer
        self._layer = layer
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self.refresh)

        def schedule(*_: Any) -> None:
            if self._layer is not None and self._layer in viewer.layers and self._layer.visible:
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


def _polygons_yx(gdf: Any) -> list[np.ndarray]:
    """Exterior of each shape (the largest part of a multipolygon), as (y, x) vertices."""
    out = []
    for g in gdf.geometry:
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        xy = np.asarray(g.exterior.coords)[:-1]
        out.append(xy[:, ::-1])
    return out


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
        with layer.events.data.blocker():
            layer.data = []
            if len(gdf):
                layer.add_polygons(_polygons_yx(gdf))
        rows = gdf["__row__"].to_numpy() if len(gdf) else np.empty(0, dtype=np.int64)
        layer.metadata["_n_indices"] = len(gdf)
        layer.metadata["indices"] = list(gdf.index) if len(gdf) else []
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

    def connect(self, viewer: Viewer, layer: Any, delay_ms: int = 150) -> None:
        from qtpy.QtCore import QTimer

        self._viewer = viewer
        self._layer = layer
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self.refresh)

        def schedule(*_: Any) -> None:
            if self._layer is not None and self._layer in viewer.layers and self._layer.visible:
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
