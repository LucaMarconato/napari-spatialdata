"""Points with a spatial index (RFC 3): only the points in view are loaded, sampled down to a budget."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from napari.utils.events import EventedList

spatial_index = pytest.importorskip("spatialdata._io.spatial_index")

from spatialdata import SpatialData, build_points_spatial_index, read_zarr  # noqa: E402
from spatialdata.models import PointsModel  # noqa: E402

from napari_spatialdata._sdata_widgets import SdataWidget  # noqa: E402
from napari_spatialdata._tiled_points import TiledPointsLoader, get_points_spatial_index  # noqa: E402
from napari_spatialdata.constants import config  # noqa: E402
from napari_spatialdata.utils._test_utils import click_list_widget_item, get_center_pos_listitem  # noqa: E402


@pytest.fixture(params=["parquet", "zarr"])
def sdata_tiled(tmp_path, request) -> SpatialData:
    rng = np.random.default_rng(0)
    n = 50_000
    xy = rng.uniform(0, 1000, size=(n, 2))
    genes = rng.choice(["a", "b", "c", "rare"], size=n, p=[0.5, 0.3, 0.199, 0.001])
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "gene": pd.Categorical(genes)})
    points = PointsModel.parse(df, feature_key="gene", npartitions=2)
    path = tmp_path / "data.zarr"
    SpatialData(points={"transcripts": points}).write(path)
    sdata = read_zarr(path)
    build_points_spatial_index(
        sdata, "transcripts", storage=request.param, target_points_per_tile=1_000, rows_per_bucket=20_000
    )
    return read_zarr(path)


def test_loader_budget(sdata_tiled):
    index = get_points_spatial_index(sdata_tiled, "transcripts")
    assert index is not None
    loader = TiledPointsLoader(index, layer_axes=["y", "x"], max_points=5_000, feature_key="gene")
    df = loader.load()
    assert loader.last_query["n_in_view"] == 50_000
    # a uniform sample of about max_points
    assert 4_000 < len(df) < 6_000
    # a small box at full resolution
    df = loader.load(np.array([100.0, 100.0]), np.array([200.0, 200.0]))
    assert loader.last_query["budget"] == 1.0
    assert df.x.between(100, 200).all() and df.y.between(100, 200).all()
    # a rare feature: all its points, from the feature-major table
    loader.features = ["rare"]
    df = loader.load()
    assert loader.last_query["budget"] == 1.0
    assert set(df.gene) == {"rare"}
    assert len(df) == int((sdata_tiled["transcripts"]["gene"] == "rare").sum().compute())


def test_no_index_is_none(sdata_blobs):
    assert get_points_spatial_index(sdata_blobs, "blobs_points") is None


def test_tiled_points_layer_follows_camera(qtbot, make_napari_viewer, sdata_tiled, monkeypatch):
    monkeypatch.setattr(config, "POINT_THRESHOLD", 5_000)
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata_tiled]))
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")
    widget._onClick("transcripts")

    layer = viewer.layers["transcripts"]
    loader = layer.metadata["_tiled_loader"]
    assert isinstance(loader, TiledPointsLoader)
    # zoomed out: sampled down to the threshold
    assert len(layer.data) <= 6_000
    assert layer.metadata["_tiled_last_query"]["budget"] < 1.0

    # zoom in on a small region: after the debounce the layer holds all the points in view
    viewer.camera.center = (150.0, 150.0)
    viewer.camera.zoom = 20.0
    qtbot.waitUntil(lambda: layer.metadata["_tiled_last_query"]["budget"] == 1.0, timeout=5_000)
    lo, hi = layer.metadata["_tiled_last_query"]["box"]
    data = layer.data
    assert len(data) > 0
    assert np.all(data >= lo - 1e-6) and np.all(data <= hi + 1e-6)
    assert len(layer.metadata["indices"]) == len(data)

    # select a feature
    loader.set_features(["rare"])
    assert set(layer.metadata["_columns_df"]["gene"]) <= {"rare"}
