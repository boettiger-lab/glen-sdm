"""Pipeline pieces that need no network, plus one live end-to-end test (-m network)."""
import h3
import numpy as np
import pandas as pd
import pytest

from hexsdm import data, evaluate, outputs, pipeline


def test_user_points_to_cells():
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-122.27, 37.87]}, "properties": {}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-122.27, 37.87]}, "properties": {}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-121.5, 38.5]}, "properties": {"presence": 0}},
    ]}
    pres = data.user_occurrences({"geojson": gj})
    assert len(pres) == 1 and pres.n.iat[0] == 2
    assert len(pres.attrs["absences"]) == 1


def test_outlier_filter_drops_far_point():
    rng = np.random.default_rng(0)
    pts = [(37 + rng.normal(0, 1), -121 + rng.normal(0, 1)) for _ in range(60)] + [(52.5, 13.4)]
    h8 = [h3.str_to_int(h3.latlng_to_cell(la, ln, 8)) for la, ln in pts]
    pres = pd.DataFrame({"h8": np.array(h8, dtype="uint64"), "n": 1})
    kept, dropped = data.remove_spatial_outliers(pres)
    assert dropped == 1 and len(kept) == 60


def test_spatial_blocks_balanced():
    cells = np.array([h3.str_to_int(c) for c in h3.grid_disk(h3.latlng_to_cell(37, -120, 6), 60)], dtype="uint64")
    pres = cells[::7]
    folds, br = evaluate.spatial_blocks(cells, pres, 5)
    assert set(folds) == set(range(5))
    assert br < 6


def test_choose_resolution():
    assert data.choose_resolution(500_000, None, 2_000_000) == 8
    assert data.choose_resolution(16_000_000, None, 2_000_000) == 6
    assert data.choose_resolution(16_000_000, 7, 2_000_000) == 7


def test_run_id_stable_and_sensitive():
    a = pipeline.normalize({"occurrences": {"gbif": {"species": "X y"}}})
    b = pipeline.normalize({"occurrences": {"gbif": {"species": "X y"}}, "cv_folds": 5})
    c = pipeline.normalize({"occurrences": {"gbif": {"species": "X y"}}, "cv_folds": 3})
    assert pipeline.run_id(a) == pipeline.run_id(b) != pipeline.run_id(c)


def test_range_polygon_coarsens(monkeypatch):
    monkeypatch.setattr(outputs, "POLYGON_MAX_CELLS", 100)
    cells = np.array([h3.str_to_int(c) for c in h3.grid_disk(h3.latlng_to_cell(37, -120, 8), 20)], dtype="uint64")
    gj, res = outputs.range_polygon(cells, 8)
    assert res < 8 and gj["features"][0]["geometry"]["type"] in ("Polygon", "MultiPolygon")


@pytest.mark.network
def test_end_to_end_gbif(tmp_path, monkeypatch):
    monkeypatch.setattr(outputs, "OUTPUT_DIR", tmp_path)
    s = pipeline.run({"occurrences": {"gbif": {"species": "Ambystoma californiense"}},
                      "algorithms": ["maxent", "rf"], "cv_folds": 3, "covariates": ["climate"]},
                     log=lambda m: None)
    assert s["cv_metrics"]["ensemble"]["auc"] > 0.8
    assert (tmp_path / s["run_id"] / "range.geojson").exists()


def test_map_layers_local(tmp_path, monkeypatch):
    monkeypatch.setattr(outputs, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(outputs, "OUTPUT_S3", "")
    monkeypatch.setattr(outputs, "DISPLAY_MAX_CELLS", 500)
    cells = np.array([h3.str_to_int(c) for c in h3.grid_disk(h3.latlng_to_cell(37, -120, 8), 25)], dtype="uint64")
    pred = pd.DataFrame({"cell": cells, "suitability": np.linspace(0, 1, len(cells)).astype("float32")})
    pred["suitable"] = pred.suitability > 0.5
    gj, _ = outputs.range_polygon(pred.cell[pred.suitable].to_numpy(), 8)
    lay = outputs.map_layers("abc123", pred, 8, gj, "Test")
    assert lay["display_cells"] <= 500 and lay["display_h3_res"] < 8
    s = lay["suitability"]
    assert s["format"] == "geojson" and s["tile_url"].endswith("/tiles/hex/abc123/{z}/{x}/{y}.pbf")
    import json as _j
    meta = _j.loads((tmp_path / "tiles/hex/abc123/metadata.json").read_text())
    assert str(lay["display_h3_res"]) in meta["value_stats"]["suitability"]["by_res"]
    data = _j.loads((tmp_path / "tiles/hex/abc123/data.geojson").read_text())
    assert len(data["features"]) == lay["display_cells"]
