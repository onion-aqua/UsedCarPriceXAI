"""Independent checks on saved CSVs, split membership, metrics, and SHAP outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
data_path = ROOT / "data/processed_capital_cars.csv"
data = pd.read_csv(data_path, dtype={"id": str, "sku_id": str})
split = pd.read_csv(ROOT / "reports/split_manifest.csv", dtype={"id": str, "sku_id": str})
pred = pd.read_csv(ROOT / "reports/test_predictions.csv", dtype={"id": str, "sku_id": str})
meta = json.loads((ROOT / "reports/model_metadata.json").read_text(encoding="utf-8"))
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == meta["input_sha256"]
assert len(data) == len(split) == meta["data_rows"]
assert not data.sku_id.duplicated().any() and not split.sku_id.duplicated().any()
assert set(data.id) == set(split.id)
assert set(pred.id) == set(split.loc[split.partition.eq("test"), "id"])
np.testing.assert_allclose(np.expm1(data.log_price), data.price, rtol=1e-8)
np.testing.assert_allclose(data.vehicle_age_current, 2026-data.registration_year, equal_nan=True)
assert set(data.fuelType.dropna()) <= {"纯电动", "插电式混合动力", "增程式"}, "Inspect source energy labels before accepting new categories"
assert data.fuelType.notna().sum() > 0 and data.power.notna().sum() > 0
assert (data.power.dropna()>0).all()
assert not set(meta["features"]) & {"price", "log_price", "vehicle_age_current", "id", "sku_id", "source_flag_unknown"}
merged = pred.merge(data[["id", "price"]], on="id", suffixes=("_pred", "_source"), validate="one_to_one")
np.testing.assert_allclose(merged.price_pred, merged.price_source)
error = pred.price - pred.predicted_price
computed = {"mae_wan": float(abs(error).mean()), "rmse_wan": float(np.sqrt(np.mean(error**2))), "r2": float(1-np.sum(error**2)/np.sum((pred.price-pred.price.mean())**2))}
reported = next(row for row in meta["metrics"] if row["model"]==meta["selected_tree_model"] and row["partition"]=="test")
for name, value in computed.items():
    assert np.isclose(value, reported[name], rtol=1e-9)
assert np.isfinite(pred[["predicted_price", "interval_lower_90", "interval_upper_90"]]).all().all()
assert (pred.interval_lower_90 <= pred.interval_upper_90).all()
coverage = ((pred.price>=pred.interval_lower_90)&(pred.price<=pred.interval_upper_90)).mean()
assert np.isclose(coverage, meta["conformal"]["test_coverage"])
values = pd.read_csv(ROOT / "reports/shap_values.csv", dtype={"id": str})
explained = values.merge(pred[["id", "predicted_price"]], on="id", validate="one_to_one")
native = np.log1p(explained.predicted_price) if meta["target_scale"]=="log1p" else explained.predicted_price
np.testing.assert_allclose(explained[meta["features"]].sum(axis=1)+meta["shap"]["expected_value"], native, rtol=1e-5, atol=1e-5)
checks = {"saved_files_match_training_input": True, "unique_sku_and_disjoint_splits": True, "price_and_age_transformations": True, "unavailable_fields_not_fabricated": True, "metrics_independently_recomputed": True, "interval_coverage_recomputed": True, "saved_shap_additivity": True, "rows": len(data), "test_rows": len(pred)}
(ROOT / "reports/verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(checks, ensure_ascii=False))
