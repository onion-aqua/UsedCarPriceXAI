"""First-week leakage-safe price baseline and TreeSHAP demonstration.

Run from repository root: .venv/Scripts/python.exe week1/src/train_baseline.py
All reported prediction errors use the original price scale: ten thousand CNY.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
CATEGORICAL = ["brand", "model", "variant", "city", "gearbox", "emission_standard", "fuelType"]
NUMERIC = ["age_at_listing_months", "mileage_km", "annual_mileage_km", "engine_liters", "wheelbase_mm", "power"]
FEATURES = CATEGORICAL + NUMERIC
LABELS = {
    "brand": "品牌", "model": "车系", "variant": "配置版本", "city": "车源城市",
    "gearbox": "变速箱", "emission_standard": "排放标准", "age_at_listing_months": "挂牌时车龄（月）",
    "mileage_km": "表显里程（公里）", "annual_mileage_km": "年均里程（公里/年）",
    "engine_liters": "发动机排量（升）", "wheelbase_mm": "轴距（毫米）",
    "fuelType": "能源类型（部分记录有值）", "power": "最大功率（kW）",
}


def metrics(actual, predicted):
    return {"mae_wan": float(mean_absolute_error(actual, predicted)),
            "rmse_wan": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted))}


def frame_for_tree(frame, categories):
    out = frame[FEATURES].copy()
    for col in CATEGORICAL:
        values = out[col].fillna("__MISSING__").astype(str)
        # Explicitly map unseen labels to missing; never expand vocabulary using holdout rows.
        out[col] = pd.Categorical(values.where(values.isin(categories[col])), categories=categories[col])
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed_capital_cars.csv")
    parser.add_argument("--max-rows", type=int, default=0, help="0 uses every eligible record")
    args = parser.parse_args()
    for folder in ["models", "reports", "figures"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    df = pd.read_csv(args.input, dtype={"id": str, "sku_id": str})
    df = df.loc[df.price.notna() & np.isfinite(df.price) & (df.price > 0)].copy()
    assert df.sku_id.notna().all() and not df.sku_id.duplicated().any(), "Duplicate/missing SKU must be resolved before splitting"
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=SEED)
    df = df.reset_index(drop=True)
    for col in CATEGORICAL:
        df[col] = df[col].fillna("__MISSING__").astype(str)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Independent validation (model choice), calibration (interval), and test partitions.
    train, rest = train_test_split(df.index.to_numpy(), test_size=0.40, random_state=SEED)
    valid, rest = train_test_split(rest, train_size=0.375, random_state=SEED + 1)
    calib, test = train_test_split(rest, train_size=0.40, random_state=SEED + 2)
    partitions = {"train": train, "validation": valid, "calibration": calib, "test": test}
    for name, idx in partitions.items():
        df.loc[idx, "partition"] = name
    for i, a in enumerate(partitions):
        for b in list(partitions)[i + 1:]:
            assert set(df.loc[partitions[a], "sku_id"]).isdisjoint(df.loc[partitions[b], "sku_id"])
    df[["id", "sku_id", "partition"]].to_csv(ROOT / "reports/split_manifest.csv", index=False, encoding="utf-8-sig")
    y = df.price.to_numpy(dtype=float)
    baseline = Pipeline([
        ("features", ColumnTransformer([
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, max_categories=300), CATEGORICAL),
        ])),
        ("ridge", Ridge(alpha=10.0, solver="lsqr")),
    ])
    print(f"Training Ridge on {len(train):,} records", flush=True)
    baseline.fit(df.loc[train, FEATURES], np.log1p(y[train]))
    results = []
    def back_transform(prediction, target):
        values = np.expm1(prediction) if target == "log1p" else prediction
        return np.maximum(0, values)
    for part in ["validation", "test"]:
        idx = partitions[part]
        pred = back_transform(baseline.predict(df.loc[idx, FEATURES]), "log1p")
        assert np.isfinite(pred).all()
        results.append({"model": "Ridge_onehot_log1p", "partition": part, **metrics(y[idx], pred)})
    categories = {col: sorted(df.loc[train, col].unique().tolist()) for col in CATEGORICAL}
    tree_frames = {key: frame_for_tree(df.loc[idx], categories) for key, idx in partitions.items()}
    models = {}
    for target in ["raw", "log1p"]:
        print(f"Training LightGBM target={target}", flush=True)
        model = lgb.LGBMRegressor(n_estimators=220, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=35, reg_lambda=1.0, random_state=SEED,
                                 n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
        model.fit(tree_frames["train"], np.log1p(y[train]) if target == "log1p" else y[train], categorical_feature=CATEGORICAL)
        models[target] = model
        for part in ["validation", "test"]:
            pred = back_transform(model.predict(tree_frames[part]), target)
            assert np.isfinite(pred).all()
            results.append({"model": f"LightGBM_{target}", "partition": part, **metrics(y[partitions[part]], pred)})
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(ROOT / "reports/model_metrics.csv", index=False, encoding="utf-8-sig")
    candidates = result_frame[(result_frame.partition == "validation") & result_frame.model.str.startswith("LightGBM")]
    selected_name = candidates.sort_values("mae_wan").iloc[0].model
    target = selected_name.removeprefix("LightGBM_")
    model = models[target]
    test_native = model.predict(tree_frames["test"])
    test_pred = back_transform(test_native, target)
    cal_native = model.predict(tree_frames["calibration"])
    cal_actual = np.log1p(y[calib]) if target == "log1p" else y[calib]
    # Split conformal: 90% marginal coverage under exchangeability, not an anomaly truth label.
    alpha = 0.10
    scores = np.abs(cal_actual - cal_native)
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError("Calibration set too small for finite split-conformal bound at requested alpha")
    radius = float(np.partition(scores, rank - 1)[rank - 1])
    lower = back_transform(test_native - radius, target)
    upper = back_transform(test_native + radius, target)
    assert np.all(lower <= upper)
    predicted = df.loc[test, ["id", "sku_id", "city", "brand", "model", "variant", "price"]].copy()
    predicted["predicted_price"] = test_pred
    predicted["interval_lower_90"] = lower
    predicted["interval_upper_90"] = upper
    predicted["relative_residual"] = (y[test] - test_pred) / np.maximum(test_pred, 0.01)
    predicted["screening_label"] = np.where(y[test] > upper, "疑似高价", np.where(y[test] < lower, "疑似低价", "区间内"))
    predicted["absolute_residual"] = np.abs(y[test] - test_pred)
    predicted.to_csv(ROOT / "reports/test_predictions.csv", index=False, encoding="utf-8-sig")
    # Global sample is random; diagnostic local example is deliberately the largest residual.
    rng = np.random.default_rng(SEED)
    sample_pos = rng.choice(len(test), size=min(256, len(test)), replace=False)
    local_pos = int(np.argmax(predicted.absolute_residual.to_numpy()))
    explain_pos = np.unique(np.append(sample_pos, local_pos))
    explain_frame = tree_frames["test"].iloc[explain_pos]
    print(f"Computing TreeSHAP on {len(explain_frame)} held-out records", flush=True)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(explain_frame))
    expected = float(np.asarray(explainer.expected_value).ravel()[0])
    explained_predictions = model.predict(explain_frame)
    max_additivity_error = float(np.max(np.abs(expected + values.sum(axis=1) - explained_predictions)))
    assert np.allclose(expected + values.sum(axis=1), explained_predictions, rtol=1e-5, atol=1e-5), "SHAP additivity check failed"
    lookup = {int(pos): row for row, pos in enumerate(explain_pos)}
    global_values = values[[lookup[int(pos)] for pos in sample_pos]]
    importance = pd.DataFrame({"feature": FEATURES, "label": [LABELS[x] for x in FEATURES], "mean_abs_shap": np.abs(global_values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(ROOT / "reports/shap_global.csv", index=False, encoding="utf-8-sig")
    local_values = values[lookup[local_pos]]
    local = pd.DataFrame({"feature": FEATURES, "label": [LABELS[x] for x in FEATURES], "value": [str(explain_frame.iloc[lookup[local_pos]][x]) for x in FEATURES], "shap_value": local_values})
    local.to_csv(ROOT / "reports/shap_local.csv", index=False, encoding="utf-8-sig")
    shap_table = pd.DataFrame(values, columns=FEATURES)
    shap_table.insert(0, "id", df.loc[test[explain_pos], "id"].to_numpy())
    shap_table.to_csv(ROOT / "reports/shap_values.csv", index=False, encoding="utf-8-sig")
    unit = "log(1 + 挂牌价/万元)" if target == "log1p" else "挂牌价（万元）"
    fig, ax = plt.subplots(figsize=(10, 5.8))
    show = importance.iloc[::-1]
    ax.barh(show.label, show.mean_abs_shap, color="#2563a6")
    ax.set(xlabel=f"平均 |SHAP|，模型输出单位：{unit}", title=f"挂牌价格模型的特征贡献（测试集随机 {len(sample_pos)} 辆）")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/shap_global.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    order = np.argsort(np.abs(local_values))
    ax.barh([LABELS[FEATURES[i]] for i in order], local_values[order], color=["#c44e52" if x > 0 else "#2563a6" for x in local_values[order]])
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set(xlabel=f"SHAP贡献：{unit}", title="单车价格预测解释（测试集中绝对残差最大的案例）")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/shap_local.png", dpi=170)
    plt.close(fig)
    coverage = float(np.mean((y[test] >= lower) & (y[test] <= upper)))
    metadata = {
        "seed": SEED, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "data_rows": len(df), "split_sizes": {key: len(idx) for key, idx in partitions.items()},
        "features": FEATURES, "categorical_features": CATEGORICAL, "category_levels_train": categories,
        "selected_tree_model": selected_name, "selection_rule": "lowest validation MAE among two LightGBM target variants; Ridge is a separate baseline",
        "target_scale": target, "price_unit": "万元", "metrics": results,
        "conformal": {"nominal_coverage": 0.9, "calibration_radius": radius, "calibration_scale": target, "test_coverage": coverage, "mean_interval_width_wan": float(np.mean(upper - lower))},
        "shap": {"random_global_sample_size": len(sample_pos), "max_additivity_error": max_additivity_error, "expected_value": expected, "local_case_id": predicted.iloc[local_pos].id, "output_scale": unit},
        "versions": {pkg: importlib.metadata.version(pkg) for pkg in ["pandas", "numpy", "scikit-learn", "lightgbm", "shap", "matplotlib"]},
        "checks": {"sku_disjoint_partitions": True, "all_predictions_finite": True, "shap_additivity_pass": True, "target_derived_predictors_absent": True},
        "limits": ["First-week fixed holdout baseline; no cross-validation or tuned hyperparameters", "Intervals identify statistically unusual listings, not confirmed fraud or abnormality", "No gold abnormality labels; Precision/Recall/F1 not computed", "Collection date not independently authenticated", "No training data winsorization/removal based on high price"],
    }
    (ROOT / "reports/model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "models/price_baseline.pkl").open("wb") as handle:
        pickle.dump({"model": model, "categories": categories, "features": FEATURES, "target": target, "radius": radius, "ridge": baseline}, handle)
    lines = ["# 第一周：数据→模型→解释运行报告", "", "本轮用于验证链路，尚不是最终论文实验或异常检测准确率。", "", "## 数据划分与编码", "",
             f"总计 {len(df):,} 条去重车源。" + "；".join(f"{key}={len(idx):,}" for key, idx in partitions.items()) + "。",
             "同一 SKU 不跨集合。缺失值填补、One-Hot类别与原生类别词表均只在训练集拟合。Ridge 使用 One-Hot；LightGBM 将 pandas category 显式作为类别处理，不把品牌编号视为连续大小。",
             "fuelType 和 power 并非独立原始列，但可从嵌套字段部分提取。fuelType缺失单列未知，不能假定为燃油；power单位kW，Ridge仅用训练中位数插补并添加缺失指示，LightGBM原生处理数值缺失。原始价格与log1p目标以相同划分对比，所有MAE/RMSE均还原到万元计算。", "",
             "## 模型结果", "", "| 模型 | 集合 | MAE（万元） | RMSE（万元） | R² |", "|---|---|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['model']} | {row['partition']} | {row['mae_wan']:.4f} | {row['rmse_wan']:.4f} | {row['r2']:.4f} |")
    local_case = predicted.iloc[local_pos]
    lines += ["", f"按验证集 MAE，在两个树模型中选择 {selected_name} 用于SHAP和区间演示。没有用测试集选模型。原价模型RMSE更低时，应保留这一取舍，不宣称对数模型全面更优。", "", "## 初步区间与解释", "",
              f"独立校准集构建90% split-conformal区间，测试集覆盖率 {coverage:.2%}，平均区间宽度 {np.mean(upper-lower):.4f} 万元。其覆盖性质依赖可交换性，不意味着90%的异常识别准确率。",
              f"SHAP全局图随机使用 {len(sample_pos)} 个测试样本，局部案例 {predicted.iloc[local_pos].id}，加和校验最大误差 {max_additivity_error:.3g}。",
              f"局部案例：{local_case.city}，{local_case.brand} / {local_case.model} / {local_case.variant}。实际挂牌 {local_case.price:.2f} 万元，预测 {local_case.predicted_price:.2f} 万元，90%区间 [{local_case.interval_lower_90:.2f}, {local_case.interval_upper_90:.2f}] 万元，初筛为{local_case.screening_label}。这是按残差选取的极端示例，不能视为代表性样本。",
              f"SHAP输出尺度为 {unit}，不可把对数SHAP值当作万元贡献。特征贡献解释模型预测，不能证明真实异常原因。", "",
              "## 后续工作", "", "完成采集日期来源证据；扩展分组交叉验证；建立独立盲审标签；检验不同城市与价位的误差和覆盖率；再设计正式异常评分。",
              "使用同期车源作为核验依据，不能直接拿2026年平台现价判断疑似2024年快照的异常。人工标签不得以模型判定作为必要条件，以免循环验证。"]
    (ROOT / "reports/model_baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected_name, "test": metrics(y[test], test_pred), "coverage": coverage, "checks": metadata["checks"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
