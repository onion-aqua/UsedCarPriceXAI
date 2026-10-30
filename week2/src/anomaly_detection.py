"""Screen potentially unusual used-car listing prices.

This is a review queue, not a ground-truth anomaly classifier. Peer prices are
calculated from the training partition only, and the model interval comes from
the independently calibrated first-week baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEED = 2026


def load_inputs(data_path: Path, pred_path: Path, split_path: Path):
    data = pd.read_csv(data_path, dtype={"id": str, "sku_id": str})
    pred = pd.read_csv(pred_path, dtype={"id": str, "sku_id": str})
    split = pd.read_csv(split_path, dtype={"id": str, "sku_id": str})
    required_data = {"id", "sku_id", "price", "brand", "model", "city", "registration_year", "model_year", "variant", "age_at_listing_months", "mileage_km", "power", "fuelType"}
    required_pred = {"id", "sku_id", "price", "predicted_price", "interval_lower_90", "interval_upper_90", "screening_label"}
    missing = required_data - set(data.columns)
    if missing:
        raise ValueError(f"processed data missing columns: {sorted(missing)}")
    missing = required_pred - set(pred.columns)
    if missing:
        raise ValueError(f"prediction file missing columns: {sorted(missing)}")
    if data.sku_id.duplicated().any() or split.sku_id.duplicated().any():
        raise ValueError("SKU must be unique before anomaly screening")
    if not set(pred.id).issubset(set(data.id)):
        raise ValueError("Predictions contain IDs absent from processed data")
    test_ids = set(split.loc[split.partition.eq("test"), "id"])
    if set(pred.id) != test_ids:
        raise ValueError("Prediction file does not exactly match the held-out test partition")
    data = data.merge(split[["id", "partition"]], on="id", how="left", validate="one_to_one")
    scored = data.merge(pred, on=["id", "sku_id", "price"], how="inner", validate="one_to_one", suffixes=("", "_pred"))
    if len(scored) != len(pred):
        raise ValueError("Prediction merge did not preserve all scored rows")
    return data, scored


def add_peer_statistics(data: pd.DataFrame, scored: pd.DataFrame, min_peer_count: int) -> pd.DataFrame:
    train = data.loc[data.partition.eq("train")].copy()
    train["registration_year"] = pd.to_numeric(train["registration_year"], errors="coerce").astype("Int64")
    scored = scored.copy()
    scored["registration_year"] = pd.to_numeric(scored["registration_year"], errors="coerce").astype("Int64")

    levels = [
        ("city_model_year", ["city", "model", "registration_year"]),
        ("city_brand_year", ["city", "brand", "registration_year"]),
        ("city_brand", ["city", "brand"]),
        ("brand_model_year", ["brand", "model", "registration_year"]),
        ("brand_model", ["brand", "model"]),
    ]
    assigned = pd.Series(False, index=scored.index)
    for level_name, keys in levels:
        available = train.dropna(subset=keys + ["price"])
        stats = available.groupby(keys, dropna=False)["price"].agg(
            peer_count="size", peer_median="median", peer_p25=lambda s: s.quantile(.25), peer_p75=lambda s: s.quantile(.75)
        ).reset_index()
        keep = stats["peer_count"] >= min_peer_count
        stats = stats.loc[keep].copy()
        renamed = {col: f"{col}_{level_name}" for col in ["peer_count", "peer_median", "peer_p25", "peer_p75"]}
        stats = stats.rename(columns=renamed)
        matched = scored[keys].merge(stats, on=keys, how="left", sort=False)
        matched.index = scored.index
        eligible = (~assigned) & matched[f"peer_count_{level_name}"].notna()
        for source, target in renamed.items():
            if target not in scored:
                scored[target] = np.nan
            scored.loc[eligible, target] = matched.loc[eligible, target]
        scored.loc[eligible, "peer_reference_level"] = level_name
        assigned = assigned | eligible
    scored["peer_reference_level"] = scored["peer_reference_level"].fillna("none")
    scored["peer_count"] = pd.to_numeric(scored.filter(regex=r"^peer_count_").max(axis=1), errors="coerce")
    scored["peer_median"] = scored.filter(regex=r"^peer_median_").bfill(axis=1).iloc[:, 0]
    scored["peer_p25"] = scored.filter(regex=r"^peer_p25_").bfill(axis=1).iloc[:, 0]
    scored["peer_p75"] = scored.filter(regex=r"^peer_p75_").bfill(axis=1).iloc[:, 0]
    scored["peer_relative_gap"] = (scored["price"] - scored["peer_median"]) / scored["peer_median"].clip(lower=0.01)
    scored["peer_iqr_width"] = (scored["peer_p75"] - scored["peer_p25"]).clip(lower=0)
    scored["peer_comparability"] = np.select(
        [scored["peer_reference_level"].eq("city_model_year"), scored["peer_reference_level"].eq("city_brand_year"), scored["peer_reference_level"].eq("city_brand")],
        ["高", "中", "中"], default="低/仅城市或品牌"
    )
    scored["peer_reference_warning"] = np.where(
        scored["peer_comparability"].eq("低/仅城市或品牌"),
        "同类信息不足，参考价仅作背景，不直接触发同类异常信号", ""
    )
    return scored


def build_screening(scored: pd.DataFrame, peer_threshold: float) -> pd.DataFrame:
    scored = scored.copy()
    scored["interval_high_gap"] = ((scored["price"] - scored["interval_upper_90"]) / scored["interval_upper_90"].clip(lower=0.01)).clip(lower=0)
    scored["interval_low_gap"] = ((scored["interval_lower_90"] - scored["price"]) / scored["interval_lower_90"].clip(lower=0.01)).clip(lower=0)
    # City-only/brand-only medians are useful for display but too heterogeneous
    # to trigger an anomaly flag. They remain visible with a warning.
    peer_ok = (scored["peer_count"].fillna(0) >= 5) & scored["peer_comparability"].isin(["高", "中"])
    scored["peer_high_flag"] = peer_ok & (scored["peer_relative_gap"] >= peer_threshold)
    scored["peer_low_flag"] = peer_ok & (scored["peer_relative_gap"] <= -peer_threshold)
    scored["model_high_flag"] = scored["price"] > scored["interval_upper_90"]
    scored["model_low_flag"] = scored["price"] < scored["interval_lower_90"]
    scored["screening_high_flag"] = scored["model_high_flag"] | scored["peer_high_flag"]
    scored["screening_low_flag"] = scored["model_low_flag"] | scored["peer_low_flag"]
    scored["screening_label"] = np.select(
        [scored["screening_high_flag"] & scored["screening_low_flag"], scored["screening_high_flag"], scored["screening_low_flag"]],
        ["方向冲突_人工复核", "疑似异常高价_人工复核", "疑似异常低价_人工复核"], default="区间内/暂不初筛"
    )
    scored["screening_score"] = np.maximum.reduce([
        scored["interval_high_gap"].fillna(0), scored["interval_low_gap"].fillna(0),
        scored["peer_relative_gap"].abs().fillna(0).where(peer_ok, 0),
    ])
    scored["review_priority"] = np.select(
        [scored["screening_score"] >= .75, scored["screening_score"] >= .35, scored["screening_score"] > 0],
        ["高", "中", "低"], default="无"
    )
    scored["screening_reasons"] = ""
    reasons = [
        (scored["model_high_flag"], "MODEL_INTERVAL_HIGH"),
        (scored["model_low_flag"], "MODEL_INTERVAL_LOW"),
        (scored["peer_high_flag"], "PEER_PRICE_HIGH"),
        (scored["peer_low_flag"], "PEER_PRICE_LOW"),
    ]
    for mask, code in reasons:
        scored.loc[mask, "screening_reasons"] = scored.loc[mask, "screening_reasons"].where(
            scored.loc[mask, "screening_reasons"].eq(""), scored.loc[mask, "screening_reasons"] + ";"
        ) + code
    return scored


def make_review_template(scored: pd.DataFrame, output: Path, top_k: int, random_k: int, seed: int):
    high = scored.loc[scored.screening_high_flag].sort_values(["screening_score", "absolute_residual"], ascending=False).head(top_k).copy()
    low = scored.loc[scored.screening_low_flag].sort_values(["screening_score", "absolute_residual"], ascending=False).head(top_k).copy()
    normal_pool = scored.loc[scored.screening_label.eq("区间内/暂不初筛")]
    normal = normal_pool.sample(n=min(random_k, len(normal_pool)), random_state=seed).copy() if len(normal_pool) else normal_pool
    frames = []
    for source, frame in [("top_high", high), ("top_low", low), ("random_normal", normal)]:
        frame = frame.copy()
        frame.insert(0, "sample_source", source)
        frames.append(frame)
    review = pd.concat(frames, ignore_index=True)
    keep = [
        "sample_source", "id", "sku_id", "city", "brand", "model", "variant", "registration_year", "model_year",
        "age_at_listing_months", "mileage_km", "annual_mileage_km", "engine_liters", "power", "fuelType", "gearbox",
        "price", "predicted_price", "interval_lower_90", "interval_upper_90", "interval_high_gap", "interval_low_gap",
        "peer_reference_level", "peer_comparability", "peer_reference_warning", "peer_count", "peer_median", "peer_p25", "peer_p75", "peer_relative_gap", "screening_score",
        "screening_label", "screening_reasons", "review_priority", "detail_url",
    ]
    for col in keep:
        if col not in review:
            review[col] = np.nan
    review = review[keep]
    review["human_label"] = ""
    review["human_confidence"] = ""
    review["reason_codes"] = ""
    review["external_source_url_1"] = ""
    review["external_source_url_2"] = ""
    review["query_date"] = ""
    review["review_notes"] = ""
    review.to_csv(output, index=False, encoding="utf-8-sig")
    return {"top_high": len(high), "top_low": len(low), "random_normal": len(normal)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "week1/data/processed_capital_cars.csv")
    parser.add_argument("--predictions", type=Path, default=ROOT / "week1/reports/test_predictions.csv")
    parser.add_argument("--split", type=Path, default=ROOT / "week1/reports/split_manifest.csv")
    parser.add_argument("--min-peer-count", type=int, default=8)
    parser.add_argument("--peer-threshold", type=float, default=.35)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--random-k", type=int, default=30)
    args = parser.parse_args()
    out_data = ROOT / "week2/data"
    out_reports = ROOT / "week2/reports"
    out_data.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    data, scored = load_inputs(args.data, args.predictions, args.split)
    scored = add_peer_statistics(data, scored, args.min_peer_count)
    scored = build_screening(scored, args.peer_threshold)
    scored["absolute_residual"] = (scored["price"] - scored["predicted_price"]).abs()
    output_cols = ["id", "sku_id", "city", "brand", "model", "variant", "registration_year", "price", "predicted_price", "interval_lower_90", "interval_upper_90", "relative_residual", "peer_reference_level", "peer_comparability", "peer_reference_warning", "peer_count", "peer_median", "peer_p25", "peer_p75", "peer_relative_gap", "model_high_flag", "model_low_flag", "peer_high_flag", "peer_low_flag", "screening_high_flag", "screening_low_flag", "screening_score", "review_priority", "screening_label", "screening_reasons", "detail_url"]
    scored[output_cols].sort_values("screening_score", ascending=False).to_csv(out_data / "scored_listings.csv", index=False, encoding="utf-8-sig")
    review_counts = make_review_template(scored, out_data / "anomaly_review_template.csv", args.top_k, args.random_k, SEED)
    summary = {
        "purpose": "screening suspected anomalous listing prices for human review",
        "data_rows": int(len(data)), "scored_test_rows": int(len(scored)),
        "training_reference_rows": int(data.partition.eq("train").sum()), "min_peer_count": args.min_peer_count,
        "peer_relative_threshold": args.peer_threshold, "review_sample_counts": review_counts,
        "screening_counts": {str(k): int(v) for k, v in scored.screening_label.value_counts().to_dict().items()},
        "model_interval_counts": {"high": int(scored.model_high_flag.sum()), "low": int(scored.model_low_flag.sum())},
        "peer_flag_counts": {"high": int(scored.peer_high_flag.sum()), "low": int(scored.peer_low_flag.sum())},
        "peer_reference_levels": {str(k): int(v) for k, v in scored.peer_reference_level.value_counts().to_dict().items()},
        "score_quantiles": {str(q): float(scored.screening_score.quantile(q)) for q in [.5, .9, .95, .99, 1]},
        "limits": [
            "No abnormality ground truth is available. screening_label is a queue label, not a confirmed abnormality or fraud label.",
            "Peer prices are calculated from the training partition only; test prices are not used to create their peer reference.",
            "Current external listing prices may not be comparable to the source snapshot and should be recorded separately with query dates.",
            "The price model and interval were trained in week1; this script does not retrain or tune the model.",
        ],
    }
    (out_reports / "anomaly_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# 疑似异常挂牌价格初筛", "", "本报告生成的是人工复核队列，不是异常真值分类器。", "",
          "## 判定组成", "", "- 模型信号：测试集挂牌价是否越过独立校准的90%预测区间。",
          f"- 同类信号：优先使用训练集中的城市+车系+注册年，回退到城市+品牌+注册年、城市+品牌、品牌+车系+注册年、品牌+车系；至少{args.min_peer_count}辆训练车源才启用同类统计。对单独城市或全品牌均价不作车型参考，避免把豪华车与全市场混比。",
          f"- 同类偏离阈值：相对训练集同类中位价达到±{args.peer_threshold:.0%}时标记为同类高/低价信号。",
          "- 输出为疑似异常高价/疑似异常低价，必须人工查看车况、配置、价格类型和同期可比挂牌。", "",
          "## 复核样本", "", f"Top-K高价 {review_counts['top_high']} 条、Top-K低价 {review_counts['top_low']} 条、随机区间内样本 {review_counts['random_normal']} 条。" ,
          "复核模板中已预留human_label、human_confidence、reason_codes、外部平台URL、query_date和review_notes。标注人应先隐藏screening_label和screening_score，完成盲审后再合并模型结果。", "",
          "## 不应报告的指标", "", "没有真实异常标签，当前不能报告整体Precision、Recall、F1或宣称发现虚假车源。Top-K人工核查可以报告Precision@K，但必须给出样本来源和标注规则。", "",
          "## 运行", "", "```powershell", ".\\.venv\\Scripts\\python.exe -X utf8 week2\\src\\anomaly_detection.py", "```", ""]
    (out_reports / "anomaly_screening_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"rows": len(scored), "review": review_counts, "screening_counts": summary["screening_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
