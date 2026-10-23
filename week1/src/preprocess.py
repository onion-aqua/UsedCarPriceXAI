"""Reproducible week-one data audit and preprocessing (no target-based outlier removal)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAPITALS = set("北京 天津 上海 重庆 石家庄 太原 呼和浩特 沈阳 长春 哈尔滨 南京 杭州 合肥 福州 南昌 济南 郑州 武汉 长沙 广州 南宁 海口 成都 贵阳 昆明 拉萨 西安 兰州 西宁 银川 乌鲁木齐".split())
MISSING = {"", "-", "--", "暂无", "暂无报价", "暂无用车成本相关信息", "暂无参数相关信息", "暂无档案相关信息", "未知", "nan", "None", "null", "未填写"}
FIELDS = [
    ("id", "id", "原记录标识，非车辆去重依据"),
    ("详情链接", "sku_id", "URL sku_id 提取，车辆去重标识"),
    ("价格", "price", "挂牌价，万元；仅缺失/非正值排除，极端高价保留"),
    ("价格", "log_price", "log1p(price)，反变换使用expm1"),
    ("品牌名称", "brand", "品牌类别，保留语义字符串"),
    ("汽车名称", "model", "车系类别，保留语义字符串"),
    ("汽车发动机名称", "variant", "实际内容是配置版本，非发动机排量"),
    ("标题", "title", "原始车源标题"),
    ("车源城市名称", "city", "车源城市，限定31个省会/自治区首府/直辖市"),
    ("汽车第一次注时间", "registration_date", "首次注册日期，1970占位值置缺失"),
    ("汽车第一次注时间", "registration_year", "首次注册年份"),
    ("汽车生产年份", "model_year", "原字段可能为款型年，不能直接认作制造年份"),
    ("汽车第一次注时间", "vehicle_age_current", "任务要求：参考年2026减注册年；仅展示，不输入历史挂牌价模型"),
    ("汽车年龄", "age_at_listing_months", "原记录年龄解析为月，用于挂牌时点价格模型"),
    ("里程数", "mileage_km", "统一公里，保留零里程并标记"),
    ("里程数+汽车年龄", "annual_mileage_km", "mileage_km/(age_at_listing_months/12)，不足12月不外推年均"),
    ("用车成本→发动机", "engine_liters", "错位字段解析排量L；纯电等无排量时保持缺失"),
    ("用车成本→发动机", "engine_text", "原发动机描述"),
    ("用车成本→变速箱", "gearbox", "原变速箱类别"),
    ("用车成本→轴距", "wheelbase_mm", "轴距mm，明显非法值置缺失并标记"),
    ("相关参数→排放标准", "emission_standard", "档案列表中的排放标准"),
    ("相关参数→能源类型", "fuelType", "仅提取明确提供的能源类型，其余缺失，不按品牌猜测"),
    ("用车成本→最大功率", "power", "仅提取明确提供的最大功率kW，其余缺失；与排量不同"),
    ("Column1", "dealer", "实际公司名称"),
    ("详情链接", "detail_url", "原始详情来源"),
    ("档案详情", "source_flag_unknown", "实际为0/1，含义未证实，不强行命名事故/固定价"),
]


def missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().isin(MISSING)


def missing_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in frame.columns:
        count = int(missing_mask(frame[col]).sum())
        rows.append({"column": col, "rows": len(frame), "missing_count": count,
                     "missing_rate": count / len(frame) if len(frame) else 0,
                     "unique_nonmissing": int(frame.loc[~missing_mask(frame[col]), col].nunique())})
    return pd.DataFrame(rows)


def text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().mask(missing_mask(series))


def list_entries(value: str) -> tuple[dict, bool]:
    if str(value).strip() in MISSING:
        return {}, False
    try:
        entries = ast.literal_eval(value)
        if not isinstance(entries, list) or any(not isinstance(x, dict) for x in entries):
            return {}, True
        return {str(x.get("name", "")): str(x.get("value", "")).strip() for x in entries}, False
    except (SyntaxError, ValueError, TypeError):
        return {}, True


def unit_number(series: pd.Series, regex: str) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(regex, expand=False), errors="coerce")


def age_months(value: str) -> float:
    value = str(value).strip()
    if not re.fullmatch(r"(?:\d+年)?(?:\d+个?月)?", value) or not value:
        return np.nan
    year = re.search(r"(\d+)年", value)
    month = re.search(r"(\d+)个?月", value)
    return float((int(year.group(1)) * 12 if year else 0) + (int(month.group(1)) if month else 0))


def describe_price(series: pd.Series) -> dict:
    result = series.describe(percentiles=[.25, .5, .75, .95, .99]).to_dict()
    result["skewness"] = float(series.skew())
    return {str(k): float(v) for k, v in result.items()}


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def make_figures(clean: pd.DataFrame, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    values = clean["price"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].hist(values, bins=80, color="#276A9A")
    axes[0].set(title="Original price: complete range", xlabel="Listing price (10,000 CNY)", ylabel="Vehicles")
    bound = values.quantile(.99)
    axes[1].hist(values[values <= bound], bins=65, color="#276A9A")
    axes[1].set(title=f"Zoom to P99 ({bound:.2f}); plot only", xlabel="Listing price (10,000 CNY)", ylabel="Vehicles")
    axes[2].hist(clean["log_price"], bins=65, color="#D18A32")
    axes[2].set(title="log1p(price): complete range", xlabel="log1p(price in 10,000 CNY)", ylabel="Vehicles")
    fig.suptitle(f"Capital-city listings after SKU deduplication (n={len(clean):,})", fontsize=14)
    fig.text(.5, .015, "All positive high-price listings retained. P99 zoom does not remove records from the model data.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, .95])
    fig.savefig(folder / "price_distribution.png", bbox_inches="tight")
    plt.close(fig)


def run(source: Path, output: Path, reference_year: int) -> dict:
    data_dir, report_dir, figure_dir = [output / x for x in ("data", "reports", "figures")]
    for folder in (data_dir, report_dir, figure_dir):
        folder.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source, encoding="gb18030", dtype=str, keep_default_na=False)
    save_csv(missing_report(raw), report_dir / "preprocessing_missing_raw.csv")
    expected = {"id", "详情链接", "价格", "汽车年龄", "相关参数", "用车成本", "公司名称", "Column1"}
    if not expected.issubset(raw.columns):
        raise ValueError(f"Unexpected source schema; missing: {expected - set(raw.columns)}")
    archive_parsed = raw["相关参数"].map(list_entries)
    parameter_parsed = raw["用车成本"].map(list_entries)
    cost_parsed = raw["公司名称"].map(list_entries)
    archive = pd.DataFrame([x[0] for x in archive_parsed], index=raw.index)
    params = pd.DataFrame([x[0] for x in parameter_parsed], index=raw.index)
    get = lambda frame, col: text_series(frame.get(col, pd.Series("", index=raw.index)))
    result = pd.DataFrame(index=raw.index)
    result["id"] = text_series(raw["id"])
    result["source_row"] = np.arange(2, len(raw) + 2)
    result["sku_id"] = raw["详情链接"].str.extract(r"[?&]sku_id=(\d+)", expand=False)
    result["dedup_key"] = result["sku_id"].fillna("record:" + result["id"])
    if result["dedup_key"].isna().any():
        raise ValueError("A row has neither SKU nor record ID; resolve before deduplication")
    for target, source_col in [("brand", "品牌名称"), ("model", "汽车名称"), ("variant", "汽车发动机名称"),
                               ("title", "标题"), ("city", "车源城市名称"), ("dealer", "Column1"),
                               ("detail_url", "详情链接"), ("source_flag_unknown", "档案详情")]:
        result[target] = text_series(raw[source_col])
    result["price"] = unit_number(raw["价格"], r"^\s*([0-9]+(?:\.[0-9]+)?)万\s*$")
    result["flag_price_invalid"] = result["price"].isna() | (result["price"] <= 0)
    result["log_price"] = np.log1p(result["price"].where(~result["flag_price_invalid"]))
    registration = pd.to_datetime(raw["汽车第一次注时间"], format="mixed", errors="coerce")
    result["flag_registration_invalid"] = registration.isna() | (registration.dt.year < 1980) | (registration.dt.year > reference_year)
    registration = registration.mask(result["flag_registration_invalid"])
    result["registration_date"] = registration.dt.strftime("%Y-%m-%d")
    result["registration_year"] = registration.dt.year
    result["vehicle_age_current"] = reference_year - result["registration_year"]
    result["model_year"] = pd.to_numeric(raw["汽车生产年份"], errors="coerce")
    result["age_at_listing_months"] = raw["汽车年龄"].map(age_months)
    result["flag_age_invalid"] = result["age_at_listing_months"].isna() | (result["age_at_listing_months"] < 0) | (result["age_at_listing_months"] > 600)
    result["age_at_listing_months"] = result["age_at_listing_months"].mask(result["flag_age_invalid"])
    result["mileage_km"] = unit_number(raw["里程数"], r"^\s*([0-9]+(?:\.[0-9]+)?)万公里\s*$") * 10000
    plain_mileage = unit_number(raw["里程数"], r"^\s*([0-9]+(?:\.[0-9]+)?)公里\s*$")
    result["mileage_km"] = result["mileage_km"].fillna(plain_mileage)
    result["flag_mileage_missing"] = result["mileage_km"].isna()
    result["flag_mileage_zero"] = result["mileage_km"].eq(0)
    result["flag_mileage_extreme"] = result["mileage_km"].gt(1000000)
    result["annual_mileage_km"] = (result["mileage_km"] / (result["age_at_listing_months"] / 12)).where(result["age_at_listing_months"] >= 12)
    result["engine_text"] = get(params, "发动机")
    result["engine_liters"] = unit_number(result["engine_text"], r"([0-9]+(?:\.[0-9]+)?)\s*[LT]")
    result["flag_engine_liters_invalid"] = result["engine_liters"].notna() & ((result["engine_liters"] <= 0) | (result["engine_liters"] > 12))
    result["engine_liters"] = result["engine_liters"].mask(result["flag_engine_liters_invalid"])
    result["gearbox"] = get(params, "变速箱")
    result["emission_standard"] = get(archive, "排放标准")
    result["wheelbase_mm"] = unit_number(get(params, "轴距"), r"([0-9]+(?:\.[0-9]+)?)\s*mm")
    result["flag_wheelbase_invalid"] = result["wheelbase_mm"].notna() & ~result["wheelbase_mm"].between(1500, 6000)
    result["wheelbase_mm"] = result["wheelbase_mm"].mask(result["flag_wheelbase_invalid"])
    result["fuelType"] = get(archive, "能源类型")
    result["power"] = unit_number(get(params, "最大功率"), r"^\s*([0-9]+(?:\.[0-9]+)?)\s*kW\s*$")
    result["flag_power_invalid"] = result["power"].notna() & (result["power"] <= 0)
    result["power"] = result["power"].mask(result["flag_power_invalid"])
    result["flag_archive_parse_error"] = [x[1] for x in archive_parsed]
    result["flag_parameter_parse_error"] = [x[1] for x in parameter_parsed]
    result["flag_cost_parse_error"] = [x[1] for x in cost_parsed]
    result["flag_city_archive_mismatch"] = get(archive, "车源地").notna() & get(archive, "车源地").ne(result["city"])
    # Indirect dating evidence only. A vehicle's age is not an authenticated crawl timestamp.
    inferred_month = registration.dt.year * 12 + registration.dt.month - 1 + result["age_at_listing_months"]
    result["inferred_snapshot_month"] = inferred_month.map(lambda x: f"{int(x)//12:04d}-{int(x)%12+1:02d}" if pd.notna(x) else np.nan)
    result["flag_noncapital"] = ~result["city"].isin(CAPITALS)
    dup_all = result["dedup_key"].duplicated(keep=False)
    result["flag_duplicate_sku"] = dup_all
    conflict_cols = ["price", "brand", "model", "variant", "city", "registration_date", "mileage_km"]
    conflicts = result.loc[dup_all].groupby("dedup_key")[conflict_cols].nunique(dropna=False).gt(1).any(axis=1)
    conflict_keys = set(conflicts[conflicts].index)
    result["flag_duplicate_conflict"] = result["dedup_key"].isin(conflict_keys)
    result["flag_duplicate_discarded"] = result["dedup_key"].duplicated(keep="first")
    # Never pick the cheap/expensive row, mean price, or latest listing without timestamp evidence.
    exclusion = np.select([result["flag_duplicate_discarded"], result["flag_noncapital"], result["flag_price_invalid"]],
                          ["duplicate_sku_later_row", "outside_31_capitals", "missing_or_nonpositive_price"], default="included")
    result["selection_status"] = exclusion
    clean = result.loc[result["selection_status"].eq("included")].copy()
    p99 = float(clean["price"].quantile(.99))
    q1, q3 = clean["price"].quantile([.25, .75])
    threshold = float(q3 + 3 * (q3 - q1))
    result["flag_price_high_p99"] = result["price"].gt(p99)
    result["flag_price_high_3iqr"] = result["price"].gt(threshold)
    clean = result.loc[result["selection_status"].eq("included")].copy()
    raw_archive = raw.loc[dup_all].copy()
    raw_archive.insert(0, "source_row", result.loc[dup_all, "source_row"])
    raw_archive.insert(1, "sku_id", result.loc[dup_all, "sku_id"])
    raw_archive.insert(2, "duplicate_conflict", result.loc[dup_all, "flag_duplicate_conflict"])
    raw_archive.insert(3, "dedup_retained", ~result.loc[dup_all, "flag_duplicate_discarded"])
    save_csv(raw_archive, data_dir / "duplicate_sku_archive.csv")
    save_csv(result, data_dir / "preprocessing_all_rows_audit.csv")
    save_csv(clean, data_dir / "processed_capital_cars.csv")
    save_csv(missing_report(clean), report_dir / "preprocessing_missing_clean.csv")
    save_csv(result.loc[result["selection_status"].ne("included")], data_dir / "preprocessing_excluded_rows.csv")
    # Metadata vocabulary is for reading categories only, never a numeric order model input.
    category_rows = []
    for col in ["brand", "model", "variant", "city", "gearbox", "emission_standard", "fuelType"]:
        for index, category in enumerate(sorted(clean[col].dropna().unique())):
            category_rows.append({"feature": col, "display_code": index, "category": category, "model_input": "use original string and fit encoder on training split only"})
    save_csv(pd.DataFrame(category_rows), data_dir / "category_vocabulary.csv")
    mapping = pd.DataFrame(FIELDS, columns=["source_field", "output_field", "meaning_and_rule"])
    save_csv(mapping, report_dir / "data_dictionary_mapping.csv")
    flags = [c for c in clean if c.startswith("flag_")]
    core_counts = {"raw_rows": len(raw), "raw_columns": len(raw.columns), "raw_unique_ids": int(raw["id"].nunique()),
                   "raw_unique_sku": int(result["sku_id"].nunique()), "missing_sku": int(result["sku_id"].isna().sum()),
                   "duplicate_rows_including_first": int(dup_all.sum()), "duplicate_groups": int(result.loc[dup_all, "dedup_key"].nunique()),
                   "duplicate_conflict_groups": len(conflict_keys), "raw_capital_rows": int((~result["flag_noncapital"]).sum()),
                   "clean_rows": len(clean), "clean_unique_sku": int(clean["sku_id"].nunique()),
                   "clean_city_count": int(clean["city"].nunique()), "clean_brand_count": int(clean["brand"].nunique()),
                   "clean_model_count": int(clean["model"].nunique())}
    audit = {"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "source_encoding": "gb18030", "output_encoding": "utf-8-sig", "reference_year": reference_year,
             "counts": core_counts, "exclusive_selection_counts": result["selection_status"].value_counts().to_dict(),
             "price_wan": describe_price(clean["price"]), "log1p_price": describe_price(clean["log_price"]),
             "price_p99_flag_threshold_wan": p99, "price_3iqr_flag_threshold_wan": threshold,
             "clean_flag_counts": clean[flags].sum().astype(int).to_dict(),
             "clean_missing_count": {col: int(missing_mask(clean[col]).sum()) for col in clean},
             "inferred_snapshot_month_counts": result["inferred_snapshot_month"].value_counts().head(15).to_dict(),
             "registration_date_range_valid": [str(registration.min()), str(registration.max())],
             "archive_keys_counts": {str(k): int(v) for k, v in archive.notna().sum().items()},
             "parameter_keys_counts": {str(k): int(v) for k, v in params.notna().sum().items()},
             "source_flag_unknown_counts": raw["档案详情"].value_counts().to_dict(),
             "city_counts_clean": clean["city"].value_counts().to_dict(),
             "source_recency_status": "UNVERIFIED: age plus first registration provides indirect snapshot-month evidence only",
             "dedup_policy": "All source rows: retain first row per sku_id (fallback record:id), independent of price; then filter cities and valid targets. Archive every duplicate row and conflicts.",
             "encoding_policy": "Retain semantic category strings; metadata vocabulary is not ordinal model input; training pipeline must fit encoder on training data only.",
             "power_and_fuel_policy": "Extract only explicit nested energy-type and maximum-power values; kW is not engine displacement. Preserve missing entries and do not infer fuel from brand."}
    (report_dir / "preprocessing_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(clean, figure_dir)
    write_report(audit, report_dir)
    assert clean["dedup_key"].is_unique
    assert clean["price"].gt(0).all()
    assert clean["city"].isin(CAPITALS).all()
    assert np.allclose(np.expm1(clean["log_price"]), clean["price"])
    assert len(raw) == sum(audit["exclusive_selection_counts"].values())
    print(json.dumps({"output": str(data_dir / "processed_capital_cars.csv"), **core_counts}, ensure_ascii=False, indent=2))
    return audit


def write_report(audit: dict, report_dir: Path) -> None:
    counts = audit["counts"]
    price, log = audit["price_wan"], audit["log1p_price"]
    month_counts = audit["inferred_snapshot_month_counts"]
    lines = ["# 第一周数据探索与预处理报告", "", "## 结论与任务口径", "",
             "使用已有懂车帝 CSV，无需重复下载。训练数据目标是挂牌价，单位万元，不是成交价。导师清单中的 regDate/model/brand/fuelType/gearbox/power 是通用或天池风格字段，不能假设本文件具有同名字段。", "",
             f"原始 {counts['raw_rows']:,} 行、{counts['raw_columns']} 列；按车辆 SKU 全局去重，再筛选31个目标城市及有效价格，得到 {counts['clean_rows']:,} 辆。原始 id 唯一，但不能据此断定车源无重复。", "",
             "## 时效性核验", "", "采集日期尚无原始网页、下载说明或采集日志的直接证明。注册时间表示车的年龄，不表示数据采集时间；文件修改时间也不能作为采集日期。", "",
             "首次注册年月加原始“汽车年龄”可得到以下间接快照月份线索：", ""]
    lines.extend(f"- {k}：{v:,} 条" for k, v in list(month_counts.items())[:6])
    lines += ["", "这些线索支持进一步追查可能的历史采集时点，不能据此写“已确认近五年”。须补充原始数据发布页/版本日期或采集日志。旧车注册早于五年前不代表数据过期。当前参考日期为2026年10月23日，近五年应核实采集/发布时点是否在2021年10月23日之后。", "",
              "## 表头错位证据", "", "- 原“档案详情”全为0/1，语义未知，另存source_flag_unknown，不随意解释。",
              "- 原“相关参数”中列表包含上牌时间、表显里程、排放标准、车源地，因此按档案信息解析。",
              "- 原“用车成本”中列表包含发动机、变速箱、轴距等，因此按车辆参数解析。",
              "- 原“公司名称”是油耗/保险/保养成本列表或暂无信息；原Column1是商家名称。",
              "- 原“汽车发动机名称”实际是配置版本；原“汽车生产年份”与标题款型年对应，保守命名model_year。",
              "字段映射见data_dictionary_mapping.csv；解析键的全量计数见preprocessing_audit.json。", "",
              "## 缺失与异常处理", "", "原始及清洗后每一列的缺失数量、比例、非缺失基数分别保存为preprocessing_missing_raw.csv和preprocessing_missing_clean.csv。空字符串、暂无、暂无报价、未知等统一视为缺失；数值0不是通用缺失值。", "",
              "- 固定价格无有效报价，不输入模型。缺失/非正价格无法监督训练，单独归档。",
              "- 1970占位注册日期和非法注册日期置缺失，并保留标志。",
              "- 零里程保留并标记；超过100万公里保留复核。排量≤0或>12L、轴距不在1500—6000mm的字段置缺失并标记，这些只是宽松数据校验界限。",
              "- 原文件没有独立power/fuelType列，但全量嵌套键审计发现部分记录有最大功率和能源类型，因此明确提取power（kW）及fuelType；其余保持缺失，不将排量当功率、不按品牌推断能源类型。非正功率置缺失并标记。",
              f"- 清洗后fuelType缺失 {audit['clean_missing_count']['fuelType']:,} 条（{audit['clean_missing_count']['fuelType']/counts['clean_rows']:.2%}），power缺失 {audit['clean_missing_count']['power']:,} 条（{audit['clean_missing_count']['power']/counts['clean_rows']:.2%}）。能源类型只对部分新能源车提供，缺失不能直接标为汽油。",
              "- 超高价保留，使用P99和Q3+3IQR仅标记，不作为异常真实标签，也不因目标值过高删车。", "",
              "## 重复车源及冲突", "",
              f"SKU唯一数 {counts['raw_unique_sku']:,}；重复组 {counts['duplicate_groups']:,}；重复明细（含组内首条）{counts['duplicate_rows_including_first']:,} 行；关键字段存在差异的组 {counts['duplicate_conflict_groups']:,}。",
              "保留每个SKU在源文件最先出现的一行，完全不按价格选择；再筛城市及有效目标。没有时间戳，不能声称首条是最新或最正确。所有组内记录（含保留项）完整保存于duplicate_sku_archive.csv，冲突标志传递到建模数据。同一SKU只出现一次，后续划分训练/测试不会因相同SKU交叉而泄漏；无VIN仍可能存在不同SKU的同车发布。", "",
              "筛选顺序的互斥计数：", ""]
    lines.extend(f"- {k}：{v:,}" for k, v in audit["exclusive_selection_counts"].items())
    lines += ["", "## 价格分布与对数变换", "",
              f"清洗后价格中位数 {price['50%']:.3f} 万元，均值 {price['mean']:.3f} 万元，最大 {price['max']:.3f} 万元。原价格偏度 {price['skewness']:.3f}，log1p后偏度 {log['skewness']:.3f}。",
              "price_distribution.png同时展示全范围、P99以内放大视图和完整对数分布。放大视图只是绘图，不从建模数据删除尾部。保留price和log_price供模型对比；偏度下降不自动意味着预测更好。对数模型预测用expm1反变换，再在原始万元尺度计算MAE/RMSE/R²。", "",
              "## 车龄与类别编码", "",
              f"vehicle_age_current按任务公式 {audit['reference_year']}−注册年计算，用于当前车龄展示。历史挂牌价建模使用原记录汽车年龄解析得到的age_at_listing_months，避免把历史价格与2026年车龄混搭。annual_mileage_km仅对挂牌年龄≥12个月计算，年轻车辆保留缺失以免外推失真。",
              "类别在CSV中保留字符串。category_vocabulary.csv是可读编号字典，不应把编号当作连续/大小有序的输入。后续模型pipeline在训练集fit OneHotEncoder(handle_unknown='ignore')，验证/测试仅transform；缺失值插补也仅在训练集拟合。不计算全数据同类价格均值输入模型。", "",
              "## 复现与剩余事项", "", "运行：.venv\\Scripts\\python.exe week1\\src\\preprocess.py。原CSV不覆盖，输出采用UTF-8 BOM。audit.json记录源SHA256、阈值、计数和语义边界。",
              "已执行唯一SKU、目标合法、城市集合、对数反变换及行数守恒断言。仍需补充数据发布/采集时间直接证据；燃料类型和功率存在较多缺失，真实异常标签目前不可用。"]
    (report_dir / "preprocessing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dictionary = ["# 建模字段字典", "", "价格单位：万元；里程：公里；年龄：月。所有flag仅为数据质量/描述标志，不能当作真实异常标签。", "", "| 原字段 | 输出字段 | 规则 |", "|---|---|---|"]
    dictionary += [f"| {a} | {b} | {c} |" for a, b, c in FIELDS]
    dictionary += ["", "source_row是原CSV物理记录序号（表头计作第1行）；dedup_key是SKU优先的去重键；selection_status记录排除原因。inferred_snapshot_month仅为间接时点线索。", "",
                   "推荐第一周模型数值输入：age_at_listing_months、mileage_km、annual_mileage_km、engine_liters、wheelbase_mm、power（kW）；类别输入：brand、model、variant、city、gearbox、emission_standard、fuelType。功率和能源类型高缺失，插补/缺失指示须只在训练流程拟合。", "",
                   "禁止将price、log_price、price高价标志、source_row、id/SKU、URL、推断月份、当前车龄作为此基线模型输入。人工标签与未来目标相关统计必须独立设计。"]
    (report_dir / "data_dictionary.md").write_text("\n".join(dictionary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "懂车帝全国各个省会城市二手车相关数据.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "week1")
    parser.add_argument("--reference-year", type=int, default=2026)
    args = parser.parse_args()
    run(args.source, args.output, args.reference_year)
