"""Streamlit prototype for the used-car listing-price screening study.

Run from the repository root with:
    streamlit run week2/app.py

The dashboard reads the week-one artifacts. It never treats a model screening
label as a confirmed abnormality or transaction truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - displayed by Streamlit
    st.error("缺少 plotly。请先执行 pip install -r week2/requirements-dashboard.txt")
    st.stop()


ROOT = Path(__file__).resolve().parents[1]
WEEK1 = ROOT / "week1"
WEEK2 = ROOT / "week2"
DATA = WEEK1 / "data" / "processed_capital_cars.csv"
PREDICTIONS = WEEK1 / "reports" / "test_predictions.csv"
SHAP_VALUES = WEEK1 / "reports" / "shap_values.csv"
SHAP_LOCAL = WEEK1 / "reports" / "shap_local.csv"
METADATA = WEEK1 / "reports" / "model_metadata.json"
AUDIT = WEEK1 / "reports" / "preprocessing_audit.json"
ANOMALY = WEEK2 / "data" / "scored_listings.csv"
EXTERNAL = WEEK2 / "reports" / "external_validation_candidates.csv"


st.set_page_config(page_title="二手车挂牌价格分析", page_icon="🚘", layout="wide")


@st.cache_data(show_spinner="正在加载数据…")
def load_data():
    if not DATA.exists():
        return None, None, None
    columns = [
        "id", "sku_id", "price", "brand", "model", "variant", "city",
        "registration_year", "vehicle_age_current", "age_at_listing_months",
        "mileage_km", "annual_mileage_km", "engine_liters", "gearbox",
        "emission_standard", "wheelbase_mm", "fuelType", "power",
    ]
    clean = pd.read_csv(DATA, usecols=columns, low_memory=False)
    predictions = pd.read_csv(PREDICTIONS, low_memory=False) if PREDICTIONS.exists() else pd.DataFrame()
    screened = pd.read_csv(ANOMALY, low_memory=False) if ANOMALY.exists() else predictions.copy()
    if not predictions.empty:
        merge_cols = ["id", "predicted_price", "interval_lower_90", "interval_upper_90", "relative_residual", "screening_label", "absolute_residual"]
        available = [c for c in merge_cols if c in predictions.columns]
        clean = clean.merge(predictions[available].drop_duplicates("id"), on="id", how="left", suffixes=("", "_prediction"))
    if ANOMALY.exists() and not screened.empty:
        anomaly_cols = [c for c in [
            "id", "peer_reference_level", "peer_count", "peer_median", "peer_p25", "peer_p75",
            "peer_relative_gap", "model_high_flag", "model_low_flag", "peer_high_flag", "peer_low_flag",
            "screening_high_flag", "screening_low_flag", "screening_score", "review_priority",
            "screening_label", "screening_reasons", "detail_url",
        ] if c in screened.columns]
        clean = clean.drop(columns=[c for c in anomaly_cols if c != "id" and c in clean.columns], errors="ignore")
        clean = clean.merge(screened[anomaly_cols].drop_duplicates("id"), on="id", how="left", suffixes=("", "_screened"))
    for col in ["price", "predicted_price", "interval_lower_90", "interval_upper_90", "relative_residual", "absolute_residual", "mileage_km", "age_at_listing_months", "power"]:
        if col in clean:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    return clean, screened, predictions


@st.cache_data(show_spinner=False)
def load_shap():
    values = pd.read_csv(SHAP_VALUES) if SHAP_VALUES.exists() else pd.DataFrame()
    local = pd.read_csv(SHAP_LOCAL) if SHAP_LOCAL.exists() else pd.DataFrame()
    return values, local


@st.cache_data(show_spinner=False)
def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


clean, screened, predictions = load_data()
shap_values, shap_local = load_shap()
metadata = load_json(METADATA)
audit = load_json(AUDIT)

st.title("二手车挂牌价格分析与疑似异常初筛")
st.caption("研究原型：懂车帝车源挂牌价快照 · 31个目标城市 · 结果需要人工复核")

if clean is None:
    st.error("没有找到清洗数据。请先按仓库README运行 week1/src/preprocess.py 和 week1/src/train_baseline.py。")
    st.stop()

with st.sidebar:
    st.header("筛选条件")
    city_options = sorted(clean["city"].dropna().unique().tolist())
    brand_options = sorted(clean["brand"].dropna().unique().tolist())
    selected_city = st.multiselect("车源城市", city_options, default=[])
    selected_brand = st.multiselect("品牌", brand_options, default=[])
    min_price, max_price = float(clean.price.min()), float(clean.price.max())
    default_upper = min(max_price, max(200.0, float(clean.price.quantile(.99))))
    price_range = st.slider("挂牌价（万元）", min_price, max_price, (min_price, default_upper))
    screening_options = sorted([str(x) for x in clean.get("screening_label", pd.Series(dtype=str)).dropna().unique()])
    selected_screening = st.multiselect("模型初筛标签", screening_options, default=[])
    st.divider()
    st.caption("当前实验快照：2026-10-23")
    st.caption("价格单位：万元；模型输出是挂牌价参考，不是成交价。")

filtered = clean.copy()
if selected_city:
    filtered = filtered[filtered.city.isin(selected_city)]
if selected_brand:
    filtered = filtered[filtered.brand.isin(selected_brand)]
filtered = filtered[filtered.price.between(price_range[0], price_range[1])]
if selected_screening and "screening_label" in filtered:
    filtered = filtered[filtered.screening_label.isin(selected_screening)]

tab_overview, tab_market, tab_anomaly, tab_detail, tab_external, tab_method = st.tabs(["数据概览", "城市与品牌", "异常初筛", "单车诊断", "外部核验", "方法说明"])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前记录数", f"{len(filtered):,}")
    c2.metric("挂牌价中位数", f"{filtered.price.median():.2f} 万元")
    c3.metric("品牌数", f"{filtered.brand.nunique():,}")
    c4.metric("城市数", f"{filtered.city.nunique():,}")
    st.subheader("挂牌价分布")
    fig = px.histogram(filtered, x="price", nbins=70, labels={"price": "挂牌价（万元）", "count": "车源数"}, color_discrete_sequence=["#2563a6"])
    fig.update_layout(height=420, bargap=0.03)
    st.plotly_chart(fig, use_container_width=True)
    st.info("右偏和高价长尾是本数据的特征。高价记录在预处理中保留，后续只作为统计初筛对象，不直接视为错误数据。")
    with st.expander("数据审计摘要"):
        counts = audit.get("counts", {})
        st.json({k: counts.get(k) for k in ["raw_rows", "raw_columns", "raw_unique_sku", "clean_rows", "clean_city_count", "duplicate_groups", "duplicate_conflict_groups"]})

with tab_market:
    left, right = st.columns(2)
    with left:
        city_summary = filtered.groupby("city", as_index=False).agg(车源数=("id", "size"), 中位挂牌价=("price", "median"), 平均挂牌价=("price", "mean")).sort_values("车源数", ascending=False).head(31)
        fig = px.bar(city_summary.sort_values("车源数"), x="车源数", y="city", orientation="h", color="中位挂牌价", color_continuous_scale="Blues", labels={"city": "城市"}, title="城市车源数与中位挂牌价")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        brand_summary = filtered.groupby("brand", as_index=False).agg(车源数=("id", "size"), 中位挂牌价=("price", "median")).sort_values("车源数", ascending=False).head(20)
        fig = px.bar(brand_summary.sort_values("车源数"), x="车源数", y="brand", orientation="h", color="中位挂牌价", color_continuous_scale="Oranges", title="车源最多的品牌（前20）")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("价格与车龄、里程")
    plot_data = filtered.sample(min(5000, len(filtered)), random_state=2026) if len(filtered) else filtered
    fig = px.scatter(plot_data, x="age_at_listing_months", y="price", color="city", hover_data=["brand", "model", "mileage_km"], opacity=.55, labels={"age_at_listing_months": "挂牌时车龄（月）", "price": "挂牌价（万元）"})
    fig.update_layout(height=460)
    st.plotly_chart(fig, use_container_width=True)

with tab_anomaly:
    st.subheader("模型筛出的疑似异常挂牌")
    if "screening_label" not in filtered:
        st.warning("当前没有异常初筛文件。请先运行 week2/src/anomaly_detection.py。")
    else:
        labels = filtered.screening_label.fillna("").astype(str)
        anomaly_counts = pd.Series({
            "疑似高价": int(labels.str.contains("高价", regex=False).sum()),
            "疑似低价": int(labels.str.contains("低价", regex=False).sum()),
            "区间内": int(labels.str.contains("区间内", regex=False).sum()),
        })
        a, b, c = st.columns(3)
        a.metric("疑似高价", f"{int(anomaly_counts.get('疑似高价', 0)):,}")
        b.metric("疑似低价", f"{int(anomaly_counts.get('疑似低价', 0)):,}")
        c.metric("区间内", f"{int(anomaly_counts.get('区间内', 0)):,}")
        display_cols = [c for c in ["id", "city", "brand", "model", "variant", "price", "predicted_price", "interval_lower_90", "interval_upper_90", "relative_residual", "peer_reference_level", "peer_count", "peer_median", "peer_relative_gap", "screening_score", "review_priority", "screening_label", "screening_reasons", "human_label", "review_notes"] if c in filtered.columns]
        table = filtered.sort_values("absolute_residual", ascending=False)[display_cols].head(500).copy()
        st.dataframe(table, use_container_width=True, height=470)
        st.download_button("下载当前初筛表", table.to_csv(index=False).encode("utf-8-sig"), "anomaly_screening_filtered.csv", "text/csv")
        review_template = WEEK2 / "data" / "anomaly_review_template.csv"
        if review_template.exists():
            st.download_button("下载人工复核模板", review_template.read_bytes(), "anomaly_review_template.csv", "text/csv")
        st.caption("screening_label 是模型区间与训练集同类价格共同产生的初筛结果；人工标签需要独立盲审后填写。")

with tab_detail:
    st.subheader("单车价格与SHAP诊断")
    ids = filtered.id.dropna().astype(str).tolist()
    if not ids:
        st.warning("当前筛选没有可诊断车源。")
    else:
        selected_id = st.selectbox("选择车源ID", ids[:5000])
        row = filtered[filtered.id.astype(str).eq(selected_id)].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("实际挂牌价", f"{row.price:.2f} 万元")
        c2.metric("模型预测价", f"{row.predicted_price:.2f} 万元" if pd.notna(row.get("predicted_price")) else "暂无")
        c3.metric("初筛标签", str(row.get("screening_label", "暂无")))
        c4.metric("相对残差", f"{row.relative_residual:.1%}" if pd.notna(row.get("relative_residual")) else "暂无")
        info_cols = ["city", "brand", "model", "variant", "registration_year", "vehicle_age_current", "age_at_listing_months", "mileage_km", "annual_mileage_km", "engine_liters", "gearbox", "emission_standard", "fuelType", "power"]
        st.dataframe(pd.DataFrame({"字段": info_cols, "值": [row.get(c, "") for c in info_cols]}), hide_index=True, use_container_width=True)
        local = shap_local.copy()
        if not shap_values.empty and selected_id in set(shap_values.id.astype(str)):
            vals = shap_values[shap_values.id.astype(str).eq(selected_id)].iloc[0]
            feature_labels = {"brand":"品牌", "model":"车系", "variant":"配置版本", "city":"车源城市", "gearbox":"变速箱", "emission_standard":"排放标准", "fuelType":"能源类型", "age_at_listing_months":"挂牌时车龄（月）", "mileage_km":"表显里程（公里）", "annual_mileage_km":"年均里程（公里/年）", "engine_liters":"发动机排量（升）", "wheelbase_mm":"轴距（毫米）", "power":"最大功率（kW）"}
            shap_cols = [c for c in feature_labels if c in vals.index]
            local = pd.DataFrame({"feature": shap_cols, "label": [feature_labels[c] for c in shap_cols], "shap_value": [float(vals[c]) for c in shap_cols], "value": [row.get(c, "") for c in shap_cols]})
        if local.empty:
            st.info("当前车源没有保存的局部SHAP值。本周模型默认只保存随机解释样本和一个极端残差案例。")
        else:
            local = local.sort_values("shap_value")
            fig = px.bar(local, x="shap_value", y="label", orientation="h", color="shap_value", color_continuous_scale=["#2563a6", "#f1f5f9", "#c44e52"], hover_data=["value"], labels={"shap_value": "SHAP贡献（log1p价格模型输出）", "label": "特征"})
            fig.update_layout(height=480, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("SHAP解释模型预测贡献，单位是log1p价格模型输出；它不能证明挂牌异常的真实原因或欺诈行为。")

with tab_external:
    st.subheader("跨平台挂牌价格辅助核验")
    if EXTERNAL.exists():
        external = pd.read_csv(EXTERNAL, low_memory=False)
        st.dataframe(external, use_container_width=True, height=390)
        st.warning("这些记录来自2026-10-30的公开搜索摘要，而源CSV可能是2024年快照。它们只能作为人工核验线索，不能作为历史异常真值、成交价或欺诈证据。正式标注应点击原始详情页，记录页面更新时间、配置、车况和查询日期。")
        st.download_button("下载外部核验记录", external.to_csv(index=False).encode("utf-8-sig"), "external_validation_candidates.csv", "text/csv")
    else:
        st.info("尚未生成外部核验记录。请先完成少量公开车源查询，并保留查询日期和页面链接。")

with tab_method:
    st.subheader("方法与边界")
    st.markdown("""
**当前课题输出：**使用历史挂牌价格建立参考模型，对预测区间外的车源进行“疑似高价/疑似低价”初筛，再交给人工复核。

**实验隔离：**训练集拟合类别词表和缺失处理，验证集选择模型，独立校准集构建90% split-conformal区间，测试集只用于评估。没有真实异常标签，因此页面不展示Precision、Recall或F1。

**人工复核建议：**比较同城同车型、相近年份与里程的同期挂牌车源，并记录来源、查询日期、可比样本数和判断理由。跨平台挂牌价是参考，不是真实成交价。

**解释限制：**SHAP说明哪些特征影响模型输出。车辆实际车况、事故记录、配置细节和采集时点等未观测变量，可能造成实际价格偏离模型。
""")
    st.json({"model": metadata.get("selected_tree_model", "暂无"), "rows": metadata.get("data_rows", len(clean)), "features": metadata.get("features", []), "test_coverage": metadata.get("conformal", {}).get("test_coverage", None)})
