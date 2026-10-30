# 第二周可视化原型

这是课题的 Streamlit 演示页面，使用 `week1` 的清洗数据、测试集预测、SHAP输出和 `week2/data/scored_listings.csv`。页面中的“疑似高价/疑似低价”是统计初筛标签，不是真实异常或成交价真值。

## 运行

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r week2\requirements-dashboard.txt
.\.venv\Scripts\python.exe -m streamlit run week2\app.py
```

如果没有本地 `.venv`，使用 Python 3.12 创建环境后安装依赖。页面需要先生成 `week1/data/processed_capital_cars.csv`、`week1/reports/test_predictions.csv` 和 `week2/data/scored_listings.csv`。没有第二周初筛文件时，会先使用第一周测试预测中的 `screening_label`，页面会提示当前没有同类参考统计和人工标签。

## 页面

- 数据概览：记录数、价格分布、清洗审计。
- 城市与品牌：城市/品牌车源结构、挂牌价和车龄关系。
- 异常初筛：按城市、品牌、价格、模型标签筛选并下载结果，也可下载130条人工复核模板。
- 单车诊断：查看车辆字段、预测区间、SHAP局部贡献。
- 外部核验：查看少量公开搜索线索和查询日期，明确历史快照时点限制。
- 方法说明：数据隔离、人工复核和研究边界。

外部平台的相近车源不能自动当作标签。若后续添加人工核验表，应把查询日期、平台、可比条件和人工判断理由作为独立字段保存。
