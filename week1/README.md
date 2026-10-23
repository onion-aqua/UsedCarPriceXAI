# 第一周交付：文献、预处理、模型和解释

报告日期：2026年10月23日。

本周主线已实际执行：懂车帝原始数据 → 清洗与车源去重 → 类别编码 → 基线建模 → SHAP解释。先验证流程，不把本轮随机划分结果当作最终论文结论。

## 先看这四份成果

1. [18篇文献与引用台账](literature/literature_review_ledger.md)：7篇中文、11篇英文，全部期刊；含原文入口、方法摘要、可引用转述和后续精读记录位置。
2. [数据探索与预处理报告](reports/preprocessing_report.md)：字段映射、缺失率、异常处理、重复车源和时效性证据。
3. [基线与SHAP运行报告](reports/model_baseline_report.md)：真实运行指标、数据划分、解释尺度与局限。
4. [数据准备说明及真实样例](data/README.md)：全量清洗CSV共119,662条、31个城市，由脚本本地生成，不随Git仓库提供。

## 对照老师的任务清单

| 任务 | 本周结果 | 证据与文件 |
|---|---|---|
| 15—20篇近三年文献，含外文、期刊为主 | 18篇期刊，7中文＋11英文；逐篇记录引用用途 | literature/literature_review_ledger.md、references.bib |
| 核心期刊多一点 | 6篇《软件学报》取得原刊核心/CSCD收录介绍证据；英文期刊收录待核验 | 中文阅读笔记中的期刊证据链接；以学校采用目录为准 |
| 下载加载数据与近五年核验 | 已读取本目录现有完整CSV，无需重复下载；采集日期尚未取得直接证据 | 原年龄和注册年月间接指向2024年4—5月；不能把文件修改日期当采集日期 |
| 记录数据基本情况 | 原始143,793行、21列、197个城市、224品牌、2,078车系 | preprocessing_audit.json |
| 统计每列缺失 | 原始及处理后逐列统计数量、比例、基数 | preprocessing_missing_raw.csv、preprocessing_missing_clean.csv |
| 绘制price直方图和考察对数变换 | 全范围、P99内放大、log1p三图；保留两种目标对比 | figures/price_distribution.png |
| regDate提取当前车龄 | 映射首次注册时间，保存2026−注册年；另存挂牌时车龄供历史价格建模 | vehicle_age_current、age_at_listing_months |
| model、brand、fuelType、gearbox编码 | 保留中文类别；Ridge用训练集One-Hot，LightGBM显式原生类别；未知类别独立处理 | src/train_baseline.py、模型元数据训练词表 |
| power为0、price过高等异常 | 提取kW功率并检测非正值；原始功率可提取值中未发现非正值。高价仅标记，保留潜在异常研究对象 | 清洗CSV的flag列和审计报告 |
| 保存预处理CSV | 已导出UTF-8 BOM格式 | data/processed_capital_cars.csv |
| 跑通数据→模型→解释 | Ridge、LightGBM原价/对数目标，已输出全局/局部SHAP与初步区间 | reports/model_baseline_report.md、figures/shap_*.png |

## 数据处理的重要决定

- 原记录ID无重复，不代表车辆不重复。详情链接提取的SKU有重复，保留每个SKU在原文件最先出现的一行，不按价格选择；全部重复原档另存。
- 原始143,793行，133,762个独立SKU；去重后再限定31个城市，最终119,662行。原来的126,227条省会记录是去重前数量，两个数字不是同一口径。
- 最大挂牌价2,288.8万元保留。不能仅因价格极端就删去，否则会提前删除研究对象。
- 按清洗数据计算，price偏度约11.753，log1p后约0.541。对数处理减少偏斜，但模型是否更好必须看还原到万元后的验证误差。
- fuelType和power并非原始独立列，而是部分嵌套记录中的能源类型、最大功率。能源类型缺失91.67%，最大功率缺失70.76%；缺失能源类型不等于燃油，排量不等于功率。
- 缺失注册时间873条、挂牌时年龄1021条。清洗CSV保留缺失，由训练流程处理；不先用全数据均值填补再划分。
- 没有加入任何由价格计算的均价差、原产地价格、价格分位标志作为模型特征，避免目标泄漏。

## 环境与复现

实验采用独立 `.venv`，Python 3.12。克隆仓库后须自行创建环境。依赖包括Pandas、NumPy、scikit-learn、LightGBM、SHAP、Matplotlib，准确版本保存于requirements-lock.txt。

准备好原始CSV后，在仓库根目录的PowerShell中运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 week1\src\preprocess.py
.\.venv\Scripts\python.exe -X utf8 week1\src\train_baseline.py
.\.venv\Scripts\python.exe -X utf8 week1\src\verify_outputs.py
```

重新创建环境时，使用已安装的Python 3.12执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r week1\requirements-lock.txt
```

默认使用全部119,662条清洗数据；固定随机种子2026。划分为训练60%、验证15%、校准10%、测试15%。验证集选模型，校准集定区间，测试集仅评估。同SKU不跨集合。

保存模型为 models/price_baseline.pkl。仅加载自己生成或可信来源的pickle模型。预处理与训练脚本每次运行会更新week1下的对应生成结果，原始CSV保持不变。

## 已做的验证

- 原始到筛选结果的行数守恒、31城市范围、SKU唯一性。
- price与log_price反变换、当前车龄公式、价格单位核对。
- 训练/验证/校准/测试划分的SKU互斥，编码及插补仅训练集拟合。
- 从保存的预测CSV独立重算MAE、RMSE、R²及区间覆盖率。
- SHAP贡献加基准值等于模型输出，明确对数输出尺度。
- 价格直方图、SHAP全局和局部图的视觉检查。
- 文献18条DOI唯一、年份与题录来源检查。摘要阅读不冒充全文精读。

验证结果见 reports/verification.json。

## 提交前仍需补齐

1. 数据原始下载页或采集日志。当前仅能推测2024年4—5月快照，不能正式完成“近五年真实采集时间”的证明。旧车注册早于五年前不等于数据过期。
2. 与导师确认“近三年”按正式卷期还是首次在线发表计算。台账目前采用2024—2026卷期；4篇在线日期为2023，已逐条标记。英文期刊收录与学校核心目录版本需图书馆核验。
3. 后续精读重点文献并补页码。本轮18篇均有摘要阅读依据，1篇另做全文定向阅读，不等于全部精读。
4. 第二周再做正式异常定义、盲审协议、分组交叉验证和分城市误差。预测区间越界不是异常真值；目前没有人工金标准，所以未报告Precision、Recall、F1。
5. 若核验历史挂牌，使用同期可比车源；不可直接拿2026年现价判2024年报价是否异常。人工标签不能以模型判定为必要条件。

## 建议本周学习顺序

先读预处理报告并对照原字段，再运行preprocess.py；接着看三个目标/模型的MAE与RMSE差别，最后查看SHAP图。文献先读CN01、EN01、EN02、EN03、EN08、EN10六篇，与数据、解释和异常评估任务逐一对应。先理解并复现现有流程，再增加算法。
