# 第一周英文文献阅读与引用笔记

核验日期：2026-10-23。11 篇均为期刊论文，正式卷期在 2024—2026 年；Huang 一篇存在 2023 年先行在线与 2024 年正式卷期差异，已单独标注。主清单是 `english_candidates.json`，本文件用于阅读与写作，题录须引用 DOI 所指原文。

这些是基于摘要及一篇全文定向阅读的首轮阅读卡，不等于已经精读 11 篇全文。没有逐刊核验学校认定的核心目录、SCIE/SSCI 收录年份或 JCR 分区，不能据此宣称“11 篇核心”。DOI 中的年份不自动等于发表年份。以正式卷期计年；如导师要求首次在线发表日期，则 EN01 转补充。

## 检索与验证方法

- 检索主题：used car price prediction、vehicle resale value、SHAP/LIME、tabular explainability evaluation、data leakage、explainable anomaly detection、conformal prediction、imbalanced regression。
- 用 Crossref 核对标题、作者、年份、刊名、卷期页码与 DOI；多数另用 OpenAlex 核对并读取摘要。EN05 的正式候选依据 Crossref，尚未进行独立第二数据库核验。
- 同一论文的期刊版、arXiv/SSRN 版只计一篇；预印本仅用于辅助阅读，不另外增加数量。
- 旧项目 `springbootvue/docs/literature_links.md` 仅用于发现候选，所有采用的书目信息本次重新联网核验。未修改旧项目。
- 原始检索证据见 `english_metadata_verified.json`；它保留检索过程候选，其中旧 EN05（Neurocomputing）因无摘要未入选最终主清单。最终 EN05 改为 WSMOTER，证据 URL 在最终 JSON 内。

## EN01 二手车价格预测与不确定性

HUANG J, SAW S N, FENG W, et al. A Latent Factor-Based Bayesian Neural Networks Model in Cloud Platform for Used Car Price Prediction[J]. IEEE Transactions on Engineering Management, 2024, 71: 12487-12497. [DOI](https://doi.org/10.1109/TEM.2023.3270301).

- 中文题意：云平台上基于潜因子与贝叶斯神经网络的二手车价格预测模型。
- 阅读深度：Crossref 题录、OpenAlex 重建摘要；未读全文。
- 方法与发现：对结构化二手车特征提取潜因子，贝叶斯网络以概率分布表示权重，在两个真实数据集检验预测效果；关注数据噪声、过拟合与不确定性。
- 可引用转述：二手车价格建模除了点预测误差，还需要考虑数据噪声与预测不确定性；相关研究尝试把潜因子表示与贝叶斯建模结合。
- 放入章节：国外研究现状、预测区间研究动机。
- 对项目启示：先建立简单树模型，不必为了引用本文而复现贝叶斯网络；残差异常判定应考虑预测的不确定性。
- 引用限制：不能将文中的 MAE 数字与本项目直接横比，摘要不足以确认所有量纲与实验协议。Crossref 正式卷期 2024，OpenAlex 首次发表 2023-05-31。
- 下一次精读：确认两数据集来源、采集时间、价格单位、训练测试划分和不确定性如何输出。

## EN02 挂牌数据与车辆保值率

ROBERSON L, PANTHA S, HELVESTON J P. Battery-powered bargains? Assessing electric vehicle resale value in the United States[J]. Environmental Research Letters, 2024, 19(5): 054053. [DOI](https://doi.org/10.1088/1748-9326/ad3fce).

- 中文题意：评估美国电动车的二手保值表现。
- 阅读深度：Crossref 题录与摘要；未读全文。
- 数据：2016—2022 年约 900 万条美国在线二手车挂牌记录。此处是挂牌，不是实际成交。
- 方法与发现：在品牌—车型层面比较纯电、插混、混动与燃油车保值率，分析车型年份、续航、补贴和疫情时期价格变化。摘要指出不同动力车折旧模式存在差异且随车型代际变化。
- 可引用转述：大规模挂牌数据可用于分析二手车保值表现，但价格关系受车辆技术属性与市场时间背景共同影响。
- 放入章节：研究背景、特征工程、数据时效性与外部效度。
- 对项目启示：保留动力类型；明确采集时点。用当前平台价格核验多年前挂牌价时，不能忽视市场变化。
- 引用限制：美国市场具体百分比不能直接搬到中国；2024 年论文使用更早数据，并不证明本项目数据满足近五年要求。
- 下一次精读：查看回归方程、时间控制、去重与挂牌样本选择。

## EN03 SHAP 与 LIME 的适用边界

SALIH A M, RAISI-ESTABRAGH Z, GALAZZO I B, et al. A Perspective on Explainable Artificial Intelligence Methods: SHAP and LIME[J]. Advanced Intelligent Systems, 2025, 7(1): 2400304. [DOI](https://doi.org/10.1002/aisy.202400304).

- 中文题意：SHAP 和 LIME 可解释方法的观点与使用注意事项。
- 阅读深度：Crossref 题录与摘要；未读全文。在线 2024，正式卷期 2025。
- 方法与发现：通过生物医学分类案例讨论模型依赖性与共线性，指出两种解释均受到所用模型与相关特征的影响。
- 可引用转述：SHAP 的解释结果需要结合模型结构与特征相关性解读，不能把单张解释图视为独立于模型的客观规律。
- 放入章节：解释方法、局部案例分析、研究局限。
- 对项目启示：车龄与注册年份、里程与年均里程可能高度相关；单个特征贡献在相关变量之间如何分配需要谨慎解释。
- 引用限制：模型归因不是因果证明；解释“预测价格为何高/低”不等于证明“实际挂牌为何异常”。
- 下一次精读：关注作者怎样设置共线性实验、怎样定义解释稳定性。

## EN04 表格解释方法如何评价

VELMURUGAN M, OUYANG C, XU Y, et al. Developing guidelines for functionally-grounded evaluation of explainable artificial intelligence using tabular data[J]. Engineering Applications of Artificial Intelligence, 2025, 141: 109772. [DOI](https://doi.org/10.1016/j.engappai.2024.109772). [公开预印本](https://arxiv.org/abs/2410.12803).

- 中文题意：表格数据可解释人工智能的功能性评价指南。
- 阅读深度：题录与 OpenAlex 摘要；未读全文。
- 方法与发现：梳理局部事后解释评价文献，归纳 20 项评价准则及其方法，并讨论何时和怎样评价。
- 可引用转述：解释工具在图像或文本上的评价方式未必直接适合表格数据，评价准则和操作方式也可能影响结果。
- 放入章节：解释评价方案、研究方法合理性。
- 对项目启示：在展示 SHAP 图之外，后续增加稳定性检查与人工可读性核验；本周先保证特征名、输出尺度及单样本加和关系正确。
- 引用限制：尚未阅读全文，不能列出未经核对的“20 项准则”或声称已经全部实施。
- 下一次精读：阅读评价准则表，选择 1—2 项适合毕业设计工作量的标准。

## EN05 长尾价格与不平衡回归

CAMACHO L, BACAO F. WSMOTER: a novel approach for imbalanced regression[J]. Applied Intelligence, 2024, 54(19): 8789-8799. [DOI](https://doi.org/10.1007/s10489-024-05608-6). [已读摘要证据](https://api.crossref.org/works/10.1007/s10489-024-05608-6).

- 中文题意：处理不平衡回归问题的 WSMOTER 方法。
- 阅读深度：Crossref 题录与摘要；未读全文，出版社页面没有返回可读摘要。
- 方法与发现：提出 Weighting SMOTE for Regression；讨论连续目标中稀有事件预测的重要性，并报告实验优于所比较方案，摘要未给出可直接引用的细节。
- 可引用转述：不平衡学习不仅是类别比例问题，也会出现在连续目标回归中；稀有目标区间可能恰恰是业务重点。
- 放入章节：价格长尾特征、分价格段模型评估、研究现状。
- 对项目启示：单独报告高价车与主流价格段 MAE；总体指标好不意味着少数高价车辆预测可靠。
- 引用限制：本周不引入过采样；合成数据会改变训练分布，残差校准必须使用独立真实数据。不能引用摘要没有给出的提升幅度。
- 下一次精读：查看稀有区间定义、重采样位置和评价指标。

## EN06 表格异常检测与扰动解释

MOK C, KIM S B. Corruption-based anomaly detection and interpretation in tabular data[J]. Pattern Recognition, 2025, 159: 111149. [DOI](https://doi.org/10.1016/j.patcog.2024.111149). [SSRN 同题预印本](https://doi.org/10.2139/ssrn.4921053).

- 中文题意：基于变量扰动的表格数据异常检测与解释。
- 阅读深度：期刊题录已核验；[同题同作者 SSRN 摘要](https://api.crossref.org/works/10.2139/ssrn.4921053)已读，期刊全文未读，版本细节可能变化。
- 方法与发现：选择性扰动变量，按扰动程度分配标签以构造学习任务；再分析扰动对异常分数的影响，识别重要变量。
- 可引用转述：针对缺乏明显结构先验的表格数据，研究者通过变量扰动构造自监督任务，并将扰动对异常评分的影响用于辅助解释。
- 放入章节：异常检测研究现状、合成异常测试的动机。
- 对项目启示：价格乘倍率可用作受控敏感性测试，但只是有限形式的扰动，不覆盖真实车况、配置和金融价异常。
- 引用限制：不把论文的训练框架等同于本项目简单的合成测试；不声称已复现该方法。
- 下一次精读：对照最终期刊版与公开代码，确认扰动分布、训练集污染设置和解释指标。

## EN07 数据泄漏与虚高指标

ALTURAYEIF N, HASSINE J. Data leakage detection in machine learning code: transfer learning, active learning, or low-shot prompting?[J]. PeerJ Computer Science, 2025, 11: e2730. [DOI](https://doi.org/10.7717/peerj-cs.2730).

- 中文题意：使用迁移学习、主动学习与少样本提示检测机器学习代码的数据泄漏。
- 阅读深度：题录与摘要；未读全文。
- 方法与发现：比较有限标注代码上的三类检测方式，主动学习在其任务中以较少标注获得较优 F2；明确泄漏可能发生在数据构造层面和代码层面。
- 可引用转述：测试信息在数据构造或代码执行过程中进入训练，会导致泛化评估过于乐观。
- 放入章节：预处理流程、数据划分、实验可信度。
- 对项目启示：先划分数据再拟合编码器、填补器和价格统计；同车型参考价格从训练集计算，训练记录需折外处理；不要把实际价与均价之差作为价格预测模型的输入。
- 引用限制：上述工程动作是对本项目的实施建议，不声称逐条来自本文摘要；文中 F2 是代码漏洞任务指标，不是二手车异常性能。
- 下一次精读：提取数据泄漏定义、常见代码模式及其评价协议。

## EN08 异常解释的人工核验

RÖCHNER P, ROTHLAUF F. Evaluating robustly standardized explainable anomaly detection of implausible variables in cancer data[J]. Journal of the American Medical Informatics Association, 2025, 32(4): 724-735. [DOI](https://doi.org/10.1093/jamia/ocaf011). [已读全文页面](https://pmc.ncbi.nlm.nih.gov/articles/PMC12005620/).

- 中文题意：利用人工核验评价癌症记录中异常变量的解释质量。
- 阅读深度：摘要及 Introduction、Methods、Explanation score evaluation、Limitations 定向阅读；未逐表完成全文精读。
- 数据与方法：18587 条记录、8 个类别变量，选择 800 条高异常分数记录；用医学编码人员的变量核验作为参照，比较原始、普通标准化与稳健标准化解释分数。
- 评价：变量级分类用 precision、recall、F1；排序用查全不合理变量所需的排序深度。文中的 coverage 是排序评价量，不是保形区间覆盖率，不能混用术语。
- 可引用转述：解释质量可以通过人工核验的变量标签进行分类和排序评价；单人审核降低了人工成本，但不能衡量标注者之间的一致性。
- 放入章节：人工标注方案、解释评价、研究局限。
- 对项目启示：人工盲审时不显示模型预测价、SHAP 或异常标签；保留“信息不足”，与正常/高价/低价分开；将清单有效性与全体检测召回率分开报告。
- 引用限制：论文只验证 800 个高分记录的解释，不能支撑“所有车源的召回率已验证”。本文自编码器变量解释也不同于价格预测 SHAP。
- 已读全文局限：每条由一名编码者审核，无法计算标注者间一致性；来自一个登记机构，跨机构外推未验证；抽取前约 4.3% 异常分数记录，扩大审核比例的效果需另研究。
- 下一次精读：细读 Medical Coders’ Validation，提取人工操作规则与质控方式。

## EN09 城市群体变化与区间覆盖

BHATTACHARYYA A, BARBER R F. Group-weighted conformal prediction[J]. Electronic Journal of Statistics, 2026, 20(1): 1171-1199. [DOI](https://doi.org/10.1214/26-EJS2506).

- 中文题意：群体加权的保形预测。
- 阅读深度：题录与摘要；未读全文。
- 方法与发现：研究由有限群体决定协变量漂移的情况，分析权重估计误差与加权保形预测覆盖保证。
- 可引用转述：当训练与测试的群体分布发生变化时，加权保形预测依赖的分布权重及其估计误差会影响覆盖保证。
- 放入章节：城市差异、模型外推与未来工作。
- 对项目启示：先报告整体区间覆盖率，再按城市和价格段检查；训练集城市占比与使用场景不同可能改变结果。
- 引用限制：总体边际覆盖不等于每城市、每车型、每辆车的条件覆盖保证。本周不实现该复杂方法。
- 下一次精读：明确论文假设、分组权重如何得到、是否适合本数据稀疏城市。

## EN10 保形预测入门

KAISER T, HERZOG P. A Tutorial on Distribution-Free Uncertainty Quantification Using Conformal Prediction[J]. Advances in Methods and Practices in Psychological Science, 2025, 8(4): 25152459251380452. [DOI](https://doi.org/10.1177/25152459251380452).

- 中文题意：分布无关不确定性量化的保形预测教程。
- 阅读深度：题录与摘要；未读全文。
- 方法与发现：用三个递进公开数据与 R 代码示例介绍保形预测，强调点预测之外表达不确定性的必要性与边际覆盖。
- 可引用转述：保形预测为不同点预测模型提供构造预测区间的统一思路，可用于避免仅看预测值而过度相信模型。
- 放入章节：预测区间定义、方法基础。
- 对项目启示：后续训练集、校准集、测试集职责分离；测试集只能用于最终覆盖率和区间宽度评价，不反复调阈值。
- 引用限制：“分布无关”不等于“无需任何假设”；越界只是复核信号，不是异常真实性概率。
- 下一次精读：从最简单示例理解有限样本分位数、边际覆盖和异方差处理。

## EN11 误报漏报代价与损失控制

WANG D, WANG P, JI Z, et al. Conformal Loss-Controlling Prediction[J]. IEEE Transactions on Neural Networks and Learning Systems, 2025, 36(2): 2973-2983. [DOI](https://doi.org/10.1109/TNNLS.2024.3356512). [公开版本](https://arxiv.org/abs/2301.02424).

- 中文题意：将保形预测扩展为一般损失控制。
- 阅读深度：Crossref 题录、OpenAlex 摘要；未读全文。期刊在线 2024、卷期 2025；公开预印本 2023 不单独计数。
- 方法与发现：在可交换假设下讨论有限样本损失控制，并在分类及数值天气预报后处理上实验。
- 可引用转述：预测区间或预测集合的评价可从单一覆盖指标延伸到与实际任务有关的损失，但需要明确损失定义和适用假设。
- 放入章节：误报漏报成本讨论、后续研究。
- 对项目启示：异常高价和低价的复核目的不同，漏报与误报成本须结合具体业务说明，不能未经论证认定漏报一定更严重。
- 引用限制：不能声称简单残差阈值具有本文的损失控制保证；本周只用于研究思路。
- 下一次精读：确认其 point-wise loss 控制与期望风险控制的区别，再决定是否写入正文方法。

## 本周可执行阅读顺序

1. EN02、EN01：理解挂牌价业务与数据时点；记录采集日期、交易价/挂牌价、样本筛选。
2. EN07、EN03：先避免数据泄漏与解释尺度错误，服务本周“数据—模型—解释”流程。
3. EN08、EN10：为后续人工标注和区间异常检测做准备。
4. EN04、EN05、EN06、EN09、EN11：作为方法扩展及论文讨论资料，不把所有算法都塞进本周。

写论文前，每条引用继续补充：原文页码/章节、阅读日期、原文证据位置、自己的转述、拟用章节。当前笔记中的可引用句均为中文转述草稿，不是原文直接引语。
