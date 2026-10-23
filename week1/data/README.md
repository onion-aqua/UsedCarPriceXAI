# 数据准备

仓库保留代码、文献、汇总报告和图表，不包含146,354,762字节的原始CSV、全量派生数据、虚拟环境及模型二进制。

## 原始文件

- 文件名：`懂车帝全国各个省会城市二手车相关数据.csv`
- 编码：GB18030/CP936兼容。
- 行数：143,793条数据记录，21列。
- SHA-256：`605bf34562a6e909365df295de10ba084207fcbbbf2604a11553df6ca696a521`
- 放置位置：仓库根目录，或通过`preprocess.py --source`指定。
- 来源发布页与采集日志尚待补齐，不能虚构下载链接或以文件修改时间证明采集日期。

数据并非只有省会车源。预处理会按SKU去重、筛选31个目标城市，得到119,662行。实际字段映射见[字段字典](../reports/data_dictionary.md)。

## 真实样例

`sample_capital_cars.csv`从清洗数据固定随机抽取100行（seed=2026），保留模型使用的属性和挂牌价。删除商家、链接、源记录标识等非必要字段，只新增行号sample_id。

这是真实样本的子集，不是模拟数据，也不是完整训练集，不能用这100行复现报告中的全量指标。读取样例不需要原始数据：

```python
import pandas as pd
sample = pd.read_csv("week1/data/sample_capital_cars.csv")
print(sample.head())
```

## 本地生成文件

- `processed_capital_cars.csv`：主训练数据，price为万元，power为kW，mileage_km为公里。
- `preprocessing_all_rows_audit.csv`：全部输入记录的清洗状态。
- `duplicate_sku_archive.csv`：含保留项的完整重复原档。
- `preprocessing_excluded_rows.csv`：排除原因和记录。
- `category_vocabulary.csv`：可读类别字典；不作为连续数值编码输入模型。

上述全量文件由预处理脚本生成，已加入.gitignore。缺失值由模型训练流程处理，不伪造原始能源类型或功率。
