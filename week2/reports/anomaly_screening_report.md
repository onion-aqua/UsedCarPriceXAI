# 疑似异常挂牌价格初筛

本报告生成的是人工复核队列，不是异常真值分类器。

## 判定组成

- 模型信号：测试集挂牌价是否越过独立校准的90%预测区间。
- 同类信号：优先使用训练集中的城市+车系+注册年，回退到城市+品牌+注册年、城市+品牌、品牌+车系+注册年、品牌+车系；至少8辆训练车源才启用同类统计。对单独城市或全品牌均价不作车型参考，避免把豪华车与全市场混比。
- 同类偏离阈值：相对训练集同类中位价达到±35%时标记为同类高/低价信号。
- 输出为疑似异常高价/疑似异常低价，必须人工查看车况、配置、价格类型和同期可比挂牌。

## 复核样本

Top-K高价 50 条、Top-K低价 50 条、随机区间内样本 30 条。
复核模板中已预留human_label、human_confidence、reason_codes、外部平台URL、query_date和review_notes。标注人应先隐藏screening_label和screening_score，完成盲审后再合并模型结果。

## 不应报告的指标

没有真实异常标签，当前不能报告整体Precision、Recall、F1或宣称发现虚假车源。Top-K人工核查可以报告Precision@K，但必须给出样本来源和标注规则。

## 运行

```powershell
.\.venv\Scripts\python.exe -X utf8 week2\src\anomaly_detection.py
```
