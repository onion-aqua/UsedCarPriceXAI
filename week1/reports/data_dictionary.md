# 建模字段字典

价格单位：万元；里程：公里；年龄：月。所有flag仅为数据质量/描述标志，不能当作真实异常标签。

| 原字段 | 输出字段 | 规则 |
|---|---|---|
| id | id | 原记录标识，非车辆去重依据 |
| 详情链接 | sku_id | URL sku_id 提取，车辆去重标识 |
| 价格 | price | 挂牌价，万元；仅缺失/非正值排除，极端高价保留 |
| 价格 | log_price | log1p(price)，反变换使用expm1 |
| 品牌名称 | brand | 品牌类别，保留语义字符串 |
| 汽车名称 | model | 车系类别，保留语义字符串 |
| 汽车发动机名称 | variant | 实际内容是配置版本，非发动机排量 |
| 标题 | title | 原始车源标题 |
| 车源城市名称 | city | 车源城市，限定31个省会/自治区首府/直辖市 |
| 汽车第一次注时间 | registration_date | 首次注册日期，1970占位值置缺失 |
| 汽车第一次注时间 | registration_year | 首次注册年份 |
| 汽车生产年份 | model_year | 原字段可能为款型年，不能直接认作制造年份 |
| 汽车第一次注时间 | vehicle_age_current | 任务要求：参考年2026减注册年；仅展示，不输入历史挂牌价模型 |
| 汽车年龄 | age_at_listing_months | 原记录年龄解析为月，用于挂牌时点价格模型 |
| 里程数 | mileage_km | 统一公里，保留零里程并标记 |
| 里程数+汽车年龄 | annual_mileage_km | mileage_km/(age_at_listing_months/12)，不足12月不外推年均 |
| 用车成本→发动机 | engine_liters | 错位字段解析排量L；纯电等无排量时保持缺失 |
| 用车成本→发动机 | engine_text | 原发动机描述 |
| 用车成本→变速箱 | gearbox | 原变速箱类别 |
| 用车成本→轴距 | wheelbase_mm | 轴距mm，明显非法值置缺失并标记 |
| 相关参数→排放标准 | emission_standard | 档案列表中的排放标准 |
| 相关参数→能源类型 | fuelType | 仅提取明确提供的能源类型，其余缺失，不按品牌猜测 |
| 用车成本→最大功率 | power | 仅提取明确提供的最大功率kW，其余缺失；与排量不同 |
| Column1 | dealer | 实际公司名称 |
| 详情链接 | detail_url | 原始详情来源 |
| 档案详情 | source_flag_unknown | 实际为0/1，含义未证实，不强行命名事故/固定价 |

source_row是原CSV物理记录序号（表头计作第1行）；dedup_key是SKU优先的去重键；selection_status记录排除原因。inferred_snapshot_month仅为间接时点线索。

推荐第一周模型数值输入：age_at_listing_months、mileage_km、annual_mileage_km、engine_liters、wheelbase_mm、power（kW）；类别输入：brand、model、variant、city、gearbox、emission_standard、fuelType。功率和能源类型高缺失，插补/缺失指示须只在训练流程拟合。

禁止将price、log_price、price高价标志、source_row、id/SKU、URL、推断月份、当前车龄作为此基线模型输入。人工标签与未来目标相关统计必须独立设计。
