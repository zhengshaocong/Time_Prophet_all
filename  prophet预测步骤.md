## Prophet 预测步骤（优化版）

本指南提供一套可直接落地的 Prophet 预测流程，覆盖数据校验、特征、参数、交叉验证、预测输出与上线监控，符合本项目产物与目录规范。

### 目录与产物规范
- **数据输出**: `output/data/`
- **图像输出**: `output/images/prophet/`
- **配置/日志**: 与运行日志一并保存，便于复现实验

### 1. 数据理解与基础校验
- **粒度与时区**: 明确日/周/月/小时，统一为单一时区
- **连续性**: 检查时间连续性，定位缺口（后续是否插值或剔除）
- **重复时间点**: 发现则聚合（求和/均值，依据业务指标）
- **异常值**: 识别明显错误点（负值、尖峰），记录修正策略

### 2. 数据清洗与缺失处理
- **类型**: `ds` 转标准时间，`y` 转数值
- **补齐频率**: 统一 `freq`（如 `D`、`H`），按频率补齐时间轴
- **缺失策略**:
  - 小缺口：线性插值或前后值填充
  - 大缺口：剔除该区段并备注
- **排序去重**: `ds` 单调递增、唯一；去除 NaN

### 3. 数据格式标准化
- 仅保留 Prophet 必需列：`ds`（时间）、`y`（目标）
- 外生因素/事件：保留独立列（后续 `add_regressor`/节假日）
- 统一频率与时区，确保 `ds` 唯一

### 4. 特征与业务事件（可选）
- **节假日**: `add_country_holidays('CN')` 或自定义节假日 DataFrame
- **自定义季节性**: 如月度 `period=30.5`、小时级 `period=24`
- **回归因子**: 促销/价格/公告等，用 `add_regressor`；注意未来值可用性，避免数据泄漏
- **容量上限**: 存在上限/地板时改用 `growth='logistic'` 并设置 `cap/floor`

### 5. 模型参数配置（小步试探）
- **核心**
  - `growth`: 'linear' 或 'logistic'（有上限）
  - `seasonality_mode`: 'additive'（幅度稳定）/ 'multiplicative'（幅度随水平放大）
  - `yearly/weekly/daily_seasonality`: 按粒度/业务启用
- **关键先验**
  - `changepoint_prior_scale`: 趋势拐点灵活度（默认 0.05；不稳→0.01，变化快→0.1~0.5）
  - `seasonality_prior_scale`: 季节性力度（如 5/10/15）
  - `holidays_prior_scale`: 节假日力度（如 5/10/15）
  - `changepoint_range`: 拐点搜索范围（默认 0.8）

### 6. 交叉验证与调参（滚动起点）
- **切分**: 例如历史前 80% 训练，滚动向前预测 7/14/30 天
- **指标**: RMSE、MAE、MAPE/SMAPE；覆盖率（yhat 区间命中率）
- **小网格优先**:
  - `changepoint_prior_scale ∈ {0.01, 0.05, 0.1}`
  - `seasonality_mode ∈ {additive, multiplicative}`
  - `seasonality_prior_scale ∈ {5, 10, 15}`
  - 观察稳定性后再扩展搜索

### 7. 训练
- 用标准化后的 `ds/y` 及确认的节假日/自定义季节性/回归因子拟合
- 保存：模型对象、参数、训练时间范围、数据校验摘要

### 8. 预测时间范围构造
- 使用 `make_future_dataframe(periods, freq, include_history=True)`，`freq` 与历史一致
- 若有回归因子，需在 future 中补齐未来值（如促销计划）；缺失将导致预测异常

### 9. 生成预测与结果还原
- 输出列：`ds`, `yhat`, `yhat_lower`, `yhat_upper`
- 若训练前做过对数/标准化，需对三列同时逆变换，保持置信区间一致性
- 产物：保存至 `output/data/prophet_predictions_时间戳.csv`

### 10. 可视化与解读
- 历史 vs 预测对比图；组件图 `plot_components`（趋势/季节/节假日）
- 结合业务事件与回归因子曲线解释异常点
- 图像输出：`output/images/prophet/`

### 11. 评估与对标（可选）
- 回测窗口计算 RMSE、MAE、MAPE；与基线（Naive、移动平均）对比
- 记录版本号、数据切分、参数，支持回溯

### 12. 上线与监控
- **滚动更新**: 按日/周重训或增量拟合，结合数据新鲜度
- **漂移监控**: 最近窗口误差上升、残差偏移、覆盖率下降告警
- **可恢复性**: 预测 CSV、图像、参数与日志均落盘并带版本

### 参数与策略建议（速查）
- 明显比例型季节性：`seasonality_mode='multiplicative'`
- 历史变化快：提高 `changepoint_prior_scale`，用交叉验证兜底防过拟合
- 业务上限/地板：改 `growth='logistic'` 并设置 `cap/floor`
- 回归因子：确保未来可得值，避免泄漏

### 常见坑位
- 时间重复/频率混乱→先聚合或重采样
- 未来回归因子缺失→预测异常或偏移
- 对数变换忘记逆变换→预测偏小
- 训练集过短/结构突变→适当增大拐点灵活度，缩短预测跨度

### 最小示例（仅核对思路）
```python
from prophet import Prophet

m = Prophet(
    growth="linear",
    yearly_seasonality=True,
    weekly_seasonality=True,
    seasonality_mode="additive",
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    holidays_prior_scale=10,
    changepoint_range=0.8,
)
# 自定义季节性/节假日/回归因子（按需启用）
m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
# m.add_country_holidays("CN")
# m.add_regressor("promo")

m.fit(df_train[["ds", "y"]])
future = m.make_future_dataframe(periods=30, freq="D", include_history=True)
# future["promo"] = ...  # 若使用回归因子，需补未来值
pred = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
# 若有对数/标准化，请在此处对三列做逆变换
# 保存到 output/data/ 、图像保存到 output/images/prophet/
```