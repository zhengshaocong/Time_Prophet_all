# -*- coding: utf-8 -*-
"""
全局预测时间配置
统一控制各模型（ARIMA、Prophet、经典分解）的预测时间范围
"""

GLOBAL_FORECAST_CONFIG = {
    # 是否启用全局预测结束日期（若启用，则以该日期为截止，不包含该日）
    "enabled": True,
    # 全局预测的结束日期（不包含当天），例如希望预测到 2015-01-01 之前，则填写 "2015-01-01"
    "end_date_exclusive": "2015-01-01",
    # 日期格式
    "date_format": "%Y-%m-%d",
    # 频率（目前统一按日频率处理）
    "freq": "D"
}
