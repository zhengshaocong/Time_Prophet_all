# -*- coding: utf-8 -*-
"""
数据字段映射配置文件
定义数据字段的映射关系和格式
"""

# ==================== 数据字段映射配置 ====================
# 默认数据字段映射配置（作为备用，只包含通用字段）
DEFAULT_FIELD_MAPPING = {
    "时间字段": "report_date",
    "时间格式": "%Y%m%d",
    "用户ID字段": "user_id"
}

# 当前使用的数据源配置
CURRENT_DATA_SOURCE = "auto_detected" 