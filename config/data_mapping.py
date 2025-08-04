# -*- coding: utf-8 -*-
"""
数据字段映射配置文件
定义数据字段的映射关系和格式
"""

from config.paths import CACHE_DIR

# ==================== 数据字段映射配置 ====================
# 缓存文件路径
FIELD_MAPPING_CACHE_FILE = CACHE_DIR / "field_mapping_cache.json"

# 默认数据字段映射配置（作为备用）
DEFAULT_FIELD_MAPPING = {
    "时间字段": "report_date",
    "时间格式": "%Y%m%d",
    "申购金额字段": "total_purchase_amt",
    "赎回金额字段": "total_redeem_amt",
    "当前余额字段": "tBalance",
    "昨日余额字段": "yBalance",
    "用户ID字段": "user_id",
    "消费金额字段": "consume_amt",
    "转账金额字段": "transfer_amt",
    "分类字段": ["category1", "category2", "category3", "category4"]
}

# 当前使用的数据源配置
CURRENT_DATA_SOURCE = "auto_detected" 