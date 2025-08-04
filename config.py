# -*- coding: utf-8 -*-
"""
程序配置文件
用于设定程序运行时的各种参数和路径

注意：此文件现在作为配置的统一入口，实际配置已分散到config文件夹中的各个文件
"""

# ==================== 导入分散的配置文件 ====================
# 路径配置
from config.paths import (
    ROOT_DIR, DATA_DIR, CACHE_DIR, TEMP_DIR,
    OUTPUT_DIR, IMAGES_DIR, OUTPUT_DATA_DIR,
    UTILS_DIR, SRC_DIR, SCRIPT_DIR, TESTS_DIR,
    SUPPORTED_IMAGE_FORMATS, SUPPORTED_DATA_FORMATS,
    DEFAULT_IMAGE_FORMAT, DEFAULT_DATA_FORMAT
)

# 数据字段映射配置
from config.data_mapping import (
    DEFAULT_FIELD_MAPPING, CURRENT_DATA_SOURCE
)

# 数据处理配置
from config.data_processing import (
    GLOBAL_DATA_PROCESSING_CONFIG, DATA_PREPROCESSING_CONFIG,
    DATA_PROCESSING_CONFIG
)

# 特征工程配置
from config.features import (
    FEATURE_ENGINEERING_CONFIG, DATA_TRANSFORMATION_CONFIG
)

# 模型配置
from config.models import (
    ARIMA_TRAINING_CONFIG
)

# 功能模块配置
from config.modules import (
    BASIC_ANALYSIS_CONFIG, PURCHASE_REDEMPTION_CONFIG,
    CASH_FLOW_PREDICTION_CONFIG, MODEL_EVALUATION_CONFIG
)

# ==================== 配置导出 ====================
# 为了保持向后兼容性，将所有配置重新导出
__all__ = [
    # 路径配置
    'ROOT_DIR', 'DATA_DIR', 'CACHE_DIR', 'TEMP_DIR',
    'OUTPUT_DIR', 'IMAGES_DIR', 'OUTPUT_DATA_DIR',
    'UTILS_DIR', 'SRC_DIR', 'SCRIPT_DIR', 'TESTS_DIR',
    'SUPPORTED_IMAGE_FORMATS', 'SUPPORTED_DATA_FORMATS',
    'DEFAULT_IMAGE_FORMAT', 'DEFAULT_DATA_FORMAT',
    
    # 数据字段映射配置
    'DEFAULT_FIELD_MAPPING', 'CURRENT_DATA_SOURCE',
    
    # 数据处理配置
    'GLOBAL_DATA_PROCESSING_CONFIG', 'DATA_PREPROCESSING_CONFIG',
    'DATA_PROCESSING_CONFIG',
    
    # 特征工程配置
    'FEATURE_ENGINEERING_CONFIG', 'DATA_TRANSFORMATION_CONFIG',
    
    # 模型配置
    'ARIMA_TRAINING_CONFIG',
    
    # 功能模块配置
    'BASIC_ANALYSIS_CONFIG', 'PURCHASE_REDEMPTION_CONFIG',
    'CASH_FLOW_PREDICTION_CONFIG', 'MODEL_EVALUATION_CONFIG'
] 