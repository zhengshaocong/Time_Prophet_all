# -*- coding: utf-8 -*-
"""
文件命名配置
控制程序生成文件时的命名模式
"""

# ==================== 文件命名模式配置 ====================
# 全局文件命名模式
# 可选值：
# - "overwrite": 覆盖模式，新文件覆盖旧文件（推荐）
# - "version": 版本模式，新文件添加版本号
# - "timestamp": 时间戳模式，新文件添加时间戳
GLOBAL_NAMING_MODE = "overwrite"

# ==================== 各模块文件命名配置 ====================
# ARIMA预测模块
ARIMA_NAMING_CONFIG = {
    "模式": "overwrite",  # 继承全局模式
    "预测结果文件": "multi_arima_predictions.csv",
    "综合预测图": "multi_comprehensive_predictions.png",
    "预测摘要图": "multi_prediction_summary.png"
}

# 经典分解法预测模块
CLASSICAL_DECOMPOSITION_NAMING_CONFIG = {
    "模式": "overwrite",  # 继承全局模式
    "主预测结果": "prediction_results.csv",
    "申购赎回预测": "purchase_redeem_predictions.csv",
    "预测结果JSON": "prediction_results.json",
    "对比预测图": "purchase_redeem_compare.png"
}

# 数据处理模块
DATA_PROCESSING_NAMING_CONFIG = {
    "模式": "overwrite",  # 继承全局模式
    "处理后数据": "processed_data.csv",
    "处理日志": "processing_log.json",
    "异常检测结果": "anomalies.json"
}

# Prophet预测模块
PROPHET_NAMING_CONFIG = {
    "模式": "overwrite",  # 继承全局模式
    "拟合明细": "prophet_fit.csv",
    "拟合报告": "prophet_fit_report.txt"
}

# 参数优化模块
OPTIMIZATION_NAMING_CONFIG = {
    "模式": "overwrite",  # 继承全局模式
    "申购优化结果": "purchase_optimization_results.csv",
    "赎回优化结果": "redeem_optimization_results.csv",
    "优化参数配置": "optimized_params_config.py"
}

# ==================== 文件扩展名配置 ====================
# 数据文件扩展名
DATA_EXTENSIONS = {
    "CSV": ".csv",
    "JSON": ".json",
    "EXCEL": ".xlsx",
    "PICKLE": ".pkl",
    "PARQUET": ".parquet"
}

# 图片文件扩展名
IMAGE_EXTENSIONS = {
    "PNG": ".png",
    "JPG": ".jpg",
    "JPEG": ".jpeg",
    "SVG": ".svg",
    "PDF": ".pdf"
}

# 报告文件扩展名
REPORT_EXTENSIONS = {
    "TEXT": ".txt",
    "MARKDOWN": ".md",
    "HTML": ".html"
}

# ==================== 目录结构配置 ====================
# 输出目录结构
OUTPUT_DIRECTORY_STRUCTURE = {
    "数据": "data",
    "图片": "images",
    "报告": "reports",
    "日志": "logs",
    "配置": "configs"
}

# 各模块输出目录
MODULE_OUTPUT_DIRECTORIES = {
    "ARIMA": "images/arima",
    "经典分解法": "classical_decomposition",
    "Prophet": "prophet",
    "数据处理": "data",
    "参数优化": "classical_decomposition"  # 经典分解法参数优化
}

# ==================== 文件命名规则配置 ====================
# 文件名前缀规则
FILENAME_PREFIXES = {
    "预测结果": "prediction",
    "处理结果": "processed",
    "优化结果": "optimization",
    "评估结果": "evaluation",
    "配置": "config"
}

# 文件名后缀规则
FILENAME_SUFFIXES = {
    "主结果": "",
    "详细": "_detailed",
    "摘要": "_summary",
    "对比": "_compare",
    "报告": "_report"
}

# ==================== 时间格式配置 ====================
# 时间戳格式（仅在时间戳模式下使用）
TIMESTAMP_FORMATS = {
    "日期时间": "%Y%m%d_%H%M%S",
    "日期": "%Y%m%d",
    "时间": "%H%M%S",
    "ISO": "%Y-%m-%dT%H:%M:%S"
}

# ==================== 版本号配置 ====================
# 版本号格式（仅在版本模式下使用）
VERSION_FORMATS = {
    "简单": "v{version}",
    "详细": "v{version}_{date}",
    "完整": "v{version}.{minor}.{patch}"
}

# ==================== 配置验证 ====================
def validate_naming_config():
    """验证文件命名配置的有效性"""
    valid_modes = ["overwrite", "version", "timestamp"]
    
    if GLOBAL_NAMING_MODE not in valid_modes:
        raise ValueError(f"无效的全局命名模式: {GLOBAL_NAMING_MODE}")
    
    # 验证各模块配置
    modules = [
        ARIMA_NAMING_CONFIG,
        CLASSICAL_DECOMPOSITION_NAMING_CONFIG,
        DATA_PROCESSING_NAMING_CONFIG,
        PROPHET_NAMING_CONFIG,
        OPTIMIZATION_NAMING_CONFIG
    ]
    
    for module_config in modules:
        if "模式" in module_config:
            mode = module_config["模式"]
            if mode != "overwrite" and mode not in valid_modes:
                raise ValueError(f"无效的模块命名模式: {mode}")

# ==================== 配置导出 ====================
__all__ = [
    'GLOBAL_NAMING_MODE',
    'ARIMA_NAMING_CONFIG',
    'CLASSICAL_DECOMPOSITION_NAMING_CONFIG',
    'DATA_PROCESSING_NAMING_CONFIG',
    'PROPHET_NAMING_CONFIG',
    'OPTIMIZATION_NAMING_CONFIG',
    'DATA_EXTENSIONS',
    'IMAGE_EXTENSIONS',
    'REPORT_EXTENSIONS',
    'OUTPUT_DIRECTORY_STRUCTURE',
    'MODULE_OUTPUT_DIRECTORIES',
    'FILENAME_PREFIXES',
    'FILENAME_SUFFIXES',
    'TIMESTAMP_FORMATS',
    'VERSION_FORMATS',
    'validate_naming_config'
]

# 验证配置
try:
    validate_naming_config()
except Exception as e:
    print(f"文件命名配置验证失败: {e}")
    # 使用默认配置
    GLOBAL_NAMING_MODE = "overwrite"
