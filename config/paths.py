# -*- coding: utf-8 -*-
"""
路径配置文件
定义程序运行时的各种路径
"""

from pathlib import Path

# ==================== 基础路径配置 ====================
# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据相关路径
DATA_DIR = ROOT_DIR / "data"           # 原始数据目录
CACHE_DIR = ROOT_DIR / "cache"         # 缓存文件目录
TEMP_DIR = ROOT_DIR / "temp"           # 临时文件目录

# 输出相关路径
OUTPUT_DIR = ROOT_DIR / "output"       # 输出根目录
IMAGES_DIR = OUTPUT_DIR / "images"     # 图片输出目录
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"  # 数据输出目录

# 程序相关路径
UTILS_DIR = ROOT_DIR / "utils"         # 工具模块目录
SRC_DIR = ROOT_DIR / "src"             # 源代码目录
SCRIPT_DIR = ROOT_DIR / "script"       # 脚本目录
TESTS_DIR = ROOT_DIR / "tests"         # 测试目录

# ==================== 文件格式配置 ====================
# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

# 支持的数据格式
SUPPORTED_DATA_FORMATS = ['.csv', '.json', '.xlsx', '.txt', '.pickle']

# 默认输出格式
DEFAULT_IMAGE_FORMAT = '.png'
DEFAULT_DATA_FORMAT = '.csv' 