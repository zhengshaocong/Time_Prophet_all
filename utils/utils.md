# Utils 工具模块说明

`utils/` 文件夹用于存放各种辅助函数和工具类，提供程序运行所需的通用功能。

## 文件结构

```
utils/
├── __init__.py                    # 包初始化文件
├── file_utils.py                  # 文件操作工具
├── data_utils.py                  # 数据处理工具
├── data_processor.py              # 基础数据处理器
├── data_processor_advanced.py     # 高级数据处理器
├── data_processing_manager.py     # 数据处理管理器
├── cache_utils.py                 # 缓存管理工具
├── config_utils.py                # 配置管理工具
├── visualization_utils.py         # 可视化工具
├── interactive_utils.py           # 交互式界面工具
└── autocorrelation_utils.py       # 自相关分析工具
```

## 功能说明

### file_utils.py
- JSON/CSV/文本文件读写
- 文件格式检查和验证
- 文件名清理和路径处理
- 文件大小格式化

### data_utils.py
- 数据清洗和验证
- 数据过滤和排序
- 数据合并和采样
- DataFrame转换

### data_processor.py
- 基础数据处理功能
- 数据加载和预处理
- 字段映射和转换
- 数据结构分析

### data_processor_advanced.py
- 大数据量处理
- 数据采样和过滤
- 异常值处理
- 数据聚合和转换

### data_processing_manager.py
- 统一数据处理管理
- 模块化数据处理配置
- 数据质量检查
- 处理后数据缓存管理

### config_utils.py
- 配置信息获取和验证
- 数据字段映射管理
- 数据源切换和验证
- 配置管理菜单功能

### cache_utils.py
- 字段映射缓存管理
- 缓存文件读写操作
- 缓存状态检查和清理
- 缓存信息统计和列表

### interactive_utils.py
- 交互式菜单系统
- 用户输入验证
- 进度条和加载动画
- 表格显示和格式化
- 确认对话框

### autocorrelation_utils.py
- 数值化自相关分析
- ACF和PACF计算
- Ljung-Box检验
- 基于ACF/PACF的ARIMA参数确定
- 残差自相关检验
- 综合自相关分析

## 使用示例

### 基础工具使用
```python
from utils.file_utils import read_json, write_csv
from utils.data_utils import clean_data, validate_data
from utils.config_utils import get_data_field_mapping, get_field_name
from utils.cache_utils import save_field_mapping_cache, load_field_mapping_cache
from utils.autocorrelation_utils import quick_autocorrelation_check, analyze_residuals
```

### 数据处理管理器使用
```python
from utils.data_processing_manager import (
    get_data_for_module, should_process_data, get_processed_data_path
)

# 检查模块是否需要数据处理
if should_process_data("basic_analysis"):
    print("基础分析模块启用数据处理")

# 为指定模块获取处理后的数据
processed_data = get_data_for_module("cash_flow_prediction")

# 获取处理后数据路径
data_path = get_processed_data_path("arima_prediction")

### 自相关分析工具使用
```python
from utils.autocorrelation_utils import AutocorrelationAnalyzer, quick_autocorrelation_check

# 快速自相关检查
result = quick_autocorrelation_check(time_series)
if result:
    print(f"建议的ARIMA模型: {result['summary']['suggested_model']}")

# 详细自相关分析
analyzer = AutocorrelationAnalyzer()
analysis = analyzer.comprehensive_analysis(time_series)
if analysis:
    print(f"序列是否平稳: {analysis['summary']['is_stationary']}")
    print(f"建议参数: p={analysis['arima_params']['suggested_p']}, q={analysis['arima_params']['suggested_q']}")

# 残差分析
residual_analysis = analyzer.residual_analysis(model.resid)
if residual_analysis:
    print(f"残差质量: {residual_analysis['quality_assessment']['overall_quality']}")
```
```

### 交互式界面使用
```python
from utils.interactive_utils import (
    print_header, print_success, print_error,
    confirm, select_option, show_menu, show_table
)

# 显示标题
print_header("我的程序", "版本 1.0")

# 显示成功消息
print_success("操作完成")

# 确认操作
if confirm("确定要删除文件吗？", default=False):
    # 执行删除操作
    pass

# 显示菜单
menu_items = [
    {"name": "选项1", "action": func1},
    {"name": "选项2", "action": func2}
]
show_menu(menu_items, "主菜单")

# 显示表格
data = [{"姓名": "张三", "年龄": 25}, {"姓名": "李四", "年龄": 30}]
show_table(data, "用户列表")
```

## 交互式界面特性

### 1. 美观的输出格式
- 彩色状态图标 (✓ ✗ ⚠ ℹ)
- 格式化标题和分隔线
- 进度条和加载动画

### 2. 用户友好的输入
- 智能默认值处理
- 输入验证和错误提示
- 支持中文确认 (是/否)

### 3. 菜单系统
- 多级菜单支持
- 描述性选项说明
- 优雅的退出机制

### 4. 数据展示
- 自动格式化表格
- 列宽自适应
- 清晰的分隔线 