# Config 配置文件夹说明

## 概述
config文件夹包含了程序的所有配置参数，按照功能模块进行了分散管理，便于维护和扩展。

## 文件结构

### 1. `__init__.py`
- **功能**: 配置模块的初始化文件
- **作用**: 提供配置的统一访问接口，导入所有分散的配置文件
- **使用方式**: `from config import *` 或 `import config`

### 2. `paths.py`
- **功能**: 路径配置文件
- **包含内容**:
  - 项目根目录路径
  - 数据相关路径（data、cache、temp）
  - 输出相关路径（output、images、data）
  - 程序相关路径（utils、src、script、tests）
  - 文件格式配置（支持的图片和数据格式）

### 3. `data_mapping.py`
- **功能**: 数据字段映射配置
- **包含内容**:
  - 字段映射缓存文件路径
  - 默认数据字段映射配置
  - 当前使用的数据源配置

### 4. `data_processing.py`
- **功能**: 数据处理相关配置
- **包含内容**:
  - 全局数据处理开关配置
  - 数据预处理配置（缺失值、异常值、时间特征）
  - 大数据量处理配置（采样、过滤、聚合、质量检查）

### 5. `features.py`
- **功能**: 特征工程和数据转换配置
- **包含内容**:
  - 特征工程配置（基础特征、时间特征、统计特征、用户特征、业务特征）
  - 数据转换配置（标准化、编码、时间序列特征、特征选择）

### 6. `models.py`
- **功能**: 预测模型配置
- **包含内容**:
  - ARIMA模型训练配置
  - 模型参数配置
  - 预测配置
  - 性能评估配置

### 7. `modules.py`
- **功能**: 功能模块配置
- **包含内容**:
  - 基础数据分析模块配置
  - 申购赎回分析模块配置
  - 资金流预测模块配置
  - 模型评估模块配置

## 使用方式

### 方式1: 从根目录config.py导入（推荐）
```python
import config
# 或者
from config import ROOT_DIR, DATA_DIR
```

### 方式2: 从config文件夹直接导入
```python
from config.paths import ROOT_DIR, DATA_DIR
from config.data_processing import DATA_PROCESSING_CONFIG
```

### 方式3: 导入整个config模块
```python
from config import *
```

## 配置修改建议

1. **路径配置**: 修改 `paths.py` 中的路径设置
2. **数据处理**: 修改 `data_processing.py` 中的处理参数
3. **特征工程**: 修改 `features.py` 中的特征配置
4. **模型参数**: 修改 `models.py` 中的模型配置
5. **模块配置**: 修改 `modules.py` 中的功能模块配置

## 注意事项

1. 所有配置文件都使用UTF-8编码，注释使用中文
2. 配置修改后需要重启程序才能生效
3. 建议在修改配置前备份原配置文件
4. 配置参数都有详细的中文注释，便于理解和使用 