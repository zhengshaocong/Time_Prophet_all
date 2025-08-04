# Src 源代码模块说明

`src/` 文件夹用于存放主要的业务逻辑代码，包含资金流预测系统的核心功能模块。

## 文件结构

```
src/
├── __init__.py                    # 包初始化文件
├── analysis/                      # 数据分析模块
│   ├── __init__.py
│   ├── basic_analysis.py          # 基础数据分析
│   └── purchase_redemption_analysis.py  # 申购赎回分析
├── prediction/                    # 预测模块
│   ├── __init__.py
│   ├── cash_flow_predictor.py     # 资金流预测
│   └── arima_predictor.py         # ARIMA预测
├── data_processing.py             # 数据处理模块
└── cash_flow_prediction.py        # 资金流预测主模块（兼容性）
```

## 功能说明

### analysis/ 数据分析模块

#### basic_analysis.py
基础数据分析模块，包含以下主要功能：

**主要类**
- **BasicDataAnalysis**: 基础数据分析类
  - `load_data()`: 加载数据文件
  - `analyze_data_structure()`: 分析数据结构并解析字段
  - `save_data_analysis()`: 保存数据分析结果到Markdown文件
  - `visualize_data()`: 基础数据可视化，生成多种图表
  - `auto_detect_field_mapping()`: 自动检测字段映射

**主要函数**
- **run_basic_data_analysis()**: 运行基础数据分析功能
  - 自动检测数据字段映射
  - 生成字段映射缓存
  - 生成基础数据可视化图表
  - 保存分析报告

#### purchase_redemption_analysis.py
申购赎回分析模块，包含以下主要功能：

**主要类**
- **PurchaseRedemptionAnalysis**: 申购赎回分析类
  - `analyze_purchase_redemption_trends()`: 分析申购赎回趋势
  - `visualize_purchase_redemption()`: 申购赎回可视化
  - `generate_statistics()`: 生成统计信息

**主要函数**
- **run_purchase_redemption_analysis()**: 运行申购赎回分析功能
  - 预处理数据以区分申购和赎回
  - 生成申购赎回时间序列对比
  - 生成分布对比和热力图
  - 保存分析结果

### prediction/ 预测模块

#### arima_predictor.py
ARIMA预测模块，使用ARIMA模型进行申购赎回趋势预测：

**主要类**
- **ARIMAPredictor**: ARIMA预测器类
  - `prepare_training_data()`: 准备训练数据
  - `check_stationarity()`: 检查时间序列平稳性
  - `determine_differencing()`: 确定差分阶数
  - `find_best_arima_params()`: 寻找最优ARIMA参数
  - `fit_arima_model()`: 拟合ARIMA模型
  - `make_forecast()`: 进行预测
  - `visualize_results()`: 可视化预测结果
  - `save_results()`: 保存预测结果

**主要功能**
1. **数据准备**: 自动处理训练时间范围和数据极限
2. **平稳性检验**: 使用ADF检验检查时间序列平稳性
3. **参数优化**: 自动寻找最优的ARIMA(p,d,q)参数
4. **模型拟合**: 拟合ARIMA模型并评估模型质量
5. **预测**: 生成未来时间段的预测值和置信区间
6. **可视化**: 生成历史数据和预测结果的对比图表
7. **结果保存**: 保存预测结果到CSV文件

**配置参数**
- 训练时间范围：自动检测或手动指定
- 数据极限：最小/最大数据量、采样设置
- 模型参数：ARIMA参数范围、差分阶数
- 预测配置：预测步数、置信区间
- 性能评估：评估指标、基准模型

#### cash_flow_predictor.py
资金流预测模块，包含以下主要功能：

**主要类**
- **CashFlowPredictor**: 资金流预测类
  - `train_model()`: 训练预测模型
  - `make_prediction()`: 进行预测
  - `evaluate_model()`: 评估模型性能

### data_processing.py
数据处理模块，包含完整的数据处理流水线，支持数据清洗、特征工程、数据转换等功能：

**主要类**
- **DataProcessingPipeline**: 数据处理流水线类
  - `load_and_analyze_data()`: 加载并分析数据
  - `clean_data()`: 数据清洗（缺失值、异常值、数据类型转换等）
  - `engineer_features()`: 特征工程（基础特征、时间特征、统计特征等）
  - `transform_data()`: 数据转换（标准化、编码、特征选择等）
  - `save_processed_data()`: 保存处理后的数据
  - `run_full_pipeline()`: 运行完整的数据处理流水线

**主要功能**
1. **数据清洗**:
   - 缺失值处理：根据配置填充或删除缺失值
   - 异常值处理：使用统计方法检测和处理异常值
   - 数据类型转换：确保字段类型正确
   - 时间字段处理：提取时间特征
   - 数据一致性检查：检查数据逻辑一致性

2. **特征工程**:
   - 基础特征：净资金流、总资金流、资金流比率、余额变化等
   - 时间特征：年、月、日、星期、季度、年中天数、月初月末、周末、节假日等
   - 统计特征：滚动窗口统计（均值、标准差、最大值、最小值等）
   - 用户特征：用户级别统计、交易频率、用户分类等
   - 业务特征：交易活跃度、资金流类型、余额水平等

3. **数据转换**:
   - 标准化：Z-score、Min-Max、Robust标准化
   - 编码：独热编码、标签编码
   - 时间序列特征：滞后特征
   - 特征选择：缺失值、低方差、高相关性特征过滤

**配置支持**
- 支持通过config.py中的配置参数控制处理流程
- 可配置的特征工程选项
- 可配置的数据转换方法
- 可配置的特征选择阈值

**主要函数**
- **run_data_processing()**: 运行数据处理功能
  - 创建数据处理流水线
  - 执行完整的数据处理流程
  - 保存处理结果和日志

### cash_flow_prediction.py
资金流预测系统的核心模块（兼容性版本），包含以下主要功能：

#### 主要类
- **CashFlowPrediction**: 资金流预测主类
  - `load_data()`: 加载数据文件
  - `analyze_data_structure()`: 分析数据结构并解析字段
  - `save_data_analysis()`: 保存数据分析结果到Markdown文件
  - `visualize_data()`: 基础数据可视化，生成多种图表
  - `preprocess_data_for_purchase_redemption()`: 预处理数据以区分申购和赎回
  - `visualize_purchase_redemption()`: 申购与赎回时间序列可视化
  - `get_data_summary()`: 获取数据摘要信息

#### 主要函数
- **run_basic_data_analysis()**: 运行基础数据分析功能
  - 读取数据的前五行并解析字段
  - 生成基础数据可视化图表
  - 生成申购与赎回时间序列分析
  - 保存分析报告

#### 基础可视化功能
1. **时间序列图**: 显示资金流随时间的变化趋势
2. **分布直方图**: 展示资金流量的分布情况
3. **箱线图**: 显示资金流量的统计特征
4. **24小时热力图**: 展示不同时间段的资金流模式

#### 申购赎回分析功能
1. **数据预处理**: 自动区分申购和赎回数据
   - 基于中位数阈值区分申购和赎回
   - 添加时间特征（小时、日期、月份、星期）
   - 创建申购和赎回金额字段

2. **申购赎回可视化**:
   - **时间序列对比图**: 申购（绿色）与赎回（红色）的时间序列对比
   - **分布对比图**: 申购与赎回的分布直方图对比
   - **24小时热力图**: 显示不同时间段的申购赎回模式
   - **统计信息面板**: 详细的申购赎回统计信息

#### 数据输出
- 生成 `data/data_analysis.md` 文件，包含详细的字段分析报告
- 生成 `output/images/cash_flow_analysis.png` 文件，包含基础可视化图表
- 生成 `output/images/purchase_redemption_analysis.png` 文件，包含申购赎回分析图表

## 使用示例

### 基础使用
```python
from src.cash_flow_prediction import CashFlowPrediction

# 创建实例
cfp = CashFlowPrediction()

# 加载数据
cfp.load_data()

# 分析数据结构
cfp.analyze_data_structure(top_rows=5)

# 生成基础可视化
cfp.visualize_data(save_plot=True)

# 生成申购赎回分析
cfp.visualize_purchase_redemption(save_plot=True)
```

### 直接运行分析
```python
from src.cash_flow_prediction import run_basic_data_analysis

# 运行完整的基础数据分析（包含申购赎回分析）
run_basic_data_analysis()
```

### 申购赎回分析
```python
from src.cash_flow_prediction import CashFlowPrediction

cfp = CashFlowPrediction()
cfp.load_data()

# 预处理数据
cfp.preprocess_data_for_purchase_redemption()

# 生成申购赎回可视化
cfp.visualize_purchase_redemption(save_plot=True)
```

### ARIMA预测
```python
from src.prediction.arima_predictor import ARIMAPredictor

# 创建ARIMA预测器
predictor = ARIMAPredictor()

# 加载数据
predictor.load_data()

# 准备训练数据
time_series = predictor.prepare_training_data()

# 检查平稳性
is_stationary = predictor.check_stationarity(time_series)

# 确定差分阶数
d = predictor.determine_differencing(time_series)

# 寻找最优参数
best_params = predictor.find_best_arima_params(time_series, d)

# 拟合模型
predictor.fit_arima_model(time_series, best_params)

# 进行预测
predictor.make_forecast()

# 可视化结果
predictor.visualize_results()

# 保存结果
predictor.save_results()
```

### 直接运行ARIMA预测
```python
from src.prediction.arima_predictor import run_arima_prediction

# 运行完整的ARIMA预测流程
run_arima_prediction()
```

### 数据处理
```python
from src.data_processing import DataProcessingPipeline

# 创建数据处理流水线
pipeline = DataProcessingPipeline()

# 运行完整的数据处理流水线
success = pipeline.run_full_pipeline()

# 或者分步执行
pipeline.load_and_analyze_data()
pipeline.clean_data()
pipeline.engineer_features()
pipeline.transform_data()
pipeline.save_processed_data()
```

### 直接运行数据处理
```python
from src.data_processing import run_data_processing

# 运行完整的数据处理功能
run_data_processing()
```

## 数据格式要求

系统支持以下数据格式：
- **CSV文件**: 包含时间序列数据
- **必需字段**: 
  - `Datetime`: 时间戳字段 (格式: DD-MM-YYYY HH:MM)
  - `Count`: 资金流量数值字段
- **可选字段**: 其他相关特征字段

## 输出文件说明

### 分析报告 (data_analysis.md)
- 数据概览和形状信息
- 每个字段的详细分析
- 数据类型、统计信息、示例值
- 前几行数据展示

### 基础可视化图表 (cash_flow_analysis.png)
- 2x2 子图布局
- 时间序列趋势分析
- 数据分布和统计特征
- 24小时模式热力图

### 申购赎回分析图表 (purchase_redemption_analysis.png)
- 2x2 子图布局
- 申购与赎回时间序列对比
- 申购与赎回分布对比
- 24小时申购赎回热力图
- 申购赎回统计信息面板

## 申购赎回分析逻辑

系统使用以下逻辑来区分申购和赎回：
1. **基于中位数阈值**: 将大于中位数的数值标记为申购，小于中位数的标记为赎回
2. **时间特征提取**: 自动提取小时、日期、月份、星期等时间特征
3. **可视化对比**: 通过不同颜色和图表类型展示申购与赎回的差异

注意：当前的申购赎回区分逻辑是基于数值大小的简单规则，在实际应用中可能需要根据具体的业务逻辑进行调整。 