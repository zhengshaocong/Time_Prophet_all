# -*- coding: utf-8 -*-
"""
经典分解法预测模块
Classical Decomposition Prediction Module

本模块实现了经典分解法时间序列预测的完整功能，包括：
1. 数据预处理
2. 周期因子计算
3. Base值计算
4. 预测计算
5. 可视化分析
6. 性能评估

主要类：
- ClassicalDecompositionPredictor: 主预测器
- ClassicalDecompositionCore: 核心计算模块
- ClassicalDecompositionDataProcessor: 数据预处理模块
- ClassicalDecompositionVisualization: 可视化模块
- ClassicalDecompositionEvaluator: 性能评估模块
"""

from .predictor import ClassicalDecompositionPredictor
from .core import ClassicalDecompositionCore
from .data_processor import ClassicalDecompositionDataProcessor
from .visualization import ClassicalDecompositionVisualization
from .evaluator import ClassicalDecompositionEvaluator

__all__ = [
    'ClassicalDecompositionPredictor',
    'ClassicalDecompositionCore',
    'ClassicalDecompositionDataProcessor',
    'ClassicalDecompositionVisualization',
    'ClassicalDecompositionEvaluator',
]
