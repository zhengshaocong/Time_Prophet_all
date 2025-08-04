# -*- coding: utf-8 -*-
"""
预测模块
包含资金流预测功能
"""

from .cash_flow_predictor import CashFlowPredictor, run_prediction_analysis

__all__ = [
    'CashFlowPredictor',
    'run_prediction_analysis'
] 