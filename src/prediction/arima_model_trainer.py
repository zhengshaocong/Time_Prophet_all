# -*- coding: utf-8 -*-
"""
ARIMA模型训练模块
"""

import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from utils.interactive_utils import print_info, print_success, print_error, print_warning


class ARIMAModelTrainer:
    """ARIMA模型训练器"""
    
    def __init__(self):
        """初始化ARIMA模型训练器"""
        self.best_model = None
        self.best_params = None
        self.best_aic = float('inf')
        self.training_results = []
        
    def find_optimal_parameters(self, time_series, max_p=3, max_d=2, max_q=3):
        """寻找最优ARIMA参数"""
        print_info("开始寻找最优ARIMA参数...")
        
        # 确定差分次数
        d = self._determine_difference_order(time_series)
        print(f"确定的差分次数: d = {d}")
        
        # 网格搜索最优p和q值
        best_params = self._grid_search_parameters(time_series, d, max_p, max_q)
        
        print_success(f"最优参数: p={best_params[0]}, d={best_params[1]}, q={best_params[2]}")
        return best_params
    
    def _determine_difference_order(self, time_series):
        """确定差分次数"""
        d = 0
        series = time_series.copy()
        
        # 检查原始序列的平稳性
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] <= 0.05
        
        if is_stationary:
            print("原始序列已平稳，无需差分")
            return d
        
        # 进行差分直到序列平稳
        while d < 2:
            d += 1
            series = series.diff().dropna()
            
            if len(series) < 10:
                break
                
            adf_result = adfuller(series.dropna())
            is_stationary = adf_result[1] <= 0.05
            
            if is_stationary:
                print(f"经过 {d} 次差分后序列平稳")
                break
        
        return d
    
    def _grid_search_parameters(self, time_series, d, max_p, max_q):
        """网格搜索最优p和q参数"""
        print_info("开始网格搜索参数...")
        
        p_values = range(0, max_p + 1)
        q_values = range(0, max_q + 1)
        
        best_params = None
        best_aic = float('inf')
        
        total_combinations = len(p_values) * len(q_values)
        current_combination = 0
        
        for p, q in itertools.product(p_values, q_values):
            current_combination += 1
            print(f"测试参数组合 {current_combination}/{total_combinations}: p={p}, d={d}, q={q}")
            
            try:
                model = ARIMA(time_series, order=(p, d, q))
                fitted_model = model.fit()
                
                aic = fitted_model.aic
                
                self.training_results.append({
                    'p': p, 'd': d, 'q': q,
                    'aic': aic,
                    'bic': fitted_model.bic,
                    'hqic': fitted_model.hqic
                })
                
                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, d, q)
                    print(f"  发现更优参数: AIC = {aic:.4f}")
                
            except Exception as e:
                print_warning(f"  参数组合 (p={p}, d={d}, q={q}) 训练失败: {e}")
                continue
        
        return best_params
    
    def train_model(self, time_series, params=None):
        """训练ARIMA模型"""
        print_info("开始训练ARIMA模型...")
        
        try:
            if params is None:
                params = self.find_optimal_parameters(time_series)
            
            p, d, q = params
            
            print(f"使用参数 p={p}, d={d}, q={q} 训练模型...")
            model = ARIMA(time_series, order=(p, d, q))
            fitted_model = model.fit()
            
            self.best_model = fitted_model
            self.best_params = params
            self.best_aic = fitted_model.aic
            
            print_success("ARIMA模型训练完成")
            print(f"模型AIC: {fitted_model.aic:.4f}")
            
            return True
            
        except Exception as e:
            print_error(f"模型训练失败: {e}")
            return False
    
    def get_model_summary(self):
        """获取模型摘要信息"""
        if self.best_model is None:
            return None
        
        return {
            'params': self.best_params,
            'aic': self.best_model.aic,
            'bic': self.best_model.bic,
            'hqic': self.best_model.hqic,
            'summary': str(self.best_model.summary())
        } 