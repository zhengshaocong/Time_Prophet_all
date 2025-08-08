# -*- coding: utf-8 -*-
"""
自相关分析工具模块
提供数值化的自相关分析功能
"""

import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import pacf
from utils.interactive_utils import print_info, print_success, print_error


class AutocorrelationAnalyzer:
    """自相关分析器"""
    
    def __init__(self):
        """初始化自相关分析器"""
        pass
    
    def calculate_acf_pacf(self, time_series, max_lag=40):
        """
        计算ACF和PACF值
        
        Args:
            time_series: 时间序列数据
            max_lag: 最大滞后阶数
            
        Returns:
            dict: ACF和PACF值
        """
        try:
            print_info("计算ACF和PACF值...")
            
            # 计算ACF
            acf_values = []
            for lag in range(1, max_lag + 1):
                acf = time_series.autocorr(lag=lag)
                acf_values.append(acf)
            
            # 计算PACF
            pacf_values = pacf(time_series.dropna(), nlags=max_lag)
            pacf_values = pacf_values[1:]  # 去掉lag=0的值
            
            result = {
                'acf_values': acf_values,
                'pacf_values': pacf_values,
                'lags': list(range(1, max_lag + 1))
            }
            
            print_success("ACF和PACF计算完成")
            return result
            
        except Exception as e:
            print_error(f"计算ACF和PACF失败: {e}")
            return None
    
    def ljung_box_test(self, time_series, lags=10):
        """
        Ljung-Box检验
        
        Args:
            time_series: 时间序列数据
            lags: 滞后阶数
            
        Returns:
            dict: 检验结果
        """
        try:
            print_info("进行Ljung-Box检验...")
            
            lb_test = acorr_ljungbox(time_series.dropna(), lags=lags, return_df=True)
            
            result = {
                'statistic': lb_test['lb_stat'].iloc[-1],
                'p_value': lb_test['lb_pvalue'].iloc[-1],
                'is_white_noise': lb_test['lb_pvalue'].iloc[-1] > 0.05,
                'test_details': lb_test
            }
            
            print_success("Ljung-Box检验完成")
            print(f"  统计量: {result['statistic']:.4f}")
            print(f"  p值: {result['p_value']:.4f}")
            print(f"  是否为白噪声: {'是' if result['is_white_noise'] else '否'}")
            
            return result
            
        except Exception as e:
            print_error(f"Ljung-Box检验失败: {e}")
            return None
    
    def determine_arima_params(self, acf_values, pacf_values, threshold=0.1):
        """
        基于ACF/PACF确定ARIMA参数
        
        Args:
            acf_values: ACF值列表
            pacf_values: PACF值列表
            threshold: 截尾阈值
            
        Returns:
            dict: 建议的参数
        """
        try:
            print_info("基于ACF/PACF确定ARIMA参数...")
            
            # 找到ACF截尾点（q值）
            q = 0
            for i, acf in enumerate(acf_values):
                if abs(acf) < threshold:
                    q = i
                    break
            
            # 找到PACF截尾点（p值）
            p = 0
            for i, pacf in enumerate(pacf_values):
                if abs(pacf) < threshold:
                    p = i
                    break
            
            # 如果都找不到截尾点，使用默认值
            if p == 0 and q == 0:
                p, q = 1, 1
            
            result = {
                'suggested_p': p,
                'suggested_q': q,
                'threshold': threshold,
                'reasoning': {
                    'acf_cutoff': q,
                    'pacf_cutoff': p
                }
            }
            
            print_success(f"建议的ARIMA参数: p={p}, q={q}")
            return result
            
        except Exception as e:
            print_error(f"确定ARIMA参数失败: {e}")
            return {'suggested_p': 1, 'suggested_q': 1}
    
    def comprehensive_analysis(self, time_series, max_lag=40):
        """
        综合自相关分析
        
        Args:
            time_series: 时间序列数据
            max_lag: 最大滞后阶数
            
        Returns:
            dict: 完整的分析结果
        """
        try:
            print_info("开始综合自相关分析...")
            
            # 1. 计算ACF和PACF
            acf_pacf_result = self.calculate_acf_pacf(time_series, max_lag)
            if acf_pacf_result is None:
                return None
            
            # 2. Ljung-Box检验
            lb_result = self.ljung_box_test(time_series, lags=min(10, max_lag))
            if lb_result is None:
                return None
            
            # 3. 确定ARIMA参数
            params_result = self.determine_arima_params(
                acf_pacf_result['acf_values'],
                acf_pacf_result['pacf_values']
            )
            
            # 4. 综合结果
            comprehensive_result = {
                'acf_pacf': acf_pacf_result,
                'ljung_box_test': lb_result,
                'arima_params': params_result,
                'summary': {
                    'is_stationary': lb_result['is_white_noise'],
                    'has_autocorrelation': not lb_result['is_white_noise'],
                    'suggested_model': f"ARIMA({params_result['suggested_p']}, d, {params_result['suggested_q']})",
                    'analysis_quality': 'good' if lb_result['is_white_noise'] else 'needs_improvement'
                }
            }
            
            print_success("综合自相关分析完成")
            return comprehensive_result
            
        except Exception as e:
            print_error(f"综合自相关分析失败: {e}")
            return None
    
    def residual_analysis(self, residuals, lags=10):
        """
        残差自相关分析
        
        Args:
            residuals: 残差序列
            lags: 滞后阶数
            
        Returns:
            dict: 残差分析结果
        """
        try:
            print_info("开始残差自相关分析...")
            
            # Ljung-Box检验
            lb_result = self.ljung_box_test(residuals, lags)
            
            # 残差统计
            residual_stats = {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'skewness': residuals.skew(),
                'kurtosis': residuals.kurtosis(),
                'is_normal': abs(residuals.skew()) < 1 and abs(residuals.kurtosis()) < 3
            }
            
            result = {
                'ljung_box_test': lb_result,
                'residual_stats': residual_stats,
                'quality_assessment': {
                    'is_white_noise': lb_result['is_white_noise'],
                    'is_normal': residual_stats['is_normal'],
                    'overall_quality': 'excellent' if lb_result['is_white_noise'] and residual_stats['is_normal'] else 'good' if lb_result['is_white_noise'] else 'needs_improvement'
                }
            }
            
            print_success("残差自相关分析完成")
            return result
            
        except Exception as e:
            print_error(f"残差自相关分析失败: {e}")
            return None


def quick_autocorrelation_check(time_series):
    """
    快速自相关检查
    
    Args:
        time_series: 时间序列数据
        
    Returns:
        dict: 快速检查结果
    """
    analyzer = AutocorrelationAnalyzer()
    return analyzer.comprehensive_analysis(time_series, max_lag=20)


def analyze_residuals(residuals):
    """
    分析残差
    
    Args:
        residuals: 残差序列
        
    Returns:
        dict: 残差分析结果
    """
    analyzer = AutocorrelationAnalyzer()
    return analyzer.residual_analysis(residuals)
