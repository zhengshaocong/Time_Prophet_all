# -*- coding: utf-8 -*-
"""
ARIMA可视化模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from utils.interactive_utils import print_info, print_success, print_error
from utils.visualization_utils import setup_matplotlib, save_plot, close_plot


class ARIMAVisualizer:
    """ARIMA可视化器"""
    
    def __init__(self):
        """初始化ARIMA可视化器"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_time_series_analysis(self, time_series, save_path=None):
        """绘制时间序列分析图"""
        try:
            setup_matplotlib()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('时间序列分析', fontsize=16)
            
            # 时间序列图
            axes[0, 0].plot(time_series.index, time_series.values)
            axes[0, 0].set_title('时间序列')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 自相关函数
            plot_acf(time_series.dropna(), ax=axes[0, 1], lags=40)
            axes[0, 1].set_title('自相关函数(ACF)')
            
            # 偏自相关函数
            plot_pacf(time_series.dropna(), ax=axes[1, 0], lags=40)
            axes[1, 0].set_title('偏自相关函数(PACF)')
            
            # 直方图
            axes[1, 1].hist(time_series.dropna(), bins=30, alpha=0.7)
            axes[1, 1].set_title('数据分布')
            
            plt.tight_layout()
            
            if save_path:
                save_plot(fig, save_path)
            else:
                plt.show()
            
            close_plot(fig)
            print_success("时间序列分析图绘制完成")
            return True
            
        except Exception as e:
            print_error(f"绘制时间序列分析图失败: {e}")
            return False
    
    def plot_predictions(self, actual_series, predictions, save_path=None):
        """绘制预测结果图"""
        try:
            setup_matplotlib()
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # 绘制实际值
            ax.plot(actual_series.index, actual_series.values, 
                   label='实际值', color='blue', linewidth=2)
            
            # 绘制预测值
            ax.plot(predictions.index, predictions.values, 
                   label='预测值', color='red', linewidth=2)
            
            ax.set_title('ARIMA预测结果', fontsize=14)
            ax.set_xlabel('时间')
            ax.set_ylabel('值')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                save_plot(fig, save_path)
            else:
                plt.show()
            
            close_plot(fig)
            print_success("预测结果图绘制完成")
            return True
            
        except Exception as e:
            print_error(f"绘制预测结果图失败: {e}")
            return False
    
    def plot_model_diagnostics(self, model, save_path=None):
        """绘制模型诊断图"""
        try:
            setup_matplotlib()
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('ARIMA模型诊断图', fontsize=16)
            
            # 残差图
            residuals = model.resid
            axes[0, 0].plot(residuals)
            axes[0, 0].set_title('残差时间序列')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 残差直方图
            axes[0, 1].hist(residuals, bins=30, alpha=0.7)
            axes[0, 1].set_title('残差分布')
            
            # 残差Q-Q图
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('残差Q-Q图')
            
            # 残差自相关图
            plot_acf(residuals, ax=axes[1, 1], lags=20)
            axes[1, 1].set_title('残差自相关函数')
            
            plt.tight_layout()
            
            if save_path:
                save_plot(fig, save_path)
            else:
                plt.show()
            
            close_plot(fig)
            print_success("模型诊断图绘制完成")
            return True
            
        except Exception as e:
            print_error(f"绘制模型诊断图失败: {e}")
            return False
    
    def autocorrelation_analysis(self, time_series, max_lag=40):
        """
        数值化自相关分析
        
        Args:
            time_series: 时间序列数据
            max_lag: 最大滞后阶数
            
        Returns:
            dict: 自相关分析结果
        """
        try:
            print_info("开始自相关分析...")
            
            # 计算自相关系数
            acf_values = []
            pacf_values = []
            
            for lag in range(1, max_lag + 1):
                acf = time_series.autocorr(lag=lag)
                acf_values.append(acf)
                
                # 计算偏自相关系数（简化版本）
                if lag == 1:
                    pacf_values.append(acf)
                else:
                    # 使用statsmodels计算PACF
                    from statsmodels.tsa.stattools import pacf
                    pacf_result = pacf(time_series.dropna(), nlags=lag)
                    pacf_values.append(pacf_result[-1])
            
            # Ljung-Box检验
            lb_test = acorr_ljungbox(time_series.dropna(), lags=max_lag, return_df=True)
            
            # 分析结果
            analysis_result = {
                'acf_values': acf_values,
                'pacf_values': pacf_values,
                'ljung_box_test': lb_test,
                'ljung_box_statistic': lb_test['lb_stat'].iloc[-1],
                'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[-1],
                'is_white_noise': lb_test['lb_pvalue'].iloc[-1] > 0.05
            }
            
            print_success("自相关分析完成")
            print(f"  Ljung-Box统计量: {analysis_result['ljung_box_statistic']:.4f}")
            print(f"  Ljung-Box p值: {analysis_result['ljung_box_pvalue']:.4f}")
            print(f"  是否为白噪声: {'是' if analysis_result['is_white_noise'] else '否'}")
            
            return analysis_result
            
        except Exception as e:
            print_error(f"自相关分析失败: {e}")
            return None
    
    def determine_arima_params_from_acf_pacf(self, acf_values, pacf_values, threshold=0.1):
        """
        基于ACF/PACF图自动确定ARIMA参数
        
        Args:
            acf_values: ACF值列表
            pacf_values: PACF值列表
            threshold: 截尾阈值
            
        Returns:
            tuple: (p, q) 参数
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
            
            print_success(f"基于ACF/PACF确定的参数: p={p}, q={q}")
            return p, q
            
        except Exception as e:
            print_error(f"确定ARIMA参数失败: {e}")
            return 1, 1  # 返回默认值
    
    def residual_autocorrelation_test(self, residuals, lags=10):
        """
        残差自相关检验
        
        Args:
            residuals: 残差序列
            lags: 滞后阶数
            
        Returns:
            dict: 检验结果
        """
        try:
            print_info("开始残差自相关检验...")
            
            # Ljung-Box检验
            lb_test = acorr_ljungbox(residuals, lags=lags, return_df=True)
            
            # 判断残差是否白噪声
            is_white_noise = lb_test['lb_pvalue'].iloc[-1] > 0.05
            
            test_result = {
                'ljung_box_statistic': lb_test['lb_stat'].iloc[-1],
                'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[-1],
                'is_white_noise': is_white_noise,
                'test_details': lb_test
            }
            
            print_success("残差自相关检验完成")
            print(f"  Ljung-Box统计量: {test_result['ljung_box_statistic']:.4f}")
            print(f"  Ljung-Box p值: {test_result['ljung_box_pvalue']:.4f}")
            print(f"  残差是否为白噪声: {'是' if test_result['is_white_noise'] else '否'}")
            
            return test_result
            
        except Exception as e:
            print_error(f"残差自相关检验失败: {e}")
            return None
    
    def comprehensive_autocorrelation_analysis(self, time_series, save_path=None):
        """
        综合自相关分析（可视化+数值分析）
        
        Args:
            time_series: 时间序列数据
            save_path: 保存路径
            
        Returns:
            dict: 分析结果
        """
        try:
            print_info("开始综合自相关分析...")
            
            # 1. 数值化分析
            analysis_result = self.autocorrelation_analysis(time_series)
            
            if analysis_result is None:
                return None
            
            # 2. 可视化分析
            self.plot_time_series_analysis(time_series, save_path)
            
            # 3. 基于ACF/PACF确定参数
            p, q = self.determine_arima_params_from_acf_pacf(
                analysis_result['acf_values'], 
                analysis_result['pacf_values']
            )
            
            # 4. 综合结果
            comprehensive_result = {
                'numerical_analysis': analysis_result,
                'suggested_p': p,
                'suggested_q': q,
                'analysis_summary': {
                    'is_stationary': analysis_result['is_white_noise'],
                    'has_autocorrelation': not analysis_result['is_white_noise'],
                    'suggested_model': f"ARIMA({p}, d, {q})"
                }
            }
            
            print_success("综合自相关分析完成")
            print(f"  建议的ARIMA参数: p={p}, q={q}")
            print(f"  序列特征: {'平稳' if analysis_result['is_white_noise'] else '非平稳'}")
            
            return comprehensive_result
            
        except Exception as e:
            print_error(f"综合自相关分析失败: {e}")
            return None 