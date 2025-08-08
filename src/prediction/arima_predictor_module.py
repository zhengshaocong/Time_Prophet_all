# -*- coding: utf-8 -*-
"""
ARIMA预测模块
包含预测功能和结果评估
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.interactive_utils import print_info, print_success, print_error, print_warning
from utils.visualization_utils import setup_matplotlib, create_time_series_plot, save_plot, close_plot


class ARIMAPredictor:
    """ARIMA预测器"""
    
    def __init__(self, model=None):
        """
        初始化ARIMA预测器
        
        Args:
            model: 训练好的ARIMA模型
        """
        self.model = model
        self.predictions = None
        self.forecast_results = None
        self.evaluation_metrics = {}
        
    def make_predictions(self, steps=30, dynamic=False):
        """
        进行预测
        
        Args:
            steps: 预测步数
            dynamic: 是否使用动态预测
            
        Returns:
            pd.Series: 预测结果
        """
        if self.model is None:
            print_error("没有训练好的模型")
            return None
        
        try:
            print_info(f"开始进行{steps}步预测...")
            
            # 进行预测
            if dynamic:
                forecast = self.model.forecast(steps=steps)
            else:
                forecast = self.model.get_forecast(steps=steps)
                forecast = forecast.predicted_mean
            
            self.predictions = forecast
            
            print_success(f"预测完成，预测步数: {steps}")
            return forecast
            
        except Exception as e:
            print_error(f"预测失败: {e}")
            return None
    
    def evaluate_predictions(self, actual_values):
        """
        评估预测结果
        
        Args:
            actual_values: 实际值
            
        Returns:
            dict: 评估指标
        """
        if self.predictions is None:
            print_error("没有预测结果可评估")
            return {}
        
        try:
            print_info("开始评估预测结果...")
            
            # 确保长度一致
            min_length = min(len(self.predictions), len(actual_values))
            pred = self.predictions[:min_length]
            actual = actual_values[:min_length]
            
            # 计算评估指标
            mse = mean_squared_error(actual, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, pred)
            mape = np.mean(np.abs((actual - pred) / actual)) * 100
            
            # 计算R²
            ss_res = np.sum((actual - pred) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            self.evaluation_metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2,
                '预测步数': min_length
            }
            
            print_success("预测评估完成")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"R²: {r2:.4f}")
            
            return self.evaluation_metrics
            
        except Exception as e:
            print_error(f"评估失败: {e}")
            return {}
    
    def plot_predictions(self, actual_values=None, save_path=None):
        """
        绘制预测结果图
        
        Args:
            actual_values: 实际值
            save_path: 保存路径
        """
        if self.predictions is None:
            print_error("没有预测结果可绘制")
            return False
        
        try:
            setup_matplotlib()
            
            # 创建预测结果图
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制预测值
            ax.plot(self.predictions.index, self.predictions.values, 
                   label='预测值', color='red', linewidth=2)
            
            # 如果有实际值，也绘制
            if actual_values is not None:
                ax.plot(actual_values.index, actual_values.values, 
                       label='实际值', color='blue', linewidth=2)
            
            ax.set_title('ARIMA预测结果', fontsize=14)
            ax.set_xlabel('时间')
            ax.set_ylabel('值')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                save_plot(fig, save_path)
            else:
                plt.show()
            
            close_plot(fig)
            return True
            
        except Exception as e:
            print_error(f"绘制预测图失败: {e}")
            return False
    
    def get_forecast_confidence_intervals(self, steps=30, alpha=0.05):
        """
        获取预测置信区间
        
        Args:
            steps: 预测步数
            alpha: 显著性水平
            
        Returns:
            tuple: (预测值, 置信区间下界, 置信区间上界)
        """
        if self.model is None:
            print_error("没有训练好的模型")
            return None, None, None
        
        try:
            print_info("计算预测置信区间...")
            
            # 获取预测和置信区间
            forecast_result = self.model.get_forecast(steps=steps)
            forecast_mean = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=alpha)
            
            lower_bound = conf_int.iloc[:, 0]
            upper_bound = conf_int.iloc[:, 1]
            
            print_success(f"置信区间计算完成，置信水平: {1-alpha}")
            
            return forecast_mean, lower_bound, upper_bound
            
        except Exception as e:
            print_error(f"计算置信区间失败: {e}")
            return None, None, None
    
    def save_predictions(self, file_path):
        """
        保存预测结果
        
        Args:
            file_path: 保存路径
        """
        if self.predictions is None:
            print_error("没有预测结果可保存")
            return False
        
        try:
            # 创建结果数据框
            results_df = pd.DataFrame({
                '预测值': self.predictions.values,
                '时间': self.predictions.index
            })
            
            # 保存到文件
            results_df.to_csv(file_path, index=False, encoding='utf-8')
            print_success(f"预测结果已保存: {file_path}")
            
            return True
            
        except Exception as e:
            print_error(f"保存预测结果失败: {e}")
            return False
    
    def get_prediction_summary(self):
        """
        获取预测摘要信息
        
        Returns:
            dict: 预测摘要
        """
        if self.predictions is None:
            return {}
        
        return {
            '预测步数': len(self.predictions),
            '预测开始时间': self.predictions.index[0] if len(self.predictions) > 0 else None,
            '预测结束时间': self.predictions.index[-1] if len(self.predictions) > 0 else None,
            '预测值范围': (self.predictions.min(), self.predictions.max()),
            '预测值均值': self.predictions.mean(),
            '预测值标准差': self.predictions.std(),
            '评估指标': self.evaluation_metrics
        } 