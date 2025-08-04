# -*- coding: utf-8 -*-
"""
资金流预测模块
使用多种模型进行资金流预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.data_processor import DataProcessor
from utils.visualization_utils import (
    setup_matplotlib, create_time_series_plot, save_plot, close_plot
)
from utils.interactive_utils import print_header, print_success, print_error, print_info
from config import DATA_DIR, OUTPUT_DIR, IMAGES_DIR
from utils.config_utils import (
    get_field_name, check_data_source_dispose_config,
    get_missing_dispose_config_message
)
from utils.data_processing_manager import get_data_for_module, should_process_data

class CashFlowPredictor(DataProcessor):
    """资金流预测器类"""
    
    def __init__(self):
        """初始化资金流预测器"""
        super().__init__()
        self.module_name = "cash_flow_predictor"
        self.predictions = None
        
    def prepare_data_for_prediction(self, data_source=None):
        """
        为预测准备数据
        
        Args:
            data_source: 数据源名称，如果为None则自动检测
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
        
        print_header("预测数据准备", "时间序列处理")
        
        try:
            # 获取字段名（会自动检测数据源）
            time_field = get_field_name("时间字段", data_source)
            
            if not time_field:
                print_error("缺少时间字段映射，请先运行基础数据分析")
                return False
            
            # 确保数据已预处理
            if 'Net_Flow' not in self.data.columns:
                print("数据未预处理，正在自动预处理...")
                if not self.preprocess_data():
                    print_error("数据预处理失败")
                    return False
            
            # 转换时间字段
            self.data[time_field] = pd.to_datetime(self.data[time_field])
            
            # 按时间排序
            self.data = self.data.sort_values(time_field)
            
            # 创建时间序列数据
            time_series = self.data.groupby(time_field)['Net_Flow'].sum().reset_index()
            time_series = time_series.set_index(time_field)
            
            # 处理缺失值
            time_series = time_series.fillna(0)
            
            print(f"时间序列长度: {len(time_series)}")
            print(f"时间范围: {time_series.index.min()} 到 {time_series.index.max()}")
            
            self.time_series = time_series['Net_Flow']
            
            return True
            
        except Exception as e:
            print_error(f"预测数据准备失败: {e}")
            return False
    
    def simple_moving_average_forecast(self, window=7, forecast_steps=30):
        """
        简单移动平均预测
        
        Args:
            window: 移动平均窗口大小
            forecast_steps: 预测步数
        """
        if self.time_series is None:
            print_error("请先准备预测数据")
            return False
        
        print_header("简单移动平均预测", f"窗口大小: {window}")
        
        try:
            # 计算移动平均
            ma = self.time_series.rolling(window=window).mean()
            
            # 使用最后一个移动平均值作为预测
            last_ma = ma.iloc[-1]
            
            # 生成预测
            last_date = self.time_series.index[-1]
            forecast_dates = pd.date_range(
                start=self.time_series.index.max() + timedelta(days=1),
                periods=forecast_steps,
                freq='D'
            )
            
            forecast_values = [last_ma] * forecast_steps
            forecast_series = pd.Series(forecast_values, index=forecast_dates)
            
            self.predictions = {
                "method": "简单移动平均",
                "window": window,
                "forecast": forecast_series,
                "ma_series": ma
            }
            
            print_success("简单移动平均预测完成")
            print(f"预测范围: {forecast_dates[0]} 到 {forecast_dates[-1]}")
            print(f"预测值: {last_ma:.2f}")
            
            return True
            
        except Exception as e:
            print_error(f"简单移动平均预测失败: {e}")
            return False
    
    def exponential_smoothing_forecast(self, alpha=0.3, forecast_steps=30):
        """
        指数平滑预测
        
        Args:
            alpha: 平滑参数
            forecast_steps: 预测步数
        """
        if self.time_series is None:
            print_error("请先准备预测数据")
            return False
        
        print_header("指数平滑预测", f"平滑参数: {alpha}")
        
        try:
            # 计算指数平滑
            smoothed = self.time_series.ewm(alpha=alpha).mean()
            
            # 使用最后一个平滑值作为预测
            last_smoothed = smoothed.iloc[-1]
            
            # 生成预测
            forecast_dates = pd.date_range(
                start=self.time_series.index.max() + timedelta(days=1),
                periods=forecast_steps,
                freq='D'
            )
            
            forecast_values = [last_smoothed] * forecast_steps
            forecast_series = pd.Series(forecast_values, index=forecast_dates)
            
            self.predictions = {
                "method": "指数平滑",
                "alpha": alpha,
                "forecast": forecast_series,
                "smoothed_series": smoothed
            }
            
            print_success("指数平滑预测完成")
            print(f"预测范围: {forecast_dates[0]} 到 {forecast_dates[-1]}")
            print(f"预测值: {last_smoothed:.2f}")
            
            return True
            
        except Exception as e:
            print_error(f"指数平滑预测失败: {e}")
            return False
    
    def visualize_prediction(self, save_plot=True):
        """
        可视化预测结果
        
        Args:
            save_plot: 是否保存图片
        """
        if self.predictions is None:
            print_error("请先进行预测")
            return False
        
        print_header("预测结果可视化")
        
        # 设置matplotlib
        setup_matplotlib()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制历史数据
        ax.plot(self.time_series.index, self.time_series.values, 
                label='历史数据', color='blue', alpha=0.7)
        
        # 绘制预测值
        forecast = self.predictions["forecast"]
        ax.plot(forecast.index, forecast.values, 
                label='预测值', color='red', linestyle='--', linewidth=2)
        
        # 绘制平滑/移动平均线
        if "ma_series" in self.predictions:
            ma_series = self.predictions["ma_series"]
            ax.plot(ma_series.index, ma_series.values, 
                    label='移动平均', color='green', alpha=0.6)
        elif "smoothed_series" in self.predictions:
            smoothed_series = self.predictions["smoothed_series"]
            ax.plot(smoothed_series.index, smoothed_series.values, 
                    label='指数平滑', color='green', alpha=0.6)
        
        ax.set_title(f'{self.predictions["method"]}预测结果')
        ax.set_xlabel('时间')
        ax.set_ylabel('净资金流')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存图片
        if save_plot:
            method_name = self.predictions["method"].replace(" ", "_")
            save_plot(fig, f"{method_name}_prediction", f"{self.predictions['method']}预测结果")
        
        plt.show()
        close_plot(fig)
        
        return True
    
    def save_prediction_results(self, output_file=None):
        """
        保存预测结果
        
        Args:
            output_file: 输出文件路径
        """
        if self.predictions is None:
            print_error("请先进行预测")
            return False
        
        if output_file is None:
            method_name = self.predictions["method"].replace(" ", "_")
            output_file = OUTPUT_DIR / f"{method_name}_results.csv"
        
        try:
            # 准备保存数据
            forecast = self.predictions["forecast"]
            results_df = pd.DataFrame({
                "日期": forecast.index,
                "预测值": forecast.values
            })
            
            # 保存到CSV
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            print_success(f"预测结果已保存: {output_file}")
            
            return True
            
        except Exception as e:
            print_error(f"保存预测结果失败: {e}")
            return False


def run_prediction_analysis():
    """运行资金流预测分析"""
    print_header("资金流预测", "多种预测方法")
    
    # 创建资金流预测器实例
    predictor = CashFlowPredictor()
    
    # 加载数据
    if not predictor.load_data():
        print_error("数据加载失败")
        return False
    
    # 准备预测数据
    if not predictor.prepare_data_for_prediction():
        print_error("预测数据准备失败")
        return False
    
    # 选择预测方法
    print("\n请选择预测方法:")
    print("1. 简单移动平均预测")
    print("2. 指数平滑预测")
    
    choice = input("请输入选择 (1-2): ").strip()
    
    if choice == "1":
        # 简单移动平均预测
        window = int(input("请输入移动平均窗口大小 (默认7): ") or "7")
        steps = int(input("请输入预测步数 (默认30): ") or "30")
        
        if predictor.simple_moving_average_forecast(window, steps):
            predictor.visualize_prediction()
            predictor.save_prediction_results()
            print_success("简单移动平均预测完成")
        else:
            print_error("简单移动平均预测失败")
            
    elif choice == "2":
        # 指数平滑预测
        alpha = float(input("请输入平滑参数 (0-1, 默认0.3): ") or "0.3")
        steps = int(input("请输入预测步数 (默认30): ") or "30")
        
        if predictor.exponential_smoothing_forecast(alpha, steps):
            predictor.visualize_prediction()
            predictor.save_prediction_results()
            print_success("指数平滑预测完成")
        else:
            print_error("指数平滑预测失败")
            
    else:
        print_error("无效的选择")
        return False
    
    return True 