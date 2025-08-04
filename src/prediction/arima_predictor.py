# -*- coding: utf-8 -*-
"""
ARIMA预测模块
使用ARIMA模型进行申购赎回趋势预测
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

from utils.data_processor import DataProcessor
from utils.config_utils import (
    get_arima_config, get_arima_training_range, get_arima_data_limits,
    get_arima_model_params, get_arima_prediction_config, get_arima_evaluation_config
)
from utils.visualization_utils import (
    setup_matplotlib, create_time_series_plot, save_plot, close_plot
)
from utils.interactive_utils import print_header, print_success, print_error, print_info
from utils.file_utils import write_json, write_csv


class ARIMAPredictor(DataProcessor):
    """ARIMA预测器类"""
    
    def __init__(self):
        """初始化ARIMA预测器"""
        super().__init__()
        self.config = get_arima_config()
        self.training_range = get_arima_training_range()
        self.data_limits = get_arima_data_limits()
        self.model_params = get_arima_model_params()
        self.prediction_config = get_arima_prediction_config()
        self.evaluation_config = get_arima_evaluation_config()
        
        self.model = None
        self.fitted_model = None
        self.predictions = None
        self.forecast_results = None
    
    def prepare_training_data(self, target_field='Net_Flow'):
        """
        准备训练数据
        
        Args:
            target_field: 目标字段，默认为净资金流
            
        Returns:
            pd.Series: 时间序列数据
        """
        if self.data is None:
            print_error("请先加载数据")
            return None
        
        # 确保数据已预处理
        if target_field not in self.data.columns:
            print("数据未预处理，正在自动预处理...")
            if not self.preprocess_data():
                print_error("数据预处理失败")
                return None
        
        # 获取时间字段
        from utils.config_utils import get_field_name
        time_field = get_field_name("时间字段")
        if time_field not in self.data.columns:
            print_error(f"时间字段 '{time_field}' 不存在")
            return None
        
        # 设置时间索引
        time_series = self.data.set_index(time_field)[target_field]
        
        # 按时间排序
        time_series = time_series.sort_index()
        
        # 重新获取最新的配置
        import importlib
        import config
        importlib.reload(config)
        from utils.config_utils import get_arima_training_range
        training_range = get_arima_training_range()
        
        # 处理训练时间范围
        print(f"配置的训练时间范围: {training_range['开始日期']} 到 {training_range['结束日期']}")
        
        # 严格按照配置的时间范围
        start_date = pd.to_datetime(training_range["开始日期"])
        end_date = pd.to_datetime(training_range["结束日期"])
        print(f"使用配置时间范围: {start_date} 到 {end_date}")
        
        # 过滤时间范围
        original_length = len(time_series)
        time_series = time_series[(time_series.index >= start_date) & 
                                 (time_series.index <= end_date)]
        filtered_length = len(time_series)
        
        print(f"时间范围过滤: {original_length} -> {filtered_length} 条数据")
        
        if filtered_length == 0:
            print_error("过滤后没有数据，请检查时间范围配置")
            return None
        
        # 检查数据量
        if len(time_series) < self.data_limits["最小数据量"]:
            print_error(f"数据量不足，需要至少 {self.data_limits['最小数据量']} 条数据")
            return None
        
        if len(time_series) > self.data_limits["最大数据量"]:
            print_info(f"数据量超过限制，将进行采样")
            if self.data_limits["数据采样"]["启用采样"]:
                sample_size = int(len(time_series) * self.data_limits["数据采样"]["采样比例"])
                time_series = time_series.sample(n=sample_size).sort_index()
        
        print_success(f"训练数据准备完成，共 {len(time_series)} 条数据")
        print(f"时间范围: {time_series.index.min()} 到 {time_series.index.max()}")
        
        return time_series
    
    def check_stationarity(self, time_series):
        """
        检查时间序列的平稳性
        
        Args:
            time_series: 时间序列数据
            
        Returns:
            bool: 是否平稳
        """
        print_header("平稳性检验")
        
        # ADF检验
        adf_result = adfuller(time_series.dropna())
        
        print(f"ADF统计量: {adf_result[0]:.4f}")
        print(f"p值: {adf_result[1]:.4f}")
        print(f"临界值:")
        for key, value in adf_result[4].items():
            print(f"  {key}: {value:.4f}")
        
        is_stationary = adf_result[1] < 0.05
        if is_stationary:
            print_success("时间序列是平稳的")
        else:
            print_info("时间序列不是平稳的，需要进行差分")
        
        return is_stationary
    
    def determine_differencing(self, time_series):
        """
        确定差分阶数
        
        Args:
            time_series: 时间序列数据
            
        Returns:
            int: 差分阶数
        """
        if self.model_params["差分阶数"]["自动检测"]:
            print_header("自动检测差分阶数")
            
            d = 0
            current_series = time_series.copy()
            
            while d <= self.model_params["差分阶数"]["最大差分"]:
                if self.check_stationarity(current_series):
                    break
                
                current_series = current_series.diff().dropna()
                d += 1
                print(f"进行 {d} 阶差分...")
            
            print_success(f"检测到最优差分阶数: {d}")
            return d
        else:
            return 1  # 默认1阶差分
    
    def find_best_arima_params(self, time_series, d):
        """
        寻找最优ARIMA参数
        
        Args:
            time_series: 时间序列数据
            d: 差分阶数
            
        Returns:
            tuple: (p, d, q) 最优参数
        """
        print_header("寻找最优ARIMA参数")
        
        p_range = self.model_params["ARIMA参数"]["p_range"]
        q_range = self.model_params["ARIMA参数"]["q_range"]
        
        best_aic = float('inf')
        best_params = (1, d, 1)
        
        total_combinations = len(p_range) * len(q_range)
        current_combination = 0
        
        print(f"测试 {total_combinations} 种参数组合...")
        
        for p in p_range:
            for q in q_range:
                current_combination += 1
                print(f"进度: {current_combination}/{total_combinations} - 测试 ARIMA({p},{d},{q})")
                
                try:
                    model = ARIMA(time_series, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        print(f"发现更好的参数: ARIMA{best_params}, AIC: {best_aic:.4f}")
                        
                except Exception as e:
                    print(f"ARIMA({p},{d},{q}) 拟合失败: {e}")
                    continue
        
        print_success(f"最优参数: ARIMA{best_params}, AIC: {best_aic:.4f}")
        return best_params
    
    def fit_arima_model(self, time_series, order):
        """
        拟合ARIMA模型
        
        Args:
            time_series: 时间序列数据
            order: ARIMA参数 (p, d, q)
            
        Returns:
            bool: 是否拟合成功
        """
        print_header("拟合ARIMA模型")
        
        try:
            self.model = ARIMA(time_series, order=order)
            self.fitted_model = self.model.fit()
            
            print_success("ARIMA模型拟合成功")
            print(f"模型参数: ARIMA{order}")
            print(f"AIC: {self.fitted_model.aic:.4f}")
            print(f"BIC: {self.fitted_model.bic:.4f}")
            
            return True
            
        except Exception as e:
            print_error(f"ARIMA模型拟合失败: {e}")
            return False
    
    def make_forecast(self, steps=None):
        """
        进行预测
        
        Args:
            steps: 预测步数，如果为None则使用配置中的步数
            
        Returns:
            bool: 是否预测成功
        """
        if self.fitted_model is None:
            print_error("请先拟合ARIMA模型")
            return False
        
        if steps is None:
            steps = self.prediction_config["预测步数"]
        
        print_header("ARIMA预测")
        print(f"预测步数: {steps}")
        
        try:
            # 进行预测
            forecast = self.fitted_model.forecast(steps=steps)
            
            # 获取置信区间
            if self.prediction_config["输出格式"]["包含置信区间"]:
                conf_int = self.fitted_model.get_forecast(steps=steps).conf_int(
                    alpha=1 - self.prediction_config["置信区间"]
                )
                
                self.forecast_results = {
                    "预测值": forecast,
                    "置信区间": conf_int,
                    "置信水平": self.prediction_config["置信区间"]
                }
            else:
                self.forecast_results = {
                    "预测值": forecast,
                    "置信区间": None,
                    "置信水平": None
                }
            
            print_success("预测完成")
            print(f"预测范围: {forecast.index[0]} 到 {forecast.index[-1]}")
            
            return True
            
        except Exception as e:
            print_error(f"预测失败: {e}")
            return False
    
    def visualize_results(self, save_plot=True):
        """
        可视化预测结果
        
        Args:
            save_plot: 是否保存图片
        """
        if self.forecast_results is None:
            print_error("请先进行预测")
            return False
        
        print_header("可视化预测结果")
        
        # 设置matplotlib
        setup_matplotlib()
        
        # 获取历史数据
        time_series = self.prepare_training_data()
        if time_series is None:
            return False
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制历史数据
        ax.plot(time_series.index, time_series.values, label='历史数据', color='blue')
        
        # 绘制预测值
        forecast = self.forecast_results["预测值"]
        ax.plot(forecast.index, forecast.values, label='预测值', color='red', linestyle='--')
        
        # 绘制置信区间
        if self.forecast_results["置信区间"] is not None:
            conf_int = self.forecast_results["置信区间"]
            ax.fill_between(forecast.index, 
                           conf_int.iloc[:, 0], 
                           conf_int.iloc[:, 1], 
                           alpha=0.3, color='red', label='置信区间')
        
        ax.set_title('ARIMA模型预测结果')
        ax.set_xlabel('时间')
        ax.set_ylabel('净资金流')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存图片
        if save_plot:
            save_plot(fig, "arima_prediction", "ARIMA预测结果")
        
        plt.show()
        close_plot(fig)
        
        return True
    
    def save_results(self, output_file=None):
        """
        保存预测结果
        
        Args:
            output_file: 输出文件路径
        """
        if self.forecast_results is None:
            print_error("请先进行预测")
            return False
        
        if output_file is None:
            from config import OUTPUT_DATA_DIR
            output_file = OUTPUT_DATA_DIR / "arima_prediction_results.csv"
        
        try:
            # 准备保存数据
            forecast = self.forecast_results["预测值"]
            results_df = pd.DataFrame({
                "日期": forecast.index,
                "预测值": forecast.values
            })
            
            # 添加置信区间
            if self.forecast_results["置信区间"] is not None:
                conf_int = self.forecast_results["置信区间"]
                results_df["置信下限"] = conf_int.iloc[:, 0].values
                results_df["置信上限"] = conf_int.iloc[:, 1].values
            
            # 保存到CSV
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            print_success(f"预测结果已保存: {output_file}")
            
            return True
            
        except Exception as e:
            print_error(f"保存预测结果失败: {e}")
            return False


def run_arima_prediction():
    """运行ARIMA预测功能"""
    print_header("ARIMA预测", "申购赎回趋势预测")
    
    # 创建ARIMA预测器
    predictor = ARIMAPredictor()
    
    # 加载数据
    if not predictor.load_data():
        return
    
    # 准备训练数据
    time_series = predictor.prepare_training_data()
    if time_series is None:
        return
    
    # 检查平稳性
    is_stationary = predictor.check_stationarity(time_series)
    
    # 确定差分阶数
    d = predictor.determine_differencing(time_series)
    
    # 寻找最优参数
    best_params = predictor.find_best_arima_params(time_series, d)
    
    # 拟合模型
    if not predictor.fit_arima_model(time_series, best_params):
        return
    
    # 进行预测
    if not predictor.make_forecast():
        return
    
    # 可视化结果
    predictor.visualize_results()
    
    # 保存结果
    predictor.save_results()
    
    print_success("ARIMA预测完成！") 