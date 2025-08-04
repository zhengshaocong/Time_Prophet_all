# -*- coding: utf-8 -*-
"""
ARIMA预测模块
使用ARIMA模型进行时间序列预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

class ARIMAPredictor(DataProcessor):
    """ARIMA预测器类"""
    
    def __init__(self):
        """初始化ARIMA预测器"""
        super().__init__()
        self.module_name = "arima_predictor"
        self.model = None
        self.fitted_model = None
        self.predictions = None
        
    def prepare_data_for_arima(self, data_source=None):
        """
        为ARIMA模型准备数据
        
        Args:
            data_source: 数据源名称，如果为None则自动检测
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
        
        print_header("ARIMA数据准备", "时间序列处理")
        
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
            
            # 检查时间序列的平稳性
            print("检查时间序列平稳性...")
            adf_result = adfuller(time_series['Net_Flow'].dropna())
            print(f"ADF统计量: {adf_result[0]:.4f}")
            print(f"p值: {adf_result[1]:.4f}")
            
            if adf_result[1] > 0.05:
                print_info("时间序列非平稳，需要进行差分处理")
                # 进行一阶差分
                time_series['Net_Flow_diff'] = time_series['Net_Flow'].diff().dropna()
                
                # 再次检查平稳性
                adf_result_diff = adfuller(time_series['Net_Flow_diff'].dropna())
                print(f"差分后ADF统计量: {adf_result_diff[0]:.4f}")
                print(f"差分后p值: {adf_result_diff[1]:.4f}")
                
                if adf_result_diff[1] <= 0.05:
                    print_success("一阶差分后时间序列平稳")
                    self.time_series = time_series['Net_Flow_diff'].dropna()
                else:
                    print_info("一阶差分后仍非平稳，使用原始数据")
                    self.time_series = time_series['Net_Flow'].dropna()
            else:
                print_success("时间序列平稳")
                self.time_series = time_series['Net_Flow'].dropna()
            
            print(f"时间序列长度: {len(self.time_series)}")
            print(f"时间范围: {self.time_series.index.min()} 到 {self.time_series.index.max()}")
            
            return True
            
        except Exception as e:
            print_error(f"ARIMA数据准备失败: {e}")
            return False


def run_arima_prediction():
    """运行ARIMA预测分析"""
    print_header("ARIMA预测", "时间序列预测")
    
    # 创建ARIMA预测器实例
    predictor = ARIMAPredictor()
    
    # 加载数据
    if not predictor.load_data():
        print_error("数据加载失败")
        return False
    
    # 准备ARIMA数据
    if not predictor.prepare_data_for_arima():
        print_error("ARIMA数据准备失败")
        return False
    
    print_success("ARIMA预测功能正在开发中...")
    print_info("当前版本支持数据准备和平稳性检验")
    print_info("完整的ARIMA模型训练和预测功能将在后续版本中实现")
    
    return True 