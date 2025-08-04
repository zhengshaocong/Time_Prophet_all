# -*- coding: utf-8 -*-
"""
申购赎回分析模块
专注于申购和赎回数据的深度分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.data_processor import DataProcessor
from utils.visualization_utils import (
    setup_matplotlib, create_time_series_plot, create_histogram,
    create_balance_change_plot, create_monthly_comparison_plot,
    save_plot, close_plot
)
from utils.interactive_utils import print_header, print_success, print_error, print_info
from utils.file_utils import write_json
from config import DATA_DIR
from utils.config_utils import (
    get_field_name, check_data_source_dispose_config,
    get_missing_dispose_config_message
)
from utils.data_processing_manager import get_data_for_module, should_process_data

class PurchaseRedemptionAnalysis(DataProcessor):
    """申购赎回分析类"""
    
    def __init__(self):
        """初始化申购赎回分析"""
        super().__init__()
        self.module_name = "purchase_redemption_analysis"
    
    def analyze_purchase_redemption_patterns(self, data_source=None):
        """
        分析申购赎回模式
        
        Args:
            data_source: 数据源名称，如果为None则自动检测
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
        
        print_header("申购赎回模式分析", "深度分析")
        
        try:
            # 获取字段名（会自动检测数据源）
            time_field = get_field_name("时间字段", data_source)
            current_balance_field = get_field_name("当前余额字段", data_source)
            
            if not time_field or not current_balance_field:
                print_error("缺少必要的字段映射，请先运行基础数据分析")
                return False
            
            # 确保数据已预处理
            if 'Net_Flow' not in self.data.columns:
                print("数据未预处理，正在自动预处理...")
                if not self.preprocess_data():
                    print_error("数据预处理失败")
                    return False
            
            # 分析申购赎回模式
            print("正在分析申购赎回模式...")
            
            # 1. 总体统计
            total_purchase = self.data['Purchase_Amount'].sum()
            total_redemption = self.data['Redemption_Amount'].sum()
            net_flow = total_purchase - total_redemption
            
            print(f"总申购金额: {total_purchase:,.2f}")
            print(f"总赎回金额: {total_redemption:,.2f}")
            print(f"净资金流: {net_flow:,.2f}")
            print(f"申购赎回比例: {total_purchase/total_redemption:.2f}" if total_redemption > 0 else "无赎回记录")
            
            # 2. 时间模式分析
            self.data[time_field] = pd.to_datetime(self.data[time_field])
            
            # 按月份分析
            monthly_stats = self.data.groupby(self.data[time_field].dt.to_period('M')).agg({
                'Purchase_Amount': 'sum',
                'Redemption_Amount': 'sum',
                'Net_Flow': 'sum'
            }).reset_index()
            
            print("\n月度申购赎回统计:")
            print(monthly_stats.head(10))
            
            # 3. 用户行为分析
            if 'user_id' in self.data.columns:
                user_stats = self.data.groupby('user_id').agg({
                    'Purchase_Amount': ['sum', 'count', 'mean'],
                    'Redemption_Amount': ['sum', 'count', 'mean']
                }).round(2)
                
                print(f"\n用户行为统计 (前5个用户):")
                print(user_stats.head())
            
            print_success("申购赎回模式分析完成")
            return True
            
        except Exception as e:
            print_error(f"申购赎回模式分析失败: {e}")
            return False 