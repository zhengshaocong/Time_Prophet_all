# -*- coding: utf-8 -*-
"""
基础数据分析模块
专注于数据探索和基础可视化
"""

import matplotlib.pyplot as plt
import numpy as np
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
from utils.config_utils import get_field_name
from utils.data_processing_manager import get_data_for_module, should_process_data

class BasicDataAnalysis(DataProcessor):
    """基础数据分析类"""
    
    def __init__(self):
        """初始化基础数据分析"""
        super().__init__()
        self.module_name = "basic_analysis"
    
    def save_data_analysis(self, output_file=None):
        """
        保存数据分析结果到文件
        
        Args:
            output_file: 输出文件路径
        """
        if not self.data_info:
            print_error("请先进行数据分析")
            return False
            
        if output_file is None:
            output_file = DATA_DIR / "data_analysis.md"
            
        try:
            # 创建Markdown格式的分析报告
            md_content = f"""# 资金流数据字段分析报告

## 数据概览
- **数据形状**: {self.data_info['数据形状'][0]} 行 × {self.data_info['数据形状'][1]} 列
- **分析时间**: {self.data_info['分析时间']}

## 字段详细信息

"""
            
            # 添加字段信息
            for field, info in self.data_info['字段信息'].items():
                md_content += f"### {field}\n"
                for key, value in info.items():
                    md_content += f"- **{key}**: {value}\n"
                md_content += "\n"
            
            # 添加前几行数据
            md_content += "## 前几行数据\n"
            md_content += "```\n"
            for record in self.data_info['前几行数据']:
                md_content += str(record) + "\n"
            md_content += "```\n"
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            print_success(f"分析报告已保存: {output_file}")
            return True
            
        except Exception as e:
            print_error(f"保存分析报告失败: {e}")
            return False
    
    def visualize_data(self, save_plot=True):
        """
        基础数据可视化
        
        Args:
            save_plot: 是否保存图片
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
            
        print_header("基础数据可视化", "生成图表")
        
        # 确保数据已预处理
        if 'Net_Flow' not in self.data.columns:
            print("数据未预处理，正在自动预处理...")
            if not self.preprocess_data():
                print_error("数据预处理失败，无法生成可视化")
                return False
        
        # 设置matplotlib
        setup_matplotlib()
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('资金流数据可视化分析', fontsize=16, fontweight='bold')
        
        # 1. 时间序列图
        time_field = get_field_name("时间字段")
        if time_field in self.data.columns and 'Net_Flow' in self.data.columns:
            try:
                # 按日期聚合数据
                daily_data = self.data.groupby(time_field).agg({
                    'Net_Flow': 'sum',
                    'Purchase_Amount': 'sum',
                    'Redemption_Amount': 'sum'
                }).reset_index()
                
                # 使用工具函数创建时间序列图
                create_time_series_plot(
                    daily_data, time_field,
                    ['Net_Flow', 'Purchase_Amount', 'Redemption_Amount'],
                    ['净资金流', '申购金额', '赎回金额'],
                    ['blue', 'green', 'red'],
                    '资金流时间序列',
                    axes[0, 0]
                )
            except Exception as e:
                print(f"时间序列图生成失败: {e}")
        
        # 2. 资金流分布直方图
        if 'Net_Flow' in self.data.columns:
            create_histogram(
                self.data, 'Net_Flow', bins=30, 
                color='skyblue', title='净资金流分布',
                ax=axes[0, 1]
            )
        
        # 3. 余额变化分布图
        if 'Balance_Change' in self.data.columns:
            create_balance_change_plot(self.data, ax=axes[1, 0])
        
        # 4. 月度资金流分布图
        if 'Month' in self.data.columns and 'Net_Flow' in self.data.columns:
            create_monthly_comparison_plot(self.data, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_plot:
            # 使用完整的函数名避免冲突
            from utils.visualization_utils import save_plot as save_plot_func
            save_plot_func(fig, "cash_flow_analysis.png")
        
        # 关闭图形以释放内存
        close_plot(fig)
        return True
    
    def auto_detect_field_mapping(self):
        """
        自动检测数据字段映射
        
        Returns:
            dict: 字段映射字典
        """
        if self.data is None:
            print_error("请先加载数据")
            return None
        
        print("正在自动检测字段映射...")
        
        # 获取所有列名
        columns = list(self.data.columns)
        field_mapping = {}
        
        # 检测时间字段
        time_candidates = ['report_date', 'date', 'datetime', 'time', 'timestamp']
        time_field = None
        for candidate in time_candidates:
            if candidate in columns:
                time_field = candidate
                break
        
        if time_field:
            field_mapping["时间字段"] = time_field
            # 尝试检测时间格式
            try:
                sample_value = str(self.data[time_field].iloc[0])
                if len(sample_value) == 8 and sample_value.isdigit():
                    field_mapping["时间格式"] = "%Y%m%d"
                elif '-' in sample_value and ':' in sample_value:
                    field_mapping["时间格式"] = "%d-%m-%Y %H:%M"
                else:
                    field_mapping["时间格式"] = "%Y%m%d"  # 默认格式
            except:
                field_mapping["时间格式"] = "%Y%m%d"
        
        # 检测用户ID字段
        user_id_candidates = ['user_id', 'userid', 'id', 'user']
        for candidate in user_id_candidates:
            if candidate in columns:
                field_mapping["用户ID字段"] = candidate
                break
        
        # 检测余额相关字段
        balance_candidates = ['tbalance', 't_balance', 'balance', 'current_balance', 'tBalance']
        for candidate in balance_candidates:
            if candidate in columns:
                field_mapping["当前余额字段"] = candidate
                break
        
        ybalance_candidates = ['ybalance', 'y_balance', 'previous_balance', 'yesterday_balance', 'yBalance']
        for candidate in ybalance_candidates:
            if candidate in columns:
                field_mapping["昨日余额字段"] = candidate
                break
        
        # 检测申购赎回字段
        purchase_candidates = ['total_purchase_amt', 'purchase_amt', 'purchase_amount', 'buy_amount']
        for candidate in purchase_candidates:
            if candidate in columns:
                field_mapping["申购金额字段"] = candidate
                break
        
        redeem_candidates = ['total_redeem_amt', 'redeem_amt', 'redemption_amount', 'sell_amount']
        for candidate in redeem_candidates:
            if candidate in columns:
                field_mapping["赎回金额字段"] = candidate
                break
        
        # 检测其他字段
        consume_candidates = ['consume_amt', 'consume_amount', 'consumption']
        for candidate in consume_candidates:
            if candidate in columns:
                field_mapping["消费金额字段"] = candidate
                break
        
        transfer_candidates = ['transfer_amt', 'transfer_amount', 'transfer']
        for candidate in transfer_candidates:
            if candidate in columns:
                field_mapping["转账金额字段"] = candidate
                break
        
        # 检测分类字段
        category_fields = [col for col in columns if 'category' in col.lower()]
        if category_fields:
            field_mapping["分类字段"] = category_fields
        
        # 打印检测结果
        print("字段映射检测结果:")
        for field_type, field_name in field_mapping.items():
            print(f"  {field_type}: {field_name}")
        
        return field_mapping

def run_basic_data_analysis():
    """
    运行基础数据分析功能
    """
    print_header("资金流预测系统", "基础数据分析")
    
    # 创建基础数据分析实例
    analysis = BasicDataAnalysis()
    
    # 检查数据处理配置
    if should_process_data(analysis.module_name):
        print_info("检测到数据处理配置，将使用处理后的数据进行分析")
    else:
        print_info("使用原始数据进行基础分析")
    
    # 1.1 读取数据的前五行并解析字段
    print("\n=== 步骤 1.1: 数据加载和字段解析 ===")
    if analysis.load_data(use_data_processing=True, module_name=analysis.module_name):
        analysis.analyze_data_structure(top_rows=5)
        analysis.save_data_analysis()
        
        # 1.1.1 自动检测字段映射并生成缓存
        print("\n=== 步骤 1.1.1: 字段映射自动检测 ===")
        field_mapping = analysis.auto_detect_field_mapping()
        if field_mapping:
            from utils.cache_utils import save_field_mapping_cache
            if save_field_mapping_cache(field_mapping):
                print_success("字段映射缓存生成成功")
            else:
                print_error("字段映射缓存生成失败")
    
    # 1.2 读取数据并展示成图片
    print("\n=== 步骤 1.2: 基础数据可视化 ===")
    analysis.visualize_data(save_plot=True)
    
    # 显示数据摘要
    print("\n=== 数据摘要 ===")
    summary = analysis.get_data_summary()
    if summary:
        print(f"数据形状: {summary['数据形状']}")
        print(f"字段数量: {len(summary['字段列表'])}")
        print(f"字段列表: {', '.join(summary['字段列表'])}")
    
    print_success("基础数据分析完成")

if __name__ == "__main__":
    run_basic_data_analysis() 