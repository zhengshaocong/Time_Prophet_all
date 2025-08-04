# -*- coding: utf-8 -*-
"""
申购赎回分析模块
专注于申购与赎回的对比分析
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_processor import DataProcessor
from utils.visualization_utils import (
    setup_matplotlib, create_time_series_plot, create_histogram,
    save_plot, close_plot
)
from utils.interactive_utils import print_header, print_success, print_error
from utils.config_utils import get_field_name

class PurchaseRedemptionAnalysis(DataProcessor):
    """申购赎回分析类"""
    
    def __init__(self):
        """初始化申购赎回分析"""
        super().__init__()
    
    def analyze_purchase_redemption_trends(self, save_plot=True):
        """
        分析申购与赎回趋势
        
        Args:
            save_plot: 是否保存图片
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
            
        # 确保数据已预处理
        if 'Purchase_Amount' not in self.data.columns:
            if not self.preprocess_data():
                return False
        
        print_header("申购赎回趋势分析", "生成申购与赎回对比图")
        
        # 设置matplotlib
        setup_matplotlib()
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('申购与赎回趋势分析', fontsize=16, fontweight='bold')
        
        # 按日期聚合数据
        time_field = get_field_name("时间字段")
        current_balance_field = get_field_name("当前余额字段")
        
        daily_data = self.data.groupby(time_field).agg({
            'Purchase_Amount': 'sum',
            'Redemption_Amount': 'sum',
            'Net_Flow': 'sum',
            current_balance_field: 'mean'
        }).reset_index()
        
        # 1. 申购与赎回时间序列对比图
        create_time_series_plot(
            daily_data, time_field,
            ['Purchase_Amount', 'Redemption_Amount'],
            ['申购', '赎回'],
            ['green', 'red'],
            '申购与赎回时间序列对比',
            axes[0, 0]
        )
        
        # 2. 申购与赎回分布对比
        purchase_data = self.data[self.data['Purchase_Amount'] > 0]
        redemption_data = self.data[self.data['Redemption_Amount'] > 0]
        
        # 使用对数刻度处理大范围数据
        axes[0, 1].hist(purchase_data['Purchase_Amount'], bins=30, alpha=0.7, 
                       label='申购', color='green', edgecolor='black')
        axes[0, 1].hist(redemption_data['Redemption_Amount'], bins=30, alpha=0.7, 
                       label='赎回', color='red', edgecolor='black')
        axes[0, 1].set_title('申购与赎回分布对比')
        axes[0, 1].set_xlabel('金额')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        axes[0, 1].set_xscale('log')  # 使用对数刻度处理大范围数据
        
        # 3. 月度申购赎回趋势
        if 'Month' in self.data.columns:
            monthly_data = self.data.groupby('Month').agg({
                'Purchase_Amount': 'mean',
                'Redemption_Amount': 'mean'
            }).reset_index()
            
            x = range(len(monthly_data))
            width = 0.35
            
            axes[1, 0].bar([i - width/2 for i in x], monthly_data['Purchase_Amount'], 
                          width, label='申购金额', color='green', alpha=0.7)
            axes[1, 0].bar([i + width/2 for i in x], monthly_data['Redemption_Amount'], 
                          width, label='赎回金额', color='red', alpha=0.7)
            
            axes[1, 0].set_title('月度申购赎回趋势')
            axes[1, 0].set_xlabel('月份')
            axes[1, 0].set_ylabel('平均金额')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([f'{m}月' for m in monthly_data['Month']])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 申购赎回统计信息
        purchase_stats = purchase_data['Purchase_Amount'].describe()
        redemption_stats = redemption_data['Redemption_Amount'].describe()
        
        # 隐藏统计图，改为显示文本信息
        axes[1, 1].axis('off')
        stats_text = "申购赎回统计信息\n\n"
        stats_text += f"申购记录数: {len(purchase_data)}\n"
        stats_text += f"赎回记录数: {len(redemption_data)}\n"
        stats_text += f"申购平均值: {purchase_stats['mean']:.2f}\n"
        stats_text += f"赎回平均值: {redemption_stats['mean']:.2f}\n"
        stats_text += f"申购标准差: {purchase_stats['std']:.2f}\n"
        stats_text += f"赎回标准差: {redemption_stats['std']:.2f}\n"
        stats_text += f"申购最大值: {purchase_stats['max']:.2f}\n"
        stats_text += f"赎回最大值: {redemption_stats['max']:.2f}"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_plot:
            # 使用完整的函数名避免冲突
            from utils.visualization_utils import save_plot as save_plot_func
            save_plot_func(fig, "purchase_redemption_trends.png")
        
        # 关闭图形以释放内存
        close_plot(fig)
        return True
    
    def get_purchase_redemption_summary(self):
        """
        获取申购赎回统计摘要
        
        Returns:
            dict: 统计摘要
        """
        if self.data is None:
            return None
            
        purchase_data = self.data[self.data['Purchase_Amount'] > 0]
        redemption_data = self.data[self.data['Redemption_Amount'] > 0]
        
        summary = {
            "申购统计": {
                "记录数": len(purchase_data),
                "平均值": float(purchase_data['Purchase_Amount'].mean()),
                "中位数": float(purchase_data['Purchase_Amount'].median()),
                "标准差": float(purchase_data['Purchase_Amount'].std()),
                "最大值": float(purchase_data['Purchase_Amount'].max()),
                "最小值": float(purchase_data['Purchase_Amount'].min())
            },
            "赎回统计": {
                "记录数": len(redemption_data),
                "平均值": float(redemption_data['Redemption_Amount'].mean()),
                "中位数": float(redemption_data['Redemption_Amount'].median()),
                "标准差": float(redemption_data['Redemption_Amount'].std()),
                "最大值": float(redemption_data['Redemption_Amount'].max()),
                "最小值": float(redemption_data['Redemption_Amount'].min())
            },
            "对比分析": {
                "申购赎回比例": len(purchase_data) / len(redemption_data) if len(redemption_data) > 0 else float('inf'),
                "平均申购赎回比例": purchase_data['Purchase_Amount'].mean() / redemption_data['Redemption_Amount'].mean() if redemption_data['Redemption_Amount'].mean() > 0 else float('inf')
            }
        }
        
        return summary

def run_purchase_redemption_analysis():
    """
    运行申购赎回分析功能
    """
    print_header("申购赎回分析", "时间序列分析")
    
    # 创建申购赎回分析实例
    analysis = PurchaseRedemptionAnalysis()
    
    if analysis.load_data():
        # 预处理数据
        if analysis.preprocess_data():
            # 生成申购赎回可视化
            analysis.analyze_purchase_redemption_trends(save_plot=True)
            
            # 显示统计信息
            summary = analysis.get_purchase_redemption_summary()
            if summary:
                print("\n=== 申购赎回统计信息 ===")
                print(f"申购记录数: {summary['申购统计']['记录数']}")
                print(f"赎回记录数: {summary['赎回统计']['记录数']}")
                print(f"申购平均值: {summary['申购统计']['平均值']:.2f}")
                print(f"赎回平均值: {summary['赎回统计']['平均值']:.2f}")
                print(f"申购最大值: {summary['申购统计']['最大值']:.2f}")
                print(f"赎回最大值: {summary['赎回统计']['最大值']:.2f}")
                print(f"申购赎回比例: {summary['对比分析']['申购赎回比例']:.2f}")
            
            print_success("申购赎回分析完成")
        else:
            print_error("数据预处理失败")
    else:
        print_error("数据加载失败")

if __name__ == "__main__":
    run_purchase_redemption_analysis() 