 # -*- coding: utf-8 -*-
"""
增强的ARIMA可视化模块
支持申购赎回数据对比和预测结果可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.interactive_utils import print_info, print_success, print_error
from utils.visualization_utils import setup_matplotlib, save_plot, close_plot


class ARIMAVisualizationEnhanced:
    """增强的ARIMA可视化器"""
    
    def __init__(self):
        """初始化增强的ARIMA可视化器"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_comprehensive_predictions(self, original_data, predictions, time_field='report_date', 
                                     save_path=None, title="ARIMA预测结果综合分析"):
        """绘制综合预测结果图"""
        try:
            setup_matplotlib()
            
            # 创建子图
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # 1. 净资金流预测图
            self._plot_net_flow_prediction(axes[0], original_data, predictions, time_field)
            
            # 2. 申购金额图
            self._plot_purchase_amount(axes[1], original_data, time_field)
            
            # 3. 赎回金额图
            self._plot_redemption_amount(axes[2], original_data, time_field)
            
            plt.tight_layout()
            
            if save_path:
                save_plot(fig, save_path)
                print_success(f"综合预测图已保存: {save_path}")
            else:
                plt.show()
            
            close_plot(fig)
            return True
            
        except Exception as e:
            print_error(f"绘制综合预测图失败: {e}")
            return False
    
    def _plot_net_flow_prediction(self, ax, original_data, predictions, time_field):
        """绘制净资金流预测图"""
        # 准备原始数据
        original_net_flow = original_data.groupby(time_field)['Net_Flow'].sum()
        
        # 绘制原始数据
        ax.plot(original_net_flow.index, original_net_flow.values, 
               label='实际净资金流', color='blue', linewidth=2, alpha=0.8)
        
        # 绘制预测数据（虚线）
        ax.plot(predictions.index, predictions.values, 
               label='预测净资金流', color='red', linewidth=2, linestyle='--')
        
        # 添加垂直线分隔实际和预测数据
        if len(original_net_flow) > 0:
            last_actual_date = original_net_flow.index[-1]
            ax.axvline(x=last_actual_date, color='gray', linestyle=':', alpha=0.7)
            ax.text(last_actual_date, ax.get_ylim()[1] * 0.9, '预测开始', 
                   rotation=90, verticalalignment='top')
        
        ax.set_title('净资金流预测', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间')
        ax.set_ylabel('净资金流')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_purchase_amount(self, ax, original_data, time_field):
        """绘制申购金额图"""
        # 准备申购数据
        purchase_data = original_data.groupby(time_field)['total_purchase_amt'].sum()
        
        # 绘制申购数据
        ax.plot(purchase_data.index, purchase_data.values, 
               label='申购金额', color='green', linewidth=2, alpha=0.8)
        
        ax.set_title('申购金额趋势', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间')
        ax.set_ylabel('申购金额')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_redemption_amount(self, ax, original_data, time_field):
        """绘制赎回金额图"""
        # 准备赎回数据
        redemption_data = original_data.groupby(time_field)['total_redeem_amt'].sum()
        
        # 绘制赎回数据
        ax.plot(redemption_data.index, redemption_data.values, 
               label='赎回金额', color='orange', linewidth=2, alpha=0.8)
        
        ax.set_title('赎回金额趋势', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间')
        ax.set_ylabel('赎回金额')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def plot_prediction_summary(self, original_data, predictions, time_field='report_date', 
                              save_path=None):
        """绘制预测结果摘要图"""
        try:
            setup_matplotlib()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ARIMA预测结果摘要', fontsize=16, fontweight='bold')
            
            # 1. 预测趋势图
            self._plot_prediction_trend(axes[0, 0], original_data, predictions, time_field)
            
            # 2. 预测值分布
            self._plot_prediction_distribution(axes[0, 1], predictions)
            
            # 3. 预测统计信息
            self._plot_prediction_stats(axes[1, 0], predictions)
            
            # 4. 预测时间范围
            self._plot_prediction_timeline(axes[1, 1], predictions)
            
            plt.tight_layout()
            
            if save_path:
                save_plot(fig, save_path)
                print_success(f"预测摘要图已保存: {save_path}")
            else:
                plt.show()
            
            close_plot(fig)
            return True
            
        except Exception as e:
            print_error(f"绘制预测摘要图失败: {e}")
            return False
    
    def _plot_prediction_trend(self, ax, original_data, predictions, time_field):
        """绘制预测趋势图"""
        # 准备原始数据
        original_net_flow = original_data.groupby(time_field)['Net_Flow'].sum()
        
        # 绘制原始数据
        ax.plot(original_net_flow.index, original_net_flow.values, 
               label='实际值', color='blue', linewidth=2)
        
        # 绘制预测数据
        ax.plot(predictions.index, predictions.values, 
               label='预测值', color='red', linewidth=2, linestyle='--')
        
        ax.set_title('预测趋势')
        ax.set_xlabel('时间')
        ax.set_ylabel('净资金流')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_distribution(self, ax, predictions):
        """绘制预测值分布"""
        ax.hist(predictions.values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('预测值分布')
        ax.set_xlabel('预测值')
        ax.set_ylabel('频数')
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_stats(self, ax, predictions):
        """绘制预测统计信息"""
        stats_text = f"""
预测统计信息:
• 预测天数: {len(predictions)}
• 预测开始: {predictions.index[0].strftime('%Y-%m-%d')}
• 预测结束: {predictions.index[-1].strftime('%Y-%m-%d')}
• 预测均值: {predictions.mean():.2f}
• 预测标准差: {predictions.std():.2f}
• 预测最小值: {predictions.min():.2f}
• 预测最大值: {predictions.max():.2f}
        """
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_title('预测统计信息')
        ax.axis('off')
    
    def _plot_prediction_timeline(self, ax, predictions):
        """绘制预测时间线"""
        dates = [d.strftime('%m-%d') for d in predictions.index]
        values = predictions.values
        
        ax.bar(range(len(dates)), values, alpha=0.7, color='lightcoral')
        ax.set_title('预测时间线')
        ax.set_xlabel('日期')
        ax.set_ylabel('预测值')
        
        # 设置x轴标签
        step = max(1, len(dates) // 10)  # 最多显示10个标签
        ax.set_xticks(range(0, len(dates), step))
        ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45)
        
        ax.grid(True, alpha=0.3)