# -*- coding: utf-8 -*-
"""
经典分解法可视化模块
Classical Decomposition Visualization Module

功能：
1. 基础数据信息图
2. 周期因子分析图
3. Base分析图
4. 预测结果图
5. 分解分析图
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from utils.interactive_utils import print_info, print_success, print_warning


class ClassicalDecompositionVisualization:
    """经典分解法可视化类"""
    
    def __init__(self, config):
        """
        初始化可视化器
        
        Args:
            config: 经典分解法配置
        """
        self.config = config
        self.output_dir = Path("output/classical_decomposition")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置图像参数
        self.dpi = config.get('可视化配置', {}).get('图像质量', {}).get('dpi', 300)
        self.image_format = config.get('可视化配置', {}).get('图像质量', {}).get('图像格式', 'png')
        self.image_size = config.get('可视化配置', {}).get('图像质量', {}).get('图像大小', (15, 10))
        
    def create_basic_info_plot(self, data, basic_info, save_plot=True):
        """
        创建基础数据信息图
        
        Args:
            data: 数据
            basic_info: 基础信息
            save_plot: 是否保存图片
            
        Returns:
            str: 图片保存路径
        """
        print_info("创建基础数据信息图...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.image_size)
            fig.suptitle('经典分解法 - 基础数据信息分析', fontsize=16, fontweight='bold')
            
            # 1. 时间序列图
            axes[0, 0].plot(data['date'], data['value'], linewidth=1, alpha=0.8)
            axes[0, 0].set_title('原始时间序列')
            axes[0, 0].set_xlabel('日期')
            axes[0, 0].set_ylabel('数值')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 数值分布直方图
            axes[0, 1].hist(data['value'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('数值分布直方图')
            axes[0, 1].set_xlabel('数值')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 基本统计信息
            stats_text = f"""
            总行数: {basic_info.get('总行数', 'N/A')}
            数值列数: {len(basic_info.get('数值列统计', {}))}
            日期范围: {basic_info.get('date_日期范围', {}).get('开始日期', 'N/A')} 至 {basic_info.get('date_日期范围', {}).get('结束日期', 'N/A')}
            总天数: {basic_info.get('date_日期范围', {}).get('总天数', 'N/A')}
            """
            axes[1, 0].text(0.1, 0.5, stats_text, transform=axes[1, 0].transAxes, 
                           fontsize=12, verticalalignment='center', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
            axes[1, 0].set_title('基本统计信息')
            axes[1, 0].axis('off')
            
            # 4. 缺失值信息
            if '数据质量报告' in basic_info:
                quality_report = basic_info['数据质量报告']
                missing_info = quality_report.get('缺失值', {})
                
                missing_data = missing_info.get('各列缺失值数量', {})
                if missing_data:
                    columns = list(missing_data.keys())
                    missing_counts = list(missing_data.values())
                    
                    axes[1, 1].bar(columns, missing_counts, alpha=0.7, color='orange')
                    axes[1, 1].set_title('各列缺失值数量')
                    axes[1, 1].set_xlabel('列名')
                    axes[1, 1].set_ylabel('缺失值数量')
                    axes[1, 1].tick_params(axis='x', rotation=45)
                    axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, '无缺失值信息', transform=axes[1, 1].transAxes, 
                               fontsize=12, ha='center', va='center')
                axes[1, 1].set_title('缺失值信息')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_plot:
                file_path = self.output_dir / f"基础数据信息图.{self.image_format}"
                plt.savefig(file_path, dpi=self.dpi, bbox_inches='tight')
                print_success(f"基础数据信息图已保存: {file_path}")
                return str(file_path)
            
            plt.show()
            return None
            
        except Exception as e:
            print_warning(f"创建基础数据信息图失败: {e}")
            return None
    
    def create_periodic_factors_plot(self, periodic_factors, save_plot=True):
        """
        创建周期因子分析图
        
        Args:
            periodic_factors: 周期因子字典
            save_plot: 是否保存图片
            
        Returns:
            str: 图片保存路径
        """
        print_info("创建周期因子分析图...")
        
        try:
            # 添加调试信息
            print_info(f"接收到的周期因子: {periodic_factors}")
            
            # 调整图表尺寸，减少空白区域
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle('经典分解法 - 周期因子分析', fontsize=16, fontweight='bold')
            
            # 1. 周期因子柱状图
            weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            
            # 获取实际的周期因子数据
            if 'weekday' in periodic_factors:
                weekday_factors = periodic_factors['weekday']
                factors = [weekday_factors.get(i, 1.0) for i in range(7)]
            else:
                # 如果没有weekday键，尝试直接访问
                factors = [periodic_factors.get(i, 1.0) for i in range(7)]
            
            bars = axes[0].bar(weekdays, factors, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].set_title('周度周期因子')
            axes[0].set_xlabel('星期')
            axes[0].set_ylabel('周期因子值')
            axes[0].grid(True, alpha=0.3)
            
            # 在柱状图上添加数值标签
            for bar, factor in zip(bars, factors):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{factor:.3f}', ha='center', va='bottom')
            
            # 2. 周期因子相对变化
            baseline = 1.0
            relative_change = [(f - baseline) / baseline * 100 for f in factors]
            
            axes[1].bar(weekdays, relative_change, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1].set_title('周期因子相对变化率 (%)')
            axes[1].set_xlabel('星期')
            axes[1].set_ylabel('相对变化率 (%)')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 添加数值标签
            for i, change in enumerate(relative_change):
                axes[1].text(i, change + (0.5 if change >= 0 else -0.5), 
                           f'{change:.1f}%', ha='center', va='bottom' if change >= 0 else 'top')
            
            plt.tight_layout()
            
            if save_plot:
                file_path = self.output_dir / f"周期因子分析图.{self.image_format}"
                plt.savefig(file_path, dpi=self.dpi, bbox_inches='tight')
                print_success(f"周期因子分析图已保存: {file_path}")
                return str(file_path)
            
            plt.show()
            return None
            
        except Exception as e:
            print_warning(f"创建周期因子分析图失败: {e}")
            return None
    
    def create_base_analysis_plot(self, data, base_values, save_plot=True):
        """
        创建Base分析图
        
        Args:
            data: 原始数据
            base_values: Base值序列
            save_plot: 是否保存图片
            
        Returns:
            str: 图片保存路径
        """
        print_info("创建Base分析图...")
        
        try:
            fig, axes = plt.subplots(2, 1, figsize=(self.image_size[0], self.image_size[1] * 1.5))
            fig.suptitle('经典分解法 - Base值分析', fontsize=16, fontweight='bold')
            
            # 1. 原始值与Base值对比
            axes[0].plot(data['date'], data['value'], label='原始值', linewidth=1, alpha=0.8)
            axes[0].plot(data['date'], base_values, label='Base值', linewidth=2, color='red', alpha=0.8)
            axes[0].set_title('原始值与Base值对比')
            axes[0].set_xlabel('日期')
            axes[0].set_ylabel('数值')
            axes[0].legend()
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # 2. Base值变化趋势
            # 计算移动平均
            window_size = self.config.get('Base计算配置', {}).get('平滑处理', {}).get('平滑窗口', 3)
            moving_avg = base_values.rolling(window=window_size, center=True, min_periods=1).mean()
            
            axes[1].plot(data['date'], base_values, label='Base值', linewidth=1, alpha=0.8)
            axes[1].plot(data['date'], moving_avg, label=f'{window_size}日移动平均', linewidth=2, color='green', alpha=0.8)
            axes[1].set_title('Base值变化趋势')
            axes[1].set_xlabel('日期')
            axes[1].set_ylabel('Base值')
            axes[1].legend()
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                file_path = self.output_dir / f"Base分析图.{self.image_format}"
                plt.savefig(file_path, dpi=self.dpi, bbox_inches='tight')
                print_success(f"Base分析图已保存: {file_path}")
                return str(file_path)
            
            plt.show()
            return None
            
        except Exception as e:
            print_warning(f"创建Base分析图失败: {e}")
            return None
    
    def create_prediction_plot(self, historical_data, prediction_result, save_plot=True):
        """
        创建预测结果图
        
        Args:
            historical_data: 历史数据
            prediction_result: 预测结果
            save_plot: 是否保存图片
            
        Returns:
            str: 图片保存路径
        """
        print_info("创建预测结果图...")
        
        try:
            # 检查是否有申购和赎回数据
            has_purchase_redeem = prediction_result.get('has_purchase_redeem', False)
            
            if has_purchase_redeem:
                # 如果有申购和赎回数据，创建子图
                fig, axes = plt.subplots(2, 1, figsize=(self.image_size[0], self.image_size[1] * 1.5))
                fig.suptitle('经典分解法 - 预测结果（含申购赎回）', fontsize=16, fontweight='bold')
                
                # 第一个子图：净流入预测
                ax1 = axes[0]
                ax1.plot(historical_data['date'], historical_data['value'], 
                         label='历史数据', linewidth=2, color='blue', alpha=0.8)
                
                if 'dates' in prediction_result and 'predictions' in prediction_result:
                    future_dates = prediction_result['dates']
                    predictions = prediction_result['predictions']
                    
                    ax1.plot(future_dates, predictions, 
                             label='净流入预测值', linewidth=2, color='red', linestyle='--', alpha=0.8)
                    
                    # 绘制置信区间
                    if 'confidence_intervals' in prediction_result:
                        confidence_intervals = prediction_result['confidence_intervals']
                        lower_bounds = [ci[0] for ci in confidence_intervals]
                        upper_bounds = [ci[1] for ci in confidence_intervals]
                        
                        ax1.fill_between(future_dates, lower_bounds, upper_bounds, 
                                       alpha=0.3, color='red', label='95%置信区间')
                
                ax1.set_title('净流入预测', fontsize=14, fontweight='bold')
                ax1.set_ylabel('数值')
                ax1.legend()
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # 第二个子图：申购和赎回预测
                ax2 = axes[1]
                
                if 'purchase_predictions' in prediction_result and 'redeem_predictions' in prediction_result:
                    purchase_predictions = prediction_result['purchase_predictions']
                    redeem_predictions = prediction_result['redeem_predictions']
                    
                    ax2.plot(future_dates, purchase_predictions, 
                             label='申购预测值', linewidth=2, color='green', linestyle='-', alpha=0.8)
                    ax2.plot(future_dates, redeem_predictions, 
                             label='赎回预测值', linewidth=2, color='orange', linestyle='-', alpha=0.8)
                
                ax2.set_title('申购和赎回预测', fontsize=14, fontweight='bold')
                ax2.set_xlabel('日期')
                ax2.set_ylabel('数值')
                ax2.legend()
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
            else:
                # 原来的单图显示
                fig, axes = plt.subplots(1, 1, figsize=self.image_size)
                axes = [axes]  # 转换为列表以便统一处理
                
                # 绘制历史数据
                axes[0].plot(historical_data['date'], historical_data['value'], 
                             label='历史数据', linewidth=2, color='blue', alpha=0.8)
                
                # 绘制预测数据
                if 'dates' in prediction_result and 'predictions' in prediction_result:
                    future_dates = prediction_result['dates']
                    predictions = prediction_result['predictions']
                    
                    axes[0].plot(future_dates, predictions, 
                                 label='预测值', linewidth=2, color='red', linestyle='--', alpha=0.8)
                    
                    # 绘制置信区间
                    if 'confidence_intervals' in prediction_result:
                        confidence_intervals = prediction_result['confidence_intervals']
                        lower_bounds = [ci[0] for ci in confidence_intervals]
                        upper_bounds = [ci[1] for ci in confidence_intervals]
                        
                        axes[0].fill_between(future_dates, lower_bounds, upper_bounds, 
                                           alpha=0.3, color='red', label='95%置信区间')
                
                axes[0].set_title('经典分解法 - 预测结果', fontsize=16, fontweight='bold')
                axes[0].set_xlabel('日期')
                axes[0].set_ylabel('数值')
                axes[0].legend()
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                file_path = self.output_dir / f"预测结果图.{self.image_format}"
                plt.savefig(file_path, dpi=self.dpi, bbox_inches='tight')
                print_success(f"预测结果图已保存: {file_path}")
                return str(file_path)
            
            plt.show()
            return None
            
        except Exception as e:
            print_warning(f"创建预测结果图失败: {e}")
            return None
    
    def create_decomposition_plot(self, decomposition_result, save_plot=True):
        """
        创建分解分析图
        
        Args:
            decomposition_result: 分解结果
            save_plot: 是否保存图片
            
        Returns:
            str: 图片保存路径
        """
        print_info("创建分解分析图...")
        
        try:
            fig, axes = plt.subplots(4, 1, figsize=(self.image_size[0], self.image_size[1] * 2))
            fig.suptitle('经典分解法 - 时间序列分解', fontsize=16, fontweight='bold')
            
            # 获取数据
            original = decomposition_result.get('original', pd.Series())
            trend = decomposition_result.get('trend', pd.Series())
            seasonal = decomposition_result.get('seasonal', pd.Series())
            residual = decomposition_result.get('residual', pd.Series())
            
            if len(original) == 0:
                print_warning("分解结果数据为空，无法创建分解图")
                return None
            
            # 创建日期索引
            dates = pd.date_range(start='2023-01-01', periods=len(original), freq='D')
            
            # 1. 原始时间序列
            axes[0].plot(dates, original, linewidth=1, alpha=0.8)
            axes[0].set_title('原始时间序列')
            axes[0].set_ylabel('数值')
            axes[0].grid(True, alpha=0.3)
            
            # 2. 趋势项
            axes[1].plot(dates, trend, linewidth=2, color='red', alpha=0.8)
            axes[1].set_title('趋势项 T(t)')
            axes[1].set_ylabel('趋势值')
            axes[1].grid(True, alpha=0.3)
            
            # 3. 季节性项
            axes[2].plot(dates, seasonal, linewidth=1, color='green', alpha=0.8)
            axes[2].set_title('季节性项 S(t)')
            axes[2].set_ylabel('季节性值')
            axes[2].grid(True, alpha=0.3)
            
            # 4. 残差项
            axes[3].plot(dates, residual, linewidth=1, color='orange', alpha=0.8)
            axes[3].set_title('残差项 R(t)')
            axes[3].set_xlabel('日期')
            axes[3].set_ylabel('残差值')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                file_path = self.output_dir / f"分解分析图.{self.image_format}"
                plt.savefig(file_path, dpi=self.dpi, bbox_inches='tight')
                print_success(f"分解分析图已保存: {file_path}")
                return str(file_path)
            
            plt.show()
            return None
            
        except Exception as e:
            print_warning(f"创建分解分析图失败: {e}")
            return None
    
    def create_purchase_redeem_prediction_plot(self, data, prediction_result, save_plot=True):
        """
        创建申购和赎回预测图
        
        Args:
            data: 历史数据
            prediction_result: 预测结果
            save_plot: 是否保存图片
            
        Returns:
            str: 图片保存路径
        """
        print_info("创建申购和赎回预测图...")
        
        try:
            if not prediction_result.get('has_purchase_redeem', False):
                print_warning("预测结果中不包含申购赎回数据")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('经典分解法 - 申购和赎回资金流预测', fontsize=16, fontweight='bold')
            
            # 获取历史数据
            purchase_field = 'total_purchase_amt'
            redeem_field = 'total_redeem_amt'
            date_field = 'report_date' if 'report_date' in data.columns else 'date'
            
            # 1. 历史申购和赎回趋势
            if purchase_field in data.columns and redeem_field in data.columns:
                axes[0, 0].plot(data[date_field], data[purchase_field], 
                               label='申购金额', color='blue', alpha=0.7, linewidth=1)
                axes[0, 0].plot(data[date_field], data[redeem_field], 
                               label='赎回金额', color='red', alpha=0.7, linewidth=1)
                axes[0, 0].set_title('历史申购和赎回趋势')
                axes[0, 0].set_xlabel('日期')
                axes[0, 0].set_ylabel('金额')
                axes[0, 0].legend()
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 申购预测结果
            if 'purchase_predictions' in prediction_result:
                future_dates = prediction_result['dates']
                purchase_predictions = prediction_result['purchase_predictions']
                
                axes[0, 1].plot(future_dates, purchase_predictions, 
                               label='申购预测', color='blue', linewidth=2, marker='o', markersize=4)
                axes[0, 1].set_title('申购金额预测')
                axes[0, 1].set_xlabel('日期')
                axes[0, 1].set_ylabel('申购金额')
                axes[0, 1].legend()
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 赎回预测结果
            if 'redeem_predictions' in prediction_result:
                redeem_predictions = prediction_result['redeem_predictions']
                
                axes[1, 0].plot(future_dates, redeem_predictions, 
                               label='赎回预测', color='red', linewidth=2, marker='s', markersize=4)
                axes[1, 0].set_title('赎回金额预测')
                axes[1, 0].set_xlabel('日期')
                axes[1, 0].set_ylabel('赎回金额')
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 净资金流预测
            if 'purchase_predictions' in prediction_result and 'redeem_predictions' in prediction_result:
                net_flow_predictions = [p - r for p, r in zip(purchase_predictions, redeem_predictions)]
                
                axes[1, 1].plot(future_dates, net_flow_predictions, 
                               label='净资金流预测', color='green', linewidth=2, marker='^', markersize=4)
                axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[1, 1].set_title('净资金流预测')
                axes[1, 1].set_xlabel('日期')
                axes[1, 1].set_ylabel('净资金流')
                axes[1, 1].legend()
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                file_path = self.output_dir / f"purchase_redeem_prediction.{self.image_format}"
                plt.savefig(file_path, dpi=self.dpi, bbox_inches='tight')
                print_success(f"申购赎回预测图已保存: {file_path}")
                return str(file_path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            print_warning(f"创建申购赎回预测图失败: {e}")
            return None
    
    def create_purchase_redeem_comparison_plot(self, data, prediction_result, save_plot=True):
        """
        创建申购和赎回对比分析图
        
        Args:
            data: 历史数据
            prediction_result: 预测结果
            save_plot: 是否保存图片
            
        Returns:
            str: 图片保存路径
        """
        print_info("创建申购和赎回对比分析图...")
        
        try:
            if not prediction_result.get('has_purchase_redeem', False):
                print_warning("预测结果中不包含申购赎回数据")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('经典分解法 - 申购和赎回对比分析', fontsize=16, fontweight='bold')
            
            # 获取数据
            purchase_field = 'total_purchase_amt'
            redeem_field = 'total_redeem_amt'
            date_field = 'report_date' if 'report_date' in data.columns else 'date'
            
            # 1. 申购赎回比例饼图
            if purchase_field in data.columns and redeem_field in data.columns:
                total_purchase = data[purchase_field].sum()
                total_redeem = data[redeem_field].sum()
                
                labels = ['申购', '赎回']
                sizes = [total_purchase, total_redeem]
                colors = ['lightblue', 'lightcoral']
                
                axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('历史申购赎回比例')
            
            # 2. 申购赎回趋势对比
            if purchase_field in data.columns and redeem_field in data.columns:
                axes[0, 1].plot(data[date_field], data[purchase_field], 
                               label='申购', color='blue', alpha=0.7)
                axes[0, 1].plot(data[date_field], data[redeem_field], 
                               label='赎回', color='red', alpha=0.7)
                axes[0, 1].set_title('申购赎回趋势对比')
                axes[0, 1].set_xlabel('日期')
                axes[0, 1].set_ylabel('金额')
                axes[0, 1].legend()
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 预测期间申购赎回对比
            if 'purchase_predictions' in prediction_result and 'redeem_predictions' in prediction_result:
                future_dates = prediction_result['dates']
                purchase_predictions = prediction_result['purchase_predictions']
                redeem_predictions = prediction_result['redeem_predictions']
                
                x = np.arange(len(future_dates))
                width = 0.35
                
                axes[1, 0].bar(x - width/2, purchase_predictions, width, label='申购预测', color='blue', alpha=0.7)
                axes[1, 0].bar(x + width/2, redeem_predictions, width, label='赎回预测', color='red', alpha=0.7)
                axes[1, 0].set_title('预测期间申购赎回对比')
                axes[1, 0].set_xlabel('预测天数')
                axes[1, 0].set_ylabel('金额')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 净资金流变化趋势
            if 'purchase_predictions' in prediction_result and 'redeem_predictions' in prediction_result:
                net_flow_predictions = [p - r for p, r in zip(purchase_predictions, redeem_predictions)]
                
                # 计算累计净资金流
                cumulative_net_flow = np.cumsum(net_flow_predictions)
                
                axes[1, 1].plot(future_dates, cumulative_net_flow, 
                               label='累计净资金流', color='green', linewidth=2, marker='o')
                axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[1, 1].set_title('累计净资金流变化趋势')
                axes[1, 1].set_xlabel('日期')
                axes[1, 1].set_ylabel('累计净资金流')
                axes[1, 1].legend()
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                file_path = self.output_dir / f"purchase_redeem_comparison.{self.image_format}"
                plt.savefig(file_path, dpi=self.dpi, bbox_inches='tight')
                print_success(f"申购赎回对比分析图已保存: {file_path}")
                return str(file_path)
            else:
                plt.show()
                return None
                
        except Exception as e:
            print_warning(f"创建申购赎回对比分析图失败: {e}")
            return None
    
    def create_all_plots(self, data, basic_info, periodic_factors, base_values, 
                         prediction_result=None, decomposition_result=None):
        """
        创建所有可视化图表
        
        Args:
            data: 数据
            basic_info: 基础信息
            periodic_factors: 周期因子
            base_values: Base值
            prediction_result: 预测结果
            decomposition_result: 分解结果
            
        Returns:
            list: 保存的图片路径列表
        """
        print_info("创建所有可视化图表...")
        
        saved_files = []
        
        # 检查是否启用各种图表保存
        save_config = self.config.get('可视化配置', {}).get('保存配置', {})
        
        if save_config.get('保存基础信息图', True):
            file_path = self.create_basic_info_plot(data, basic_info, save_plot=True)
            if file_path:
                saved_files.append(file_path)
        
        if save_config.get('保存周期因子图', True):
            file_path = self.create_periodic_factors_plot(periodic_factors, save_plot=True)
            if file_path:
                saved_files.append(file_path)
        
        if save_config.get('保存Base分析图', True):
            file_path = self.create_base_analysis_plot(data, base_values, save_plot=True)
            if file_path:
                saved_files.append(file_path)
        
        if prediction_result and save_config.get('保存预测结果图', True):
            file_path = self.create_prediction_plot(data, prediction_result, save_plot=True)
            if file_path:
                saved_files.append(file_path)
        
        if decomposition_result and save_config.get('保存分解分析图', True):
            file_path = self.create_decomposition_plot(decomposition_result, save_plot=True)
            if file_path:
                saved_files.append(file_path)
        
        print_success(f"所有图表创建完成，共保存{len(saved_files)}个文件")
        return saved_files
