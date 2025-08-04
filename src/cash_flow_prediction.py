# -*- coding: utf-8 -*-
"""
资金流预测主模块
包含基础数据处理、数据可视化等功能
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np

from config import (
    DATA_DIR, IMAGES_DIR, OUTPUT_DATA_DIR,
    get_data_field_mapping, get_field_name, get_time_format, 
    get_preprocessing_config, CURRENT_DATA_SOURCE
)
from utils.interactive_utils import print_header, print_success, print_error
from utils.file_utils import write_json


class CashFlowPrediction:
    """资金流预测主类"""
    
    def __init__(self):
        """初始化资金流预测系统"""
        self.data = None
        self.data_info = {}
        
    def load_data(self, file_path=None):
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径，默认为data/user_balance_table.csv
        """
        if file_path is None:
            file_path = DATA_DIR / "user_balance_table.csv"
            
        try:
            self.data = pd.read_csv(file_path)
            print_success(f"数据加载成功: {file_path}")
            print(f"数据形状: {self.data.shape}")
            return True
        except Exception as e:
            print_error(f"数据加载失败: {e}")
            return False
    
    def analyze_data_structure(self, top_rows=5):
        """
        分析数据结构并解析字段
        
        Args:
            top_rows: 显示前几行数据
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
            
        print_header("数据结构分析", "字段解析")
        
        # 获取前几行数据
        top_data = self.data.head(top_rows)
        
        # 分析字段信息
        field_info = {}
        for column in self.data.columns:
            field_info[column] = {
                "数据类型": str(self.data[column].dtype),
                "非空值数量": int(self.data[column].count()),
                "空值数量": int(self.data[column].isnull().sum()),
                "唯一值数量": int(self.data[column].nunique()),
                "示例值": str(self.data[column].iloc[0]) if len(self.data) > 0 else "无"
            }
            
            # 对于数值型数据，添加统计信息
            if self.data[column].dtype in ['int64', 'float64']:
                field_info[column].update({
                    "最小值": float(self.data[column].min()),
                    "最大值": float(self.data[column].max()),
                    "平均值": float(self.data[column].mean()),
                    "中位数": float(self.data[column].median())
                })
        
        # 保存分析结果
        self.data_info = {
            "数据形状": list(self.data.shape),
            "字段信息": field_info,
            "前几行数据": top_data.to_dict('records'),
            "分析时间": datetime.now().isoformat()
        }
        
        # 输出分析结果
        print(f"数据形状: {self.data.shape[0]} 行 × {self.data.shape[1]} 列")
        print("\n字段信息:")
        for field, info in field_info.items():
            print(f"  {field}:")
            for key, value in info.items():
                print(f"    {key}: {value}")
        
        print(f"\n前{top_rows}行数据:")
        print(top_data.to_string(index=False))
        
        return True
    
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
            
            for field, info in self.data_info['字段信息'].items():
                md_content += f"### {field}\n"
                for key, value in info.items():
                    md_content += f"- **{key}**: {value}\n"
                md_content += "\n"
            
            md_content += "## 数据示例\n\n"
            md_content += "```\n"
            if self.data_info['前几行数据']:
                # 转换为DataFrame以便格式化显示
                df_sample = pd.DataFrame(self.data_info['前几行数据'])
                md_content += df_sample.to_string(index=False)
            md_content += "\n```\n"
            
            # 保存Markdown文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
                
            print_success(f"数据分析报告已保存: {output_file}")
            return True
            
        except Exception as e:
            print_error(f"保存分析报告失败: {e}")
            return False
    
    def preprocess_data_for_purchase_redemption(self, data_source=None):
        """
        预处理数据以区分申购和赎回
        基于配置文件中的字段映射，支持不同数据源
        
        Args:
            data_source: 数据源名称，如果为None则使用当前配置的数据源
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
            
        print_header("数据预处理", "区分申购和赎回")
        
        try:
            # 获取字段映射配置
            field_mapping = get_data_field_mapping(data_source)
            preprocessing_config = get_preprocessing_config()
            
            # 获取字段名
            time_field = field_mapping["时间字段"]
            time_format = field_mapping["时间格式"]
            purchase_field = field_mapping["申购金额字段"]
            redemption_field = field_mapping["赎回金额字段"]
            current_balance_field = field_mapping["当前余额字段"]
            previous_balance_field = field_mapping["昨日余额字段"]
            
            # 转换时间格式
            self.data[time_field] = pd.to_datetime(self.data[time_field], format=time_format)
            
            # 处理缺失值
            missing_value_config = preprocessing_config["缺失值处理"]
            self.data[purchase_field] = self.data[purchase_field].fillna(missing_value_config["申购金额字段"])
            self.data[redemption_field] = self.data[redemption_field].fillna(missing_value_config["赎回金额字段"])
            
            # 创建标准化的字段名
            self.data['Purchase_Amount'] = self.data[purchase_field]
            self.data['Redemption_Amount'] = self.data[redemption_field]
            
            # 计算净资金流
            self.data['Net_Flow'] = self.data['Purchase_Amount'] - self.data['Redemption_Amount']
            
            # 添加时间特征
            time_features = preprocessing_config["时间特征"]
            if time_features["提取年份"]:
                self.data['Year'] = self.data[time_field].dt.year
            if time_features["提取月份"]:
                self.data['Month'] = self.data[time_field].dt.month
            if time_features["提取日期"]:
                self.data['Day'] = self.data[time_field].dt.day
            if time_features["提取星期"]:
                self.data['Weekday'] = self.data[time_field].dt.dayofweek
            if time_features["提取季度"]:
                self.data['Quarter'] = self.data[time_field].dt.quarter
            if time_features["提取小时"]:
                self.data['Hour'] = self.data[time_field].dt.hour
            
            # 计算余额变化
            self.data['Balance_Change'] = self.data[current_balance_field] - self.data[previous_balance_field]
            
            # 异常值处理
            if preprocessing_config["异常值处理"]["启用异常值检测"]:
                threshold = preprocessing_config["异常值处理"]["异常值阈值"]
                for field in ['Purchase_Amount', 'Redemption_Amount', 'Net_Flow']:
                    if field in self.data.columns:
                        mean_val = self.data[field].mean()
                        std_val = self.data[field].std()
                        lower_bound = mean_val - threshold * std_val
                        upper_bound = mean_val + threshold * std_val
                        
                        if preprocessing_config["异常值处理"]["异常值处理方式"] == "clip":
                            self.data[field] = self.data[field].clip(lower_bound, upper_bound)
            
            print_success("数据预处理完成")
            print(f"数据源: {data_source or CURRENT_DATA_SOURCE}")
            print(f"总记录数: {len(self.data)}")
            print(f"有申购记录数: {len(self.data[self.data['Purchase_Amount'] > 0])}")
            print(f"有赎回记录数: {len(self.data[self.data['Redemption_Amount'] > 0])}")
            print(f"数据时间范围: {self.data[time_field].min()} 到 {self.data[time_field].max()}")
            
            return True
            
        except Exception as e:
            print_error(f"数据预处理失败: {e}")
            return False
    
    def visualize_data(self, save_plot=True):
        """
        数据可视化
        
        Args:
            save_plot: 是否保存图片
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
            
        print_header("数据可视化", "生成图表")
        
        # 确保数据已预处理
        if 'Net_Flow' not in self.data.columns:
            print("数据未预处理，正在自动预处理...")
            if not self.preprocess_data_for_purchase_redemption():
                print_error("数据预处理失败，无法生成可视化")
                return False
        
        # 设置中文字体 - 修复字体设置问题
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"字体设置警告: {e}")
            # 使用默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('资金流数据可视化分析', fontsize=16, fontweight='bold')
        
        # 1. 时间序列图 - 使用配置中的字段名
        time_field = get_field_name("时间字段")
        if time_field in self.data.columns and 'Net_Flow' in self.data.columns:
            try:
                # 按日期聚合数据
                daily_data = self.data.groupby(time_field).agg({
                    'Net_Flow': 'sum',
                    'Purchase_Amount': 'sum',
                    'Redemption_Amount': 'sum'
                }).reset_index()
                
                axes[0, 0].plot(daily_data[time_field], daily_data['Net_Flow'], 
                               linewidth=1, alpha=0.7, color='blue', label='净资金流')
                axes[0, 0].plot(daily_data[time_field], daily_data['Purchase_Amount'], 
                               linewidth=1, alpha=0.7, color='green', label='申购金额')
                axes[0, 0].plot(daily_data[time_field], daily_data['Redemption_Amount'], 
                               linewidth=1, alpha=0.7, color='red', label='赎回金额')
                axes[0, 0].set_title('资金流时间序列')
                axes[0, 0].set_xlabel('时间')
                axes[0, 0].set_ylabel('金额')
                axes[0, 0].legend()
                axes[0, 0].tick_params(axis='x', rotation=45)
            except Exception as e:
                print(f"时间序列图生成失败: {e}")
        
        # 2. 资金流分布直方图
        if 'Net_Flow' in self.data.columns:
            axes[0, 1].hist(self.data['Net_Flow'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('净资金流分布')
            axes[0, 1].set_xlabel('净资金流')
            axes[0, 1].set_ylabel('频次')
        
        # 3. 余额变化分布图（处理异常值）
        if 'Balance_Change' in self.data.columns:
            balance_change = self.data['Balance_Change']
            
            # 计算分位数来识别异常值
            q1 = balance_change.quantile(0.25)
            q3 = balance_change.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 过滤异常值用于可视化
            normal_balance_change = balance_change[(balance_change >= lower_bound) & (balance_change <= upper_bound)]
            
            # 绘制直方图而不是箱线图
            axes[1, 0].hist(normal_balance_change, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('余额变化分布（排除异常值）')
            axes[1, 0].set_xlabel('余额变化')
            axes[1, 0].set_ylabel('频次')
            
            # 添加统计信息
            stats_text = f"正常范围: {lower_bound:.0f} ~ {upper_bound:.0f}\n"
            stats_text += f"异常值数量: {len(balance_change) - len(normal_balance_change)}"
            axes[1, 0].text(0.02, 0.98, stats_text, transform=axes[1, 0].transAxes, 
                           fontsize=8, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 4. 月度资金流分布图
        if 'Month' in self.data.columns and 'Net_Flow' in self.data.columns:
            try:
                # 按月份聚合数据
                monthly_data = self.data.groupby('Month').agg({
                    'Net_Flow': 'mean',
                    'Purchase_Amount': 'mean',
                    'Redemption_Amount': 'mean'
                }).reset_index()
                
                x = range(len(monthly_data))
                width = 0.25
                
                axes[1, 1].bar([i - width for i in x], monthly_data['Purchase_Amount'], 
                              width, label='申购金额', color='green', alpha=0.7)
                axes[1, 1].bar([i + width for i in x], monthly_data['Redemption_Amount'], 
                              width, label='赎回金额', color='red', alpha=0.7)
                axes[1, 1].plot(x, monthly_data['Net_Flow'], 'o-', label='净资金流', 
                               color='blue', linewidth=2, markersize=6)
                
                axes[1, 1].set_title('月度资金流分布')
                axes[1, 1].set_xlabel('月份')
                axes[1, 1].set_ylabel('平均金额')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels([f'{m}月' for m in monthly_data['Month']])
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"月度分布图生成失败: {e}")
        
        plt.tight_layout()
        
        if save_plot:
            # 保存图片
            plot_path = IMAGES_DIR / "cash_flow_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print_success(f"图表已保存: {plot_path}")
        
        # 关闭图形以释放内存
        plt.close()
        return True
    
    def visualize_purchase_redemption(self, save_plot=True):
        """
        申购与赎回时间序列可视化
        
        Args:
            save_plot: 是否保存图片
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
            
        # 确保数据已预处理
        if 'Purchase_Amount' not in self.data.columns:
            if not self.preprocess_data_for_purchase_redemption():
                return False
        
        print_header("申购赎回可视化", "生成申购与赎回时间序列图")
        
        # 设置中文字体 - 修复字体设置问题
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"字体设置警告: {e}")
            # 使用默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('申购与赎回时间序列分析', fontsize=16, fontweight='bold')
        
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
        axes[0, 0].plot(daily_data[time_field], daily_data['Purchase_Amount'], 
                       label='申购', color='green', linewidth=1, alpha=0.7)
        axes[0, 0].plot(daily_data[time_field], daily_data['Redemption_Amount'], 
                       label='赎回', color='red', linewidth=1, alpha=0.7)
        axes[0, 0].set_title('申购与赎回时间序列对比')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('金额')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 申购与赎回分布对比
        purchase_data = self.data[self.data['Purchase_Amount'] > 0]
        redemption_data = self.data[self.data['Redemption_Amount'] > 0]
        
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
        
        # 创建统计表格
        stats_data = {
            '统计指标': ['数量', '平均值', '标准差', '最小值', '25%分位数', '中位数', '75%分位数', '最大值'],
            '申购': [
                len(purchase_data),
                f"{purchase_stats['mean']:.2f}",
                f"{purchase_stats['std']:.2f}",
                f"{purchase_stats['min']:.2f}",
                f"{purchase_stats['25%']:.2f}",
                f"{purchase_stats['50%']:.2f}",
                f"{purchase_stats['75%']:.2f}",
                f"{purchase_stats['max']:.2f}"
            ],
            '赎回': [
                len(redemption_data),
                f"{redemption_stats['mean']:.2f}",
                f"{redemption_stats['std']:.2f}",
                f"{redemption_stats['min']:.2f}",
                f"{redemption_stats['25%']:.2f}",
                f"{redemption_stats['50%']:.2f}",
                f"{redemption_stats['75%']:.2f}",
                f"{redemption_stats['max']:.2f}"
            ]
        }
        
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
            # 保存图片
            plot_path = IMAGES_DIR / "purchase_redemption_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print_success(f"申购赎回图表已保存: {plot_path}")
        
        # 关闭图形以释放内存
        plt.close()
        return True
    
    def get_data_summary(self):
        """
        获取数据摘要信息
        
        Returns:
            dict: 数据摘要信息
        """
        if self.data is None:
            return None
            
        summary = {
            "数据形状": self.data.shape,
            "字段列表": list(self.data.columns),
            "数据类型": self.data.dtypes.to_dict(),
            "缺失值统计": self.data.isnull().sum().to_dict(),
            "数值型字段统计": {}
        }
        
        # 数值型字段的统计信息
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            summary["数值型字段统计"][col] = {
                "最小值": float(self.data[col].min()),
                "最大值": float(self.data[col].max()),
                "平均值": float(self.data[col].mean()),
                "中位数": float(self.data[col].median()),
                "标准差": float(self.data[col].std())
            }
        
        return summary


def run_basic_data_analysis():
    """
    运行基础数据分析功能
    """
    print_header("资金流预测系统", "基础数据分析")
    
    # 创建资金流预测实例
    cfp = CashFlowPrediction()
    
    # 1.1 读取数据的前五行并解析字段
    print("\n=== 步骤 1.1: 数据加载和字段解析 ===")
    if cfp.load_data():
        cfp.analyze_data_structure(top_rows=5)
        cfp.save_data_analysis()
        
        # 1.1.5 数据预处理（新增步骤）
        print("\n=== 步骤 1.1.5: 数据预处理 ===")
        if cfp.preprocess_data_for_purchase_redemption():
            print_success("数据预处理完成")
        else:
            print_error("数据预处理失败")
            return
    
    # 1.2 读取数据并展示成图片
    print("\n=== 步骤 1.2: 基础数据可视化 ===")
    cfp.visualize_data(save_plot=True)
    
    # 1.3 申购与赎回时间序列分析
    print("\n=== 步骤 1.3: 申购与赎回时间序列分析 ===")
    cfp.visualize_purchase_redemption(save_plot=True)
    
    # 显示数据摘要
    print("\n=== 数据摘要 ===")
    summary = cfp.get_data_summary()
    if summary:
        print(f"数据形状: {summary['数据形状']}")
        print(f"字段数量: {len(summary['字段列表'])}")
        print(f"字段列表: {', '.join(summary['字段列表'])}")
    
    print_success("基础数据分析完成")


if __name__ == "__main__":
    run_basic_data_analysis() 