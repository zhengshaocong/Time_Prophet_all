# -*- coding: utf-8 -*-
"""
高级数据处理模块
用于处理大数据量，包括采样、过滤、聚合等功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import DATA_PROCESSING_CONFIG, OUTPUT_DATA_DIR
from utils.interactive_utils import print_header, print_success, print_error, print_info, print_warning
from utils.file_utils import write_csv, write_json


class AdvancedDataProcessor:
    """高级数据处理器"""
    
    def __init__(self, config=None):
        """
        初始化数据处理器
        
        Args:
            config: 处理配置，如果为None则使用默认配置
        """
        self.config = config or DATA_PROCESSING_CONFIG
        self.original_data = None
        self.processed_data = None
        self.processing_stats = {}
    
    def load_data(self, file_path):
        """
        加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            print_header("数据加载")
            print(f"加载文件: {file_path}")
            
            self.original_data = pd.read_csv(file_path)
            print_success(f"数据加载成功，原始数据量: {len(self.original_data):,} 条")
            print(f"数据形状: {self.original_data.shape}")
            print(f"列名: {list(self.original_data.columns)}")
            
            return True
        except Exception as e:
            print_error(f"数据加载失败: {e}")
            return False
    
    def time_range_sampling(self, data):
        """
        时间范围采样
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 采样后的数据
        """
        if not self.config["数据采样"]["时间范围采样"]["启用"]:
            return data
        
        time_config = self.config["数据采样"]["时间范围采样"]
        time_field = self.config["数据聚合"]["时间字段"]
        
        if time_field not in data.columns:
            print_error(f"时间字段 '{time_field}' 不存在")
            return data
        
        try:
            # 转换时间字段
            data[time_field] = pd.to_datetime(data[time_field])
            
            # 设置时间范围
            start_date = pd.to_datetime(time_config["开始日期"])
            end_date = pd.to_datetime(time_config["结束日期"])
            
            # 过滤时间范围
            filtered_data = data[(data[time_field] >= start_date) & 
                                (data[time_field] <= end_date)]
            
            print_info(f"时间范围采样: {len(data):,} -> {len(filtered_data):,} 条")
            print(f"时间范围: {start_date} 到 {end_date}")
            
            return filtered_data
        except Exception as e:
            print_error(f"时间范围采样失败: {e}")
            return data
    
    def random_sampling(self, data):
        """
        随机采样
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 采样后的数据
        """
        if not self.config["数据采样"]["启用采样"]:
            return data
        
        sample_config = self.config["数据采样"]
        sample_ratio = sample_config["采样比例"]
        max_samples = sample_config["最大数据量"]
        
        # 计算采样数量
        target_samples = min(int(len(data) * sample_ratio), max_samples)
        
        if target_samples >= len(data):
            return data
        
        # 随机采样
        sampled_data = data.sample(n=target_samples, random_state=42)
        
        print_info(f"随机采样: {len(data):,} -> {len(sampled_data):,} 条")
        print(f"采样比例: {sample_ratio:.1%}")
        
        return sampled_data
    
    def systematic_sampling(self, data):
        """
        系统采样
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 采样后的数据
        """
        if not self.config["数据采样"]["启用采样"]:
            return data
        
        sample_config = self.config["数据采样"]
        sample_ratio = sample_config["采样比例"]
        max_samples = sample_config["最大数据量"]
        
        # 计算采样间隔
        target_samples = min(int(len(data) * sample_ratio), max_samples)
        step = len(data) // target_samples
        
        if step <= 1:
            return data
        
        # 系统采样
        sampled_data = data.iloc[::step].head(target_samples)
        
        print_info(f"系统采样: {len(data):,} -> {len(sampled_data):,} 条")
        print(f"采样间隔: {step}")
        
        return sampled_data
    
    def detect_outliers(self, data, column):
        """
        检测异常值
        
        Args:
            data: 数据
            column: 列名
            
        Returns:
            pd.Series: 异常值掩码
        """
        outlier_config = self.config["数据过滤"]["异常值处理"]
        method = outlier_config["方法"]
        threshold = outlier_config["阈值"]
        
        if method == "iqr":
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data[column] < lower_bound) | (data[column] > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            return z_scores > threshold
        
        elif method == "percentile":
            lower_bound = data[column].quantile(threshold / 100)
            upper_bound = data[column].quantile(1 - threshold / 100)
            return (data[column] < lower_bound) | (data[column] > upper_bound)
        
        return pd.Series([False] * len(data))
    
    def handle_outliers(self, data):
        """
        处理异常值
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if not self.config["数据过滤"]["异常值处理"]["启用"]:
            return data
        
        outlier_config = self.config["数据过滤"]["异常值处理"]
        method = outlier_config["处理方式"]
        
        # 数值列
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        total_outliers = 0
        for column in numeric_columns:
            outlier_mask = self.detect_outliers(data, column)
            outlier_count = outlier_mask.sum()
            total_outliers += outlier_count
            
            if outlier_count > 0:
                if method == "remove":
                    data = data[~outlier_mask]
                elif method == "clip":
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - outlier_config["阈值"] * IQR
                    upper_bound = Q3 + outlier_config["阈值"] * IQR
                    data[column] = data[column].clip(lower_bound, upper_bound)
                elif method == "fill":
                    median_val = data[column].median()
                    data.loc[outlier_mask, column] = median_val
        
        if total_outliers > 0:
            print_info(f"异常值处理: 处理了 {total_outliers} 个异常值")
        
        return data
    
    def handle_missing_values(self, data):
        """
        处理缺失值
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if not self.config["数据过滤"]["缺失值处理"]["启用"]:
            return data
        
        missing_config = self.config["数据过滤"]["缺失值处理"]
        method = missing_config["方法"]
        fill_value = missing_config["填充值"]
        max_missing_ratio = missing_config["最大缺失比例"]
        
        # 检查缺失值比例
        missing_ratio = data.isnull().sum() / len(data)
        high_missing_columns = missing_ratio[missing_ratio > max_missing_ratio].index
        
        if len(high_missing_columns) > 0:
            print_info(f"删除高缺失值列: {list(high_missing_columns)}")
            data = data.drop(columns=high_missing_columns)
        
        # 处理剩余缺失值
        if method == "drop":
            data = data.dropna()
        elif method == "fill":
            data = data.fillna(fill_value)
        elif method == "interpolate":
            data = data.interpolate()
        
        print_info(f"缺失值处理: 处理后数据量 {len(data):,} 条")
        
        return data
    
    def remove_duplicates(self, data):
        """
        处理重复值
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if not self.config["数据质量"]["重复值处理"]["启用"]:
            return data
        
        duplicate_config = self.config["数据质量"]["重复值处理"]
        method = duplicate_config["方法"]
        
        original_count = len(data)
        
        if method == "drop":
            data = data.drop_duplicates()
        elif method == "keep_first":
            data = data.drop_duplicates(keep='first')
        elif method == "keep_last":
            data = data.drop_duplicates(keep='last')
        
        removed_count = original_count - len(data)
        if removed_count > 0:
            print_info(f"重复值处理: 删除了 {removed_count} 条重复记录")
        
        return data
    
    def aggregate_data(self, data):
        """
        数据聚合
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 聚合后的数据
        """
        if not self.config["数据聚合"]["启用聚合"]:
            print_info("跳过数据聚合 (聚合功能未启用)")
            print(f"  ⏭️  聚合功能已禁用，保持原始数据量: {len(data):,} 条")
            return data
        
        print_info("开始执行数据聚合...")
        agg_config = self.config["数据聚合"]
        time_field = agg_config["时间字段"]
        agg_way = agg_config["聚合方式"]
        agg_func = agg_config["聚合函数"]
        
        print(f"  🔄 聚合功能已启用")
        print(f"  📅 聚合方式: {agg_way}")
        print(f"  🧮 聚合函数: {agg_func}")
        print(f"  🕒 时间字段: {time_field}")
        
        if time_field not in data.columns:
            print_error(f"时间字段 '{time_field}' 不存在")
            print(f"  ❌ 聚合失败: 时间字段 '{time_field}' 不存在")
            print(f"  💡 建议: 检查时间字段名称是否正确，或使用 'auto' 自动检测")
            return data
        
        try:
            # 确保时间字段是datetime类型
            data[time_field] = pd.to_datetime(data[time_field])
            print(f"  🔄 已转换时间字段 '{time_field}' 为datetime类型")
            
            # 设置时间索引
            data = data.set_index(time_field)
            
            # 根据聚合方式设置重采样频率
            if agg_way == "daily":
                freq = "D"
            elif agg_way == "weekly":
                freq = "W"
            elif agg_way == "monthly":
                freq = "M"
            else:
                print_warning(f"不支持的聚合方式: {agg_way}")
                print(f"  ❌ 聚合失败: 不支持的聚合方式 '{agg_way}'")
                print(f"  💡 支持的聚合方式: daily, weekly, monthly")
                return data
            
            # 数值列
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            print(f"  🔍 检测到 {len(numeric_columns)} 个数值字段用于聚合")
            
            if len(numeric_columns) == 0:
                print_warning("没有找到可聚合的数值列")
                print(f"  ❌ 聚合失败: 没有找到可聚合的数值列")
                print(f"  💡 建议: 检查数据中是否包含数值字段")
                return data
            
            # 聚合
            print(f"  🚀 开始执行聚合操作...")
            if agg_func == "sum":
                aggregated = data[numeric_columns].resample(freq).sum()
            elif agg_func == "mean":
                aggregated = data[numeric_columns].resample(freq).mean()
            elif agg_func == "median":
                aggregated = data[numeric_columns].resample(freq).median()
            elif agg_func == "max":
                aggregated = data[numeric_columns].resample(freq).max()
            elif agg_func == "min":
                aggregated = data[numeric_columns].resample(freq).min()
            else:
                print_warning(f"不支持的聚合函数: {agg_func}，使用sum")
                aggregated = data[numeric_columns].resample(freq).sum()
            
            # 重置索引
            aggregated = aggregated.reset_index()
            print(f"  🔄 已重置时间索引")
            
            print_info(f"数据聚合: {len(data):,} -> {len(aggregated):,} 条")
            print(f"聚合方式: {agg_way}, 聚合函数: {agg_func}")
            
            # 添加聚合结果提示
            if len(aggregated) != len(data):
                print(f"  ✅ 聚合完成: {len(data):,} -> {len(aggregated):,} 条 (减少 {((len(data) - len(aggregated)) / len(data) * 100):.1f}%)")
            else:
                print(f"  ⚠️  聚合未生效: 数据量未变化 ({len(data):,} -> {len(aggregated):,} 条)")
            
            print(f"  ✅ 聚合操作完成!")
            return aggregated
        except Exception as e:
            print_error(f"数据聚合失败: {e}")
            print(f"  ❌ 聚合过程中发生错误: {e}")
            print(f"  💡 建议: 检查数据格式、时间字段格式、聚合配置等")
            return data
    
    def process_data(self, file_path=None, data=None):
        """
        完整的数据处理流程
        
        Args:
            file_path: 数据文件路径
            data: 输入数据，如果提供则忽略file_path
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        print_header("高级数据处理", "大数据量处理")
        
        # 加载数据
        if data is not None:
            self.original_data = data
            print_success(f"使用提供的数据，原始数据量: {len(data):,} 条")
        elif file_path is not None:
            if not self.load_data(file_path):
                return None
        else:
            print_error("请提供数据文件路径或数据")
            return None
        
        data = self.original_data.copy()
        original_count = len(data)
        
        # 记录处理统计
        self.processing_stats = {
            "原始数据量": original_count,
            "处理步骤": []
        }
        
        # 1. 时间范围采样
        if self.config["数据采样"]["时间范围采样"]["启用"]:
            data = self.time_range_sampling(data)
            self.processing_stats["处理步骤"].append({
                "步骤": "时间范围采样",
                "数据量": len(data)
            })
        
        # 2. 数据采样
        if self.config["数据采样"]["启用采样"]:
            sample_method = self.config["数据采样"]["采样方式"]
            if sample_method == "random":
                data = self.random_sampling(data)
            elif sample_method == "systematic":
                data = self.systematic_sampling(data)
            
            self.processing_stats["处理步骤"].append({
                "步骤": f"{sample_method}采样",
                "数据量": len(data)
            })
        
        # 3. 异常值处理
        if self.config["数据过滤"]["异常值处理"]["启用"]:
            data = self.handle_outliers(data)
            self.processing_stats["处理步骤"].append({
                "步骤": "异常值处理",
                "数据量": len(data)
            })
        
        # 4. 缺失值处理
        if self.config["数据过滤"]["缺失值处理"]["启用"]:
            data = self.handle_missing_values(data)
            self.processing_stats["处理步骤"].append({
                "步骤": "缺失值处理",
                "数据量": len(data)
            })
        
        # 5. 重复值处理
        if self.config["数据质量"]["重复值处理"]["启用"]:
            data = self.remove_duplicates(data)
            self.processing_stats["处理步骤"].append({
                "步骤": "重复值处理",
                "数据量": len(data)
            })
        
        # 6. 数据聚合
        if self.config["数据聚合"]["启用聚合"]:
            print_info("执行数据聚合...")
            print(f"  🔄 聚合功能已启用")
            print(f"  📅 聚合方式: {self.config['数据聚合']['聚合方式']}")
            print(f"  🧮 聚合函数: {self.config['数据聚合']['聚合函数']}")
            print(f"  🕒 时间字段: {self.config['数据聚合']['时间字段']}")
            
            original_count_before_aggregate = len(data)
            data = self.aggregate_data(data)
            
            # 聚合后的提示
            if len(data) != original_count_before_aggregate:
                print(f"  ✅ 聚合完成: {original_count_before_aggregate:,} -> {len(data):,} 条 (减少 {((original_count_before_aggregate - len(data)) / original_count_before_aggregate * 100):.1f}%)")
            else:
                print(f"  ⚠️  聚合未生效: 数据量未变化 ({original_count_before_aggregate:,} -> {len(data):,} 条)")
            
            self.processing_stats["处理步骤"].append({
                "步骤": "数据聚合",
                "数据量": len(data),
                "聚合方式": self.config["数据聚合"]["聚合方式"],
                "聚合函数": self.config["数据聚合"]["聚合函数"]
            })
        else:
            print_info("跳过数据聚合 (聚合功能未启用)")
            print(f"  ⏭️  聚合功能已禁用，保持原始数据量: {len(data):,} 条")
        
        # 保存处理后的数据
        if self.config["输出配置"]["保存处理后的数据"]:
            self.save_processed_data(data)
        
        # 更新统计信息
        self.processing_stats["最终数据量"] = len(data)
        self.processing_stats["数据减少比例"] = (original_count - len(data)) / original_count
        
        self.processed_data = data
        
        print_success("数据处理完成！")
        print(f"原始数据量: {original_count:,} 条")
        print(f"处理后数据量: {len(data):,} 条")
        print(f"数据减少比例: {self.processing_stats['数据减少比例']:.1%}")
        
        return data
    
    def save_processed_data(self, data):
        """
        保存处理后的数据
        
        Args:
            data: 处理后的数据
        """
        output_config = self.config["输出配置"]
        output_format = output_config["输出格式"]
        output_dir = Path(output_config["输出目录"])
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名（覆盖模式）
        filename = "processed_data"
        
        try:
            if output_format == "csv":
                file_path = output_dir / f"{filename}.csv"
                data.to_csv(file_path, index=False, encoding='utf-8')
            elif output_format == "parquet":
                file_path = output_dir / f"{filename}.parquet"
                data.to_parquet(file_path, index=False)
            elif output_format == "pickle":
                file_path = output_dir / f"{filename}.pkl"
                data.to_pickle(file_path)
            
            print_success(f"处理后的数据已保存: {file_path}")
            
            # 保存处理统计信息
            stats_file = output_dir / f"{filename}_stats.json"
            write_json(self.processing_stats, stats_file)
            print_info(f"处理统计信息已保存: {stats_file}")
            
        except Exception as e:
            print_error(f"保存数据失败: {e}")
    
    def get_processing_stats(self):
        """
        获取处理统计信息
        
        Returns:
            dict: 处理统计信息
        """
        return self.processing_stats


def process_large_dataset(file_path, config=None):
    """
    处理大数据集的便捷函数
    
    Args:
        file_path: 数据文件路径
        config: 处理配置
        
    Returns:
        pd.DataFrame: 处理后的数据
    """
    processor = AdvancedDataProcessor(config)
    return processor.process_data(file_path)


def process_dataframe(data, config=None):
    """
    处理DataFrame的便捷函数
    
    Args:
        data: 输入DataFrame
        config: 处理配置
        
    Returns:
        pd.DataFrame: 处理后的数据
    """
    processor = AdvancedDataProcessor(config)
    return processor.process_data(data=data) 