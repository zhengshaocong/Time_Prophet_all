# -*- coding: utf-8 -*-
"""
数据处理模块
包含数据清洗、特征工程、数据转换等完整的数据处理功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_DIR, OUTPUT_DATA_DIR, DATA_PREPROCESSING_CONFIG, 
    DATA_PROCESSING_CONFIG, FEATURE_ENGINEERING_CONFIG, 
    DATA_TRANSFORMATION_CONFIG, DEFAULT_FIELD_MAPPING
)
from utils.interactive_utils import print_header, print_success, print_error, print_info, print_warning
from utils.file_utils import write_csv, write_json
from utils.data_processor import DataProcessor
from utils.data_processor_advanced import AdvancedDataProcessor


class DataProcessingPipeline:
    """数据处理流水线"""
    
    def __init__(self, config=None):
        """
        初始化数据处理流水线
        
        Args:
            config: 处理配置，如果为None则使用默认配置
        """
        self.config = config or DATA_PROCESSING_CONFIG
        self.preprocessing_config = DATA_PREPROCESSING_CONFIG
        self.feature_config = FEATURE_ENGINEERING_CONFIG
        self.transformation_config = DATA_TRANSFORMATION_CONFIG
        self.field_mapping = DEFAULT_FIELD_MAPPING
        self.data = None
        self.processed_data = None
        self.processing_log = []
        
    def load_and_analyze_data(self, file_path=None):
        """
        加载并分析数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            bool: 是否成功
        """
        print_header("数据加载与分析")
        
        if file_path is None:
            file_path = DATA_DIR / "user_balance_table.csv"
        
        try:
            # 使用基础数据处理器加载数据
            processor = DataProcessor()
            if not processor.load_data(file_path):
                return False
            
            # 分析数据结构
            if not processor.analyze_data_structure():
                return False
            
            self.data = processor.data
            print_success("数据加载与分析完成")
            return True
            
        except Exception as e:
            print_error(f"数据加载与分析失败: {e}")
            return False
    
    def clean_data(self):
        """
        数据清洗
        
        Returns:
            bool: 是否成功
        """
        print_header("数据清洗")
        
        if self.data is None:
            print_error("请先加载数据")
            return False
        
        try:
            data = self.data.copy()
            original_count = len(data)
            
            # 1. 处理缺失值
            print_info("处理缺失值...")
            missing_config = self.preprocessing_config["缺失值处理"]
            for field, fill_value in missing_config.items():
                if field in data.columns:
                    data[field] = data[field].fillna(fill_value)
                    print(f"  {field}: 填充缺失值为 {fill_value}")
            
            # 2. 处理异常值
            if self.preprocessing_config["异常值处理"]["启用异常值检测"]:
                print_info("处理异常值...")
                data = self._handle_outliers(data)
            
            # 3. 数据类型转换
            print_info("转换数据类型...")
            data = self._convert_data_types(data)
            
            # 4. 时间字段处理
            print_info("处理时间字段...")
            data = self._process_time_field(data)
            
            # 5. 数据一致性检查
            print_info("检查数据一致性...")
            data = self._check_data_consistency(data)
            
            self.data = data
            cleaned_count = len(data)
            
            print_success(f"数据清洗完成: {original_count:,} -> {cleaned_count:,} 条")
            self.processing_log.append({
                "步骤": "数据清洗",
                "原始数据量": original_count,
                "清洗后数据量": cleaned_count,
                "时间": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print_error(f"数据清洗失败: {e}")
            return False
    
    def _handle_outliers(self, data):
        """处理异常值"""
        outlier_config = self.preprocessing_config["异常值处理"]
        threshold = outlier_config["异常值阈值"]
        method = outlier_config["异常值处理方式"]
        
        # 数值列
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['user_id', 'report_date']:  # 跳过ID和时间字段
                continue
                
            mean_val = data[column].mean()
            std_val = data[column].std()
            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val
            
            outlier_count = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
            
            if outlier_count > 0:
                if method == "clip":
                    data[column] = data[column].clip(lower_bound, upper_bound)
                    print(f"  {column}: 截断 {outlier_count} 个异常值")
                elif method == "remove":
                    data = data[~((data[column] < lower_bound) | (data[column] > upper_bound))]
                    print(f"  {column}: 删除 {outlier_count} 个异常值")
        
        return data
    
    def _convert_data_types(self, data):
        """转换数据类型"""
        # 转换时间字段
        time_field = self.field_mapping["时间字段"]
        if time_field in data.columns:
            data[time_field] = pd.to_datetime(data[time_field], format=self.field_mapping["时间格式"])
        
        # 确保数值字段为数值类型
        numeric_fields = [
            self.field_mapping["申购金额字段"],
            self.field_mapping["赎回金额字段"],
            self.field_mapping["当前余额字段"],
            self.field_mapping["昨日余额字段"]
        ]
        
        for field in numeric_fields:
            if field in data.columns:
                data[field] = pd.to_numeric(data[field], errors='coerce')
        
        return data
    
    def _process_time_field(self, data):
        """处理时间字段"""
        time_field = self.field_mapping["时间字段"]
        if time_field not in data.columns:
            return data
        
        time_features = self.preprocessing_config["时间特征"]
        
        # 提取时间特征
        if time_features["提取年份"]:
            data['Year'] = data[time_field].dt.year
        if time_features["提取月份"]:
            data['Month'] = data[time_field].dt.month
        if time_features["提取日期"]:
            data['Day'] = data[time_field].dt.day
        if time_features["提取星期"]:
            data['Weekday'] = data[time_field].dt.dayofweek
        if time_features["提取季度"]:
            data['Quarter'] = data[time_field].dt.quarter
        
        return data
    
    def _check_data_consistency(self, data):
        """检查数据一致性"""
        # 检查余额字段的一致性
        current_balance_field = self.field_mapping["当前余额字段"]
        previous_balance_field = self.field_mapping["昨日余额字段"]
        
        if current_balance_field in data.columns and previous_balance_field in data.columns:
            # 检查余额是否为负数
            negative_balance = (data[current_balance_field] < 0).sum()
            if negative_balance > 0:
                print_warning(f"发现 {negative_balance} 条负余额记录")
            
            # 检查余额变化是否合理
            balance_change = data[current_balance_field] - data[previous_balance_field]
            extreme_changes = (abs(balance_change) > balance_change.quantile(0.99)).sum()
            if extreme_changes > 0:
                print_warning(f"发现 {extreme_changes} 条极端余额变化记录")
        
        return data
    
    def engineer_features(self):
        """
        特征工程
        
        Returns:
            bool: 是否成功
        """
        print_header("特征工程")
        
        if self.data is None:
            print_error("请先加载和清洗数据")
            return False
        
        try:
            data = self.data.copy()
            original_columns = len(data.columns)
            
            # 1. 计算基础特征
            if self.feature_config["基础特征"]["启用"]:
                print_info("计算基础特征...")
                data = self._calculate_basic_features(data)
            
            # 2. 计算时间特征
            if self.feature_config["时间特征"]["启用"]:
                print_info("计算时间特征...")
                data = self._calculate_time_features(data)
            
            # 3. 计算统计特征
            if self.feature_config["统计特征"]["启用"]:
                print_info("计算统计特征...")
                data = self._calculate_statistical_features(data)
            
            # 4. 计算用户特征
            if self.feature_config["用户特征"]["启用"]:
                print_info("计算用户特征...")
                data = self._calculate_user_features(data)
            
            # 5. 计算业务特征
            if self.feature_config["业务特征"]["启用"]:
                print_info("计算业务特征...")
                data = self._calculate_business_features(data)
            
            self.data = data
            new_columns = len(data.columns)
            
            print_success(f"特征工程完成: {original_columns} -> {new_columns} 列")
            print(f"新增特征: {new_columns - original_columns} 个")
            
            self.processing_log.append({
                "步骤": "特征工程",
                "原始列数": original_columns,
                "新列数": new_columns,
                "新增特征数": new_columns - original_columns,
                "时间": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print_error(f"特征工程失败: {e}")
            return False
    
    def _calculate_basic_features(self, data):
        """计算基础特征"""
        basic_config = self.feature_config["基础特征"]
        
        # 净资金流
        if basic_config["净资金流"]:
            purchase_field = self.field_mapping["申购金额字段"]
            redemption_field = self.field_mapping["赎回金额字段"]
            
            if purchase_field in data.columns and redemption_field in data.columns:
                data['Net_Flow'] = data[purchase_field] - data[redemption_field]
        
        # 总资金流
        if basic_config["总资金流"]:
            purchase_field = self.field_mapping["申购金额字段"]
            redemption_field = self.field_mapping["赎回金额字段"]
            
            if purchase_field in data.columns and redemption_field in data.columns:
                data['Total_Flow'] = data[purchase_field] + data[redemption_field]
        
        # 资金流比率
        if basic_config["资金流比率"]:
            purchase_field = self.field_mapping["申购金额字段"]
            redemption_field = self.field_mapping["赎回金额字段"]
            
            if purchase_field in data.columns and redemption_field in data.columns:
                data['Flow_Ratio'] = data[redemption_field] / (data[purchase_field] + 1e-8)  # 避免除零
        
        # 余额变化
        if basic_config["余额变化"]:
            current_balance_field = self.field_mapping["当前余额字段"]
            previous_balance_field = self.field_mapping["昨日余额字段"]
            
            if current_balance_field in data.columns and previous_balance_field in data.columns:
                data['Balance_Change'] = data[current_balance_field] - data[previous_balance_field]
        
        # 余额变化率
        if basic_config["余额变化率"]:
            current_balance_field = self.field_mapping["当前余额字段"]
            previous_balance_field = self.field_mapping["昨日余额字段"]
            
            if current_balance_field in data.columns and previous_balance_field in data.columns:
                data['Balance_Change_Rate'] = data['Balance_Change'] / (data[previous_balance_field] + 1e-8)
        
        return data
    
    def _calculate_time_features(self, data):
        """计算时间特征"""
        time_config = self.feature_config["时间特征"]
        time_field = self.field_mapping["时间字段"]
        
        if time_field not in data.columns:
            return data
        
        # 基础时间特征（已在数据清洗阶段添加）
        if time_config["年中天数"]:
            data['DayOfYear'] = data[time_field].dt.dayofyear
        if time_config["年中周数"]:
            data['WeekOfYear'] = data[time_field].dt.isocalendar().week
        if time_config["月"]:
            data['MonthOfYear'] = data[time_field].dt.month
        
        # 月初月末
        if time_config["月初月末"]:
            data['IsMonthStart'] = data[time_field].dt.is_month_start.astype(int)
            data['IsMonthEnd'] = data[time_field].dt.is_month_end.astype(int)
        
        # 周末
        if time_config["周末"]:
            data['IsWeekend'] = (data['Weekday'] >= 5).astype(int)
        
        # 节假日（简单判断）
        if time_config["节假日"]:
            data['IsHoliday'] = ((data['Month'] == 1) & (data['Day'] <= 3)).astype(int)  # 元旦
            data['IsHoliday'] |= ((data['Month'] == 5) & (data['Day'] >= 1) & (data['Day'] <= 3)).astype(int)  # 劳动节
            data['IsHoliday'] |= ((data['Month'] == 10) & (data['Day'] >= 1) & (data['Day'] <= 7)).astype(int)  # 国庆节
        
        return data
    
    def _calculate_statistical_features(self, data):
        """计算统计特征"""
        stats_config = self.feature_config["统计特征"]
        feature_list = stats_config["特征列表"]
        windows = stats_config["滚动窗口"]
        functions = stats_config["统计函数"]
        
        # 获取可用的特征
        available_features = [col for col in feature_list if col in data.columns]
        
        if available_features:
            for feature in available_features:
                for window in windows:
                    for func in functions:
                        if func == "mean":
                            data[f'{feature}_{window}d_mean'] = data[feature].rolling(window=window, min_periods=1).mean()
                        elif func == "std":
                            data[f'{feature}_{window}d_std'] = data[feature].rolling(window=window, min_periods=1).std()
                        elif func == "max":
                            data[f'{feature}_{window}d_max'] = data[feature].rolling(window=window, min_periods=1).max()
                        elif func == "min":
                            data[f'{feature}_{window}d_min'] = data[feature].rolling(window=window, min_periods=1).min()
        
        return data
    
    def _calculate_user_features(self, data):
        """计算用户特征"""
        user_id_field = self.field_mapping["用户ID字段"]
        if user_id_field not in data.columns:
            return data
        
        # 用户级别的统计特征
        user_stats = data.groupby(user_id_field).agg({
            'Net_Flow': ['mean', 'std', 'sum', 'count'],
            'Total_Flow': ['mean', 'std', 'sum'],
            'Balance_Change': ['mean', 'std', 'sum']
        }).reset_index()
        
        # 重命名列
        user_stats.columns = [
            user_id_field,
            'User_NetFlow_Mean', 'User_NetFlow_Std', 'User_NetFlow_Sum', 'User_Transaction_Count',
            'User_TotalFlow_Mean', 'User_TotalFlow_Std', 'User_TotalFlow_Sum',
            'User_BalanceChange_Mean', 'User_BalanceChange_Std', 'User_BalanceChange_Sum'
        ]
        
        # 合并回原数据
        data = data.merge(user_stats, on=user_id_field, how='left')
        
        return data
    
    def _calculate_business_features(self, data):
        """计算业务特征"""
        # 交易活跃度
        if 'User_Transaction_Count' in data.columns:
            data['Transaction_Activity'] = pd.cut(
                data['User_Transaction_Count'],
                bins=[0, 10, 50, 100, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
        
        # 资金流类型
        if 'Net_Flow' in data.columns:
            data['Flow_Type'] = pd.cut(
                data['Net_Flow'],
                bins=[float('-inf'), -1000, 0, 1000, float('inf')],
                labels=['Large_Outflow', 'Outflow', 'Inflow', 'Large_Inflow']
            )
        
        # 余额水平
        current_balance_field = self.field_mapping["当前余额字段"]
        if current_balance_field in data.columns:
            data['Balance_Level'] = pd.cut(
                data[current_balance_field],
                bins=[0, 10000, 100000, 1000000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
        
        return data
    
    def transform_data(self):
        """
        数据转换
        
        Returns:
            bool: 是否成功
        """
        print_header("数据转换")
        
        if self.data is None:
            print_error("请先完成特征工程")
            return False
        
        try:
            data = self.data.copy()
            
            # 1. 标准化数值特征
            if self.transformation_config["标准化"]["启用"]:
                print_info("标准化数值特征...")
                data = self._standardize_numeric_features(data)
            
            # 2. 编码分类特征
            if self.transformation_config["编码"]["启用"]:
                print_info("编码分类特征...")
                data = self._encode_categorical_features(data)
            
            # 3. 处理时间序列特征
            if self.transformation_config["时间序列特征"]["启用"]:
                print_info("处理时间序列特征...")
                data = self._process_time_series_features(data)
            
            # 4. 特征选择
            if self.transformation_config["特征选择"]["启用"]:
                print_info("特征选择...")
                data = self._select_features(data)
            
            self.processed_data = data
            
            print_success(f"数据转换完成: {len(data.columns)} 个特征")
            
            self.processing_log.append({
                "步骤": "数据转换",
                "最终特征数": len(data.columns),
                "时间": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print_error(f"数据转换失败: {e}")
            return False
    
    def _standardize_numeric_features(self, data):
        """标准化数值特征"""
        standardize_config = self.transformation_config["标准化"]
        feature_list = standardize_config["特征列表"]
        method = standardize_config["方法"]
        
        available_features = [col for col in feature_list if col in data.columns]
        
        for feature in available_features:
            if method == "zscore":
                mean_val = data[feature].mean()
                std_val = data[feature].std()
                if std_val > 0:
                    data[f'{feature}_Normalized'] = (data[feature] - mean_val) / std_val
            elif method == "minmax":
                min_val = data[feature].min()
                max_val = data[feature].max()
                if max_val > min_val:
                    data[f'{feature}_Normalized'] = (data[feature] - min_val) / (max_val - min_val)
            elif method == "robust":
                median_val = data[feature].median()
                q75 = data[feature].quantile(0.75)
                q25 = data[feature].quantile(0.25)
                iqr = q75 - q25
                if iqr > 0:
                    data[f'{feature}_Normalized'] = (data[feature] - median_val) / iqr
        
        return data
    
    def _encode_categorical_features(self, data):
        """编码分类特征"""
        encode_config = self.transformation_config["编码"]
        feature_list = encode_config["特征列表"]
        method = encode_config["方法"]
        
        for feature in feature_list:
            if feature in data.columns:
                if method == "onehot":
                    # 独热编码
                    dummies = pd.get_dummies(data[feature], prefix=feature)
                    data = pd.concat([data, dummies], axis=1)
                    data = data.drop(columns=[feature])
                elif method == "label":
                    # 标签编码
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    data[f'{feature}_Encoded'] = le.fit_transform(data[feature].astype(str))
                    data = data.drop(columns=[feature])
        
        return data
    
    def _process_time_series_features(self, data):
        """处理时间序列特征"""
        ts_config = self.transformation_config["时间序列特征"]
        feature_list = ts_config["特征列表"]
        lag_periods = ts_config["滞后期数"]
        
        if ts_config["滞后特征"]:
            available_features = [col for col in feature_list if col in data.columns]
            
            for feature in available_features:
                for lag in lag_periods:
                    data[f'{feature}_Lag{lag}'] = data[feature].shift(lag)
        
        return data
    
    def _select_features(self, data):
        """特征选择"""
        selection_config = self.transformation_config["特征选择"]
        missing_threshold = selection_config["缺失值阈值"]
        correlation_threshold = selection_config["相关性阈值"]
        variance_threshold = selection_config["方差阈值"]
        
        # 移除包含太多缺失值的特征
        missing_ratio = data.isnull().sum() / len(data)
        columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
        data = data.drop(columns=columns_to_drop)
        
        # 移除常量特征
        constant_columns = [col for col in data.columns if data[col].nunique() <= 1]
        data = data.drop(columns=constant_columns)
        
        # 移除低方差特征
        numeric_data = data.select_dtypes(include=[np.number])
        variance = numeric_data.var()
        low_variance_columns = variance[variance < variance_threshold].index
        data = data.drop(columns=low_variance_columns)
        
        # 移除高度相关的特征
        correlation_matrix = numeric_data.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_columns = [column for column in upper_triangle.columns 
                           if any(upper_triangle[column] > correlation_threshold)]
        data = data.drop(columns=high_corr_columns)
        
        return data
    
    def save_processed_data(self, output_path=None):
        """
        保存处理后的数据
        
        Args:
            output_path: 输出路径
        """
        if self.processed_data is None:
            print_error("没有处理后的数据可保存")
            return False
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DATA_DIR / f"processed_data_{timestamp}.csv"
        
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存数据
            self.processed_data.to_csv(output_path, index=False, encoding='utf-8')
            print_success(f"处理后的数据已保存: {output_path}")
            
            # 保存处理日志
            log_path = output_path.parent / f"processing_log_{timestamp}.json"
            write_json(self.processing_log, log_path)
            print_info(f"处理日志已保存: {log_path}")
            
            return True
            
        except Exception as e:
            print_error(f"保存数据失败: {e}")
            return False
    
    def get_processed_data_path(self):
        """
        获取处理后的数据路径
        
        Returns:
            str: 处理后数据路径
        """
        if self.processed_data is None:
            return None
        
        # 查找最新的处理后数据文件
        output_dir = OUTPUT_DATA_DIR / "data"
        if output_dir.exists():
            processed_files = list(output_dir.glob("processed_data_*.csv"))
            if processed_files:
                # 按修改时间排序，返回最新的
                latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
                return str(latest_file)
        
        return None
    
    def run_full_pipeline(self, file_path=None):
        """
        运行完整的数据处理流水线
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            bool: 是否成功
        """
        print_header("完整数据处理流水线", "数据清洗 -> 特征工程 -> 数据转换")
        
        # 1. 加载和分析数据
        if not self.load_and_analyze_data(file_path):
            return False
        
        # 2. 数据清洗
        if not self.clean_data():
            return False
        
        # 3. 特征工程
        if not self.engineer_features():
            return False
        
        # 4. 数据转换
        if not self.transform_data():
            return False
        
        # 5. 保存结果
        if not self.save_processed_data():
            return False
        
        print_success("完整数据处理流水线执行完成！")
        return True


def run_data_processing():
    """运行数据处理功能"""
    print_header("数据处理模块", "数据清洗、特征工程、数据转换")
    
    # 创建数据处理流水线
    pipeline = DataProcessingPipeline()
    
    # 运行完整流水线
    success = pipeline.run_full_pipeline()
    
    if success:
        print_success("数据处理完成！")
        print(f"处理日志包含 {len(pipeline.processing_log)} 个步骤")
        print("处理后的数据已保存到输出目录")
    else:
        print_error("数据处理失败！")
    
    return success


if __name__ == "__main__":
    run_data_processing() 