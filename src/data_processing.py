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
        
        # 保存数据文件路径
        self.data_file_path = Path(file_path)
        
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
            
            # 1. 数据类型转换
            print_info("转换数据类型...")
            data = self._convert_data_types(data)
            
            # 2. 时间字段处理
            print_info("处理时间字段...")
            data = self._process_time_field(data)
            
            # 3. 时间范围过滤
            print_info("应用时间范围过滤...")
            data = self._apply_time_range_filter(data)
            
            # 4. 处理缺失值
            print_info("处理缺失值...")
            missing_config = self.preprocessing_config["缺失值处理"]
            for field, fill_value in missing_config.items():
                if field in data.columns:
                    data[field] = data[field].fillna(fill_value)
                    print(f"  {field}: 填充缺失值为 {fill_value}")
            
            # 5. 处理异常值
            if self.preprocessing_config["异常值处理"]["启用异常值检测"]:
                print_info("处理异常值...")
                data = self._handle_outliers(data)
            
            # 5. 数据一致性检查
            print_info("检查数据一致性...")
            data = self._check_data_consistency(data)
            
            # 6. 数据聚合
            aggregate_config = self.preprocessing_config["数据聚合"]
            if aggregate_config["启用聚合"]:
                print_info("执行数据聚合...")
                print(f"  🔄 聚合功能已启用")
                print(f"  📅 聚合方式: {aggregate_config['聚合方式']}")
                print(f"  🧮 聚合函数: {aggregate_config['聚合函数']}")
                print(f"  🕒 时间字段: {aggregate_config['时间字段']}")
                data = self._aggregate_data(data)
                # 聚合后的提示
                if len(data) != original_count:
                    print(f"  ✅ 聚合完成: {original_count:,} -> {len(data):,} 条 (减少 {((original_count - len(data)) / original_count * 100):.1f}%)")
                else:
                    print(f"  ⚠️  聚合未生效: 数据量未变化 ({original_count:,} -> {len(data):,} 条)")
            else:
                print_info("跳过数据聚合 (聚合功能未启用)")
                print(f"  ⏭️  聚合功能已禁用，保持原始数据量: {len(data):,} 条")
            
            self.data = data
            cleaned_count = len(data)
            
            print_success(f"数据清洗完成: {original_count:,} -> {cleaned_count:,} 条")
            
            # 添加聚合状态总结
            if aggregate_config["启用聚合"] and len(data) != original_count:
                print(f"\n📊 【聚合状态总结】")
                print(f"  原始数据量: {original_count:,} 条")
                print(f"  聚合后数据量: {cleaned_count:,} 条")
                print(f"  数据减少: {original_count - cleaned_count:,} 条 ({((original_count - cleaned_count) / original_count * 100):.1f}%)")
                print(f"  聚合方式: {aggregate_config['聚合方式']} | 聚合函数: {aggregate_config['聚合函数']}")
                print(f"  ✅ 聚合功能已成功执行")
            elif aggregate_config["启用聚合"]:
                print(f"\n⚠️  【聚合状态总结】")
                print(f"  聚合功能已启用但未生效")
                print(f"  可能原因: 时间字段未找到、聚合字段为空、数据格式问题等")
                print(f"  建议检查: 时间字段配置、数据格式、聚合字段设置")
            else:
                print(f"\n⏭️  【聚合状态总结】")
                print(f"  聚合功能已禁用")
                print(f"  如需启用聚合，请修改 config/data_processing.py 中的 '启用聚合': True")
            
            self.processing_log.append({
                "步骤": "数据清洗",
                "原始数据量": original_count,
                "清洗后数据量": cleaned_count,
                "聚合状态": "已启用" if aggregate_config["启用聚合"] else "已禁用",
                "聚合方式": aggregate_config["聚合方式"] if aggregate_config["启用聚合"] else "无",
                "聚合函数": aggregate_config["聚合函数"] if aggregate_config["启用聚合"] else "无",
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
    
    def _apply_time_range_filter(self, data):
        """
        应用时间范围过滤
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        time_range_config = self.preprocessing_config["时间范围限制"]
        
        if not time_range_config["启用时间范围限制"]:
            return data
        
        time_field = self.field_mapping["时间字段"]
        if time_field not in data.columns:
            print_warning(f"时间字段 '{time_field}' 不存在，跳过时间范围过滤")
            return data
        
        try:
            # 确保时间字段是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(data[time_field]):
                data[time_field] = pd.to_datetime(data[time_field])
            
            # 获取原始数据的时间范围
            original_start = data[time_field].min()
            original_end = data[time_field].max()
            
            # 解析配置中的时间范围
            config_start = pd.to_datetime(time_range_config["开始日期"])
            config_end = pd.to_datetime(time_range_config["结束日期"])
            
            # 检查配置的时间范围是否超出原始数据范围
            if time_range_config["超出范围警告"]:
                if config_start < original_start:
                    print_warning(f"配置的开始日期 {config_start.date()} 早于数据最早日期 {original_start.date()}")
                if config_end > original_end:
                    print_warning(f"配置的结束日期 {config_end.date()} 晚于数据最晚日期 {original_end.date()}")
            
            # 自动调整时间范围
            if time_range_config["自动调整"]:
                effective_start = max(config_start, original_start)
                effective_end = min(config_end, original_end)
                print_info(f"自动调整时间范围: {effective_start.date()} 到 {effective_end.date()}")
            else:
                effective_start = config_start
                effective_end = config_end
            
            # 应用时间范围过滤
            filtered_data = data[(data[time_field] >= effective_start) & 
                                (data[time_field] <= effective_end)]
            
            filtered_count = len(filtered_data)
            original_count = len(data)
            
            print_info(f"时间范围过滤: {original_count:,} -> {filtered_count:,} 条")
            print_info(f"时间范围: {effective_start.date()} 到 {effective_end.date()}")
            
            if filtered_count == 0:
                print_warning("过滤后数据为空，请检查时间范围配置")
            
            return filtered_data
            
        except Exception as e:
            print_error(f"时间范围过滤失败: {e}")
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
        """检查数据一致性并处理异常值"""
        consistency_config = self.preprocessing_config["数据一致性处理"]
        
        if not consistency_config["启用一致性检查"]:
            return data
        
        processing_method = consistency_config["处理方式"]
        original_count = len(data)
        
        # 检查余额字段的一致性
        current_balance_field = self.field_mapping["当前余额字段"]
        previous_balance_field = self.field_mapping["昨日余额字段"]
        
        if current_balance_field in data.columns and previous_balance_field in data.columns:
            # 检查余额是否为负数
            if consistency_config["处理负余额"]:
                negative_balance = (data[current_balance_field] < 0).sum()
                if negative_balance > 0:
                    print_warning(f"发现 {negative_balance} 条负余额记录")
                    if processing_method == "correct":
                        # 将负余额设为0
                        data.loc[data[current_balance_field] < 0, current_balance_field] = 0
                        print(f"  已将负余额记录设为0")
                    elif processing_method == "remove":
                        # 删除负余额记录
                        data = data[data[current_balance_field] >= 0]
                        print(f"  已删除负余额记录")
            
            # 检查余额变化是否合理
            if consistency_config["处理极端余额变化"]:
                balance_change = data[current_balance_field] - data[previous_balance_field]
                extreme_threshold = balance_change.quantile(consistency_config["极端变化阈值"])
                extreme_changes = (abs(balance_change) > extreme_threshold).sum()
                
                if extreme_changes > 0:
                    print_warning(f"发现 {extreme_changes} 条极端余额变化记录")
                    if processing_method == "correct":
                        # 处理极端余额变化：将极端变化限制在合理范围内
                        extreme_mask = abs(balance_change) > extreme_threshold
                        
                        # 对于极端变化，使用中位数作为合理的余额变化值
                        median_change = balance_change.median()
                        data.loc[extreme_mask, current_balance_field] = (
                            data.loc[extreme_mask, previous_balance_field] + median_change
                        )
                        print(f"  已将极端余额变化调整为中位数变化值")
                    elif processing_method == "remove":
                        # 删除极端余额变化记录
                        data = data[abs(balance_change) <= extreme_threshold]
                        print(f"  已删除极端余额变化记录")
        
        # 检查数据逻辑一致性
        if consistency_config["处理负金额"]:
            purchase_field = self.field_mapping["申购金额字段"]
            redemption_field = self.field_mapping["赎回金额字段"]
            
            if purchase_field in data.columns and redemption_field in data.columns:
                # 检查是否有负的申购或赎回金额
                negative_purchase = (data[purchase_field] < 0).sum()
                negative_redemption = (data[redemption_field] < 0).sum()
                
                if negative_purchase > 0:
                    print_warning(f"发现 {negative_purchase} 条负申购金额记录")
                    if processing_method == "correct":
                        data.loc[data[purchase_field] < 0, purchase_field] = 0
                        print(f"  已将负申购金额设为0")
                    elif processing_method == "remove":
                        data = data[data[purchase_field] >= 0]
                        print(f"  已删除负申购金额记录")
                
                if negative_redemption > 0:
                    print_warning(f"发现 {negative_redemption} 条负赎回金额记录")
                    if processing_method == "correct":
                        data.loc[data[redemption_field] < 0, redemption_field] = 0
                        print(f"  已将负赎回金额设为0")
                    elif processing_method == "remove":
                        data = data[data[redemption_field] >= 0]
                        print(f"  已删除负赎回金额记录")
        
        # 显示处理结果
        final_count = len(data)
        if final_count != original_count:
            print(f"  数据一致性处理完成: {original_count:,} -> {final_count:,} 条")
            print(f"  删除了 {original_count - final_count:,} 条异常记录")
        
        return data
    
    def _aggregate_data(self, data):
        """
        数据聚合
        """
        print_info("开始执行数据聚合...")
        aggregate_config = self.preprocessing_config["数据聚合"]
        print_info(f"聚合配置: {aggregate_config}")
        
        # 获取时间字段
        time_field = aggregate_config["时间字段"]
        if time_field == "auto":
            # 自动检测时间字段
            time_fields = [col for col in data.columns if any(keyword in col.lower() for keyword in ['time', 'date', '时间', '日期'])]
            if time_fields:
                time_field = time_fields[0]
                print_info(f"自动检测到时间字段: {time_field}")
            else:
                print_warning("未检测到时间字段，跳过聚合")
                print(f"  ❌ 聚合失败: 未找到时间字段")
                print(f"  💡 建议: 检查数据中是否包含时间相关字段，或手动指定时间字段")
                return data
        elif time_field not in data.columns:
            print_warning(f"时间字段 {time_field} 不存在，跳过聚合")
            print(f"  ❌ 聚合失败: 时间字段 '{time_field}' 不存在")
            print(f"  💡 建议: 检查时间字段名称是否正确，或使用 'auto' 自动检测")
            return data
        
        aggregate_method = aggregate_config["聚合方式"]
        aggregate_function = aggregate_config["聚合函数"]
        
        print_info(f"使用聚合方式: {aggregate_method}, 聚合函数: {aggregate_function}")
        
        try:
            # 确保时间字段是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(data[time_field]):
                data[time_field] = pd.to_datetime(data[time_field])
                print(f"  🔄 已转换时间字段 '{time_field}' 为datetime类型")
            
            # 设置时间索引
            data_temp = data.set_index(time_field)
            
            # 根据聚合方式选择重采样频率
            if aggregate_method == "daily":
                freq = "D"
            elif aggregate_method == "weekly":
                freq = "W"
            elif aggregate_method == "monthly":
                freq = "M"
            else:
                print_warning(f"不支持的聚合方式: {aggregate_method}，跳过聚合")
                print(f"  ❌ 聚合失败: 不支持的聚合方式 '{aggregate_method}'")
                print(f"  💡 支持的聚合方式: daily, weekly, monthly")
                return data
            
            # 选择聚合字段
            aggregate_columns = aggregate_config["聚合字段"]
            if aggregate_columns == "auto":
                # 自动检测数值列
                numeric_columns = data_temp.select_dtypes(include=[np.number]).columns
                exclude_columns = aggregate_config.get("排除字段", ['user_id'])
                aggregate_columns = [col for col in numeric_columns if col not in exclude_columns]
                print(f"  🔍 自动检测到 {len(aggregate_columns)} 个数值字段用于聚合")
            else:
                # 使用指定的聚合字段
                aggregate_columns = [col for col in aggregate_columns if col in data_temp.columns]
                print(f"  📋 使用指定的 {len(aggregate_columns)} 个字段进行聚合")
            
            print_info(f"聚合字段: {aggregate_columns}")
            
            if not aggregate_columns:
                print_warning("没有找到可聚合的数值列")
                print(f"  ❌ 聚合失败: 没有找到可聚合的数值列")
                print(f"  💡 建议: 检查数据中是否包含数值字段，或手动指定聚合字段")
                return data
            
            # 执行聚合
            print(f"  🚀 开始执行聚合操作...")
            if aggregate_function == "sum":
                aggregated = data_temp[aggregate_columns].resample(freq).sum()
            elif aggregate_function == "mean":
                aggregated = data_temp[aggregate_columns].resample(freq).mean()
            elif aggregate_function == "median":
                aggregated = data_temp[aggregate_columns].resample(freq).median()
            elif aggregate_function == "max":
                aggregated = data_temp[aggregate_columns].resample(freq).max()
            elif aggregate_function == "min":
                aggregated = data_temp[aggregate_columns].resample(freq).min()
            else:
                print_warning(f"不支持的聚合函数: {aggregate_function}，使用sum")
                aggregated = data_temp[aggregate_columns].resample(freq).sum()
            
            # 处理聚合后的缺失值
            if aggregate_config.get("处理缺失值", True):
                missing_fill = aggregate_config.get("缺失值填充", 0)
                aggregated = aggregated.fillna(missing_fill)
                print(f"  🔧 已处理聚合后的缺失值 (填充值: {missing_fill})")
            
            # 重置索引
            if aggregate_config.get("输出格式", {}).get("重置索引", True):
                aggregated = aggregated.reset_index()
                print(f"  🔄 已重置时间索引")
            
            # 输出详细聚合效果
            print("\n=================【数据聚合效果】=================")
            print(f"聚合方式: {aggregate_method} ({freq})    聚合函数: {aggregate_function}")
            print(f"聚合字段数量: {len(aggregate_columns)} 个")
            print(f"聚合字段名: {aggregate_columns}")
            print(f"聚合前数据量: {len(data):,} 条    聚合后数据量: {len(aggregated):,} 条")
            print(f"聚合时间范围: {aggregated[time_field].min() if time_field in aggregated.columns else aggregated.index.min()} ~ {aggregated[time_field].max() if time_field in aggregated.columns else aggregated.index.max()}")
            print("================================================\n")
            
            print(f"  ✅ 聚合操作完成!")
            return aggregated
        except Exception as e:
            print_error(f"数据聚合失败: {e}")
            print(f"  ❌ 聚合过程中发生错误: {e}")
            print(f"  💡 建议: 检查数据格式、时间字段格式、聚合配置等")
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
        
        # 2. 尝试加载现有配置
        data_source_name = self.data_file_path.stem if hasattr(self, 'data_file_path') else None
        config_loaded = self.load_processing_config(data_source_name)
        
        if config_loaded:
            print_info("使用现有配置文件进行处理")
        else:
            print_info("使用默认配置进行处理")
        
        # 3. 数据清洗
        if not self.clean_data():
            return False
        
        # 4. 特征工程
        if not self.engineer_features():
            return False
        
        # 5. 数据转换
        if not self.transform_data():
            return False
        
        # 6. 保存结果
        if not self.save_processed_data():
            return False
        
        # 7. 保存配置文件
        config_path = self.save_processing_config(data_source_name)
        
        # 8. 生成说明文档
        self.generate_data_folder_readme(data_source_name)
        
        print_success("完整数据处理流水线执行完成！")
        return True

    def save_processing_config(self, data_source_name=None):
        """
        保存数据处理配置到原始数据文件目录
        
        Args:
            data_source_name: 数据源名称，如果为None则自动生成
        """
        if data_source_name is None:
            # 从数据文件名生成数据源名称
            if hasattr(self, 'data_file_path') and self.data_file_path:
                data_source_name = self.data_file_path.stem
            else:
                data_source_name = "user_balance_table"
        
        config_file_path = DATA_DIR / f"{data_source_name}_preprocessing_config.json"
        
        try:
            # 生成配置信息
            config = {
                "data_source_name": data_source_name,
                "generated_time": datetime.now().isoformat(),
                "data_info": {
                    "original_shape": self.data.shape if self.data is not None else None,
                    "processed_shape": self.processed_data.shape if self.processed_data is not None else None,
                    "feature_count": len(self.processed_data.columns) if self.processed_data is not None else None
                },
                "preprocessing_config": {
                    "缺失值处理": self.preprocessing_config["缺失值处理"],
                    "异常值处理": self.preprocessing_config["异常值处理"],
                    "时间特征": self.preprocessing_config["时间特征"],
                    "数据聚合": self.preprocessing_config["数据聚合"],
                    "数据一致性处理": self.preprocessing_config["数据一致性处理"]
                },
                "feature_engineering_config": {
                    "基础特征": self.feature_config["基础特征"],
                    "时间特征": self.feature_config["时间特征"],
                    "统计特征": self.feature_config["统计特征"],
                    "用户特征": self.feature_config["用户特征"],
                    "业务特征": self.feature_config["业务特征"]
                },
                "data_transformation_config": {
                    "标准化": self.transformation_config["标准化"],
                    "编码": self.transformation_config["编码"],
                    "时间序列特征": self.transformation_config["时间序列特征"],
                    "特征选择": self.transformation_config["特征选择"]
                },
                "field_mapping": self.field_mapping,
                "processing_log": self.processing_log
            }
            
            # 保存配置文件
            write_json(config, config_file_path)
            print_success(f"数据处理配置已保存: {config_file_path}")
            
            return config_file_path
            
        except Exception as e:
            print_error(f"保存配置文件失败: {e}")
            return None
    
    def load_processing_config(self, data_source_name=None):
        """
        从原始数据文件目录加载数据处理配置
        
        Args:
            data_source_name: 数据源名称，如果为None则自动检测
            
        Returns:
            bool: 是否成功加载配置
        """
        if data_source_name is None:
            # 自动检测配置文件
            config_files = list(DATA_DIR.glob("*_preprocessing_config.json"))
            if not config_files:
                print_info("未找到预处理配置文件，将使用默认配置")
                return False
            
            # 使用最新的配置文件
            config_file_path = max(config_files, key=lambda x: x.stat().st_mtime)
            data_source_name = config_file_path.stem.replace("_preprocessing_config", "")
        else:
            config_file_path = DATA_DIR / f"{data_source_name}_preprocessing_config.json"
        
        if not config_file_path.exists():
            print_info(f"未找到配置文件: {config_file_path}")
            return False
        
        try:
            # 加载配置文件
            with open(config_file_path, 'r', encoding='utf-8') as f:
                import json
                config = json.load(f)
            
            # 更新配置
            if "preprocessing_config" in config:
                self.preprocessing_config.update(config["preprocessing_config"])
            
            if "feature_engineering_config" in config:
                self.feature_config.update(config["feature_engineering_config"])
            
            if "data_transformation_config" in config:
                self.transformation_config.update(config["data_transformation_config"])
            
            if "field_mapping" in config:
                self.field_mapping.update(config["field_mapping"])
            
            print_success(f"已加载预处理配置: {config_file_path}")
            print_info(f"数据源: {config.get('data_source_name', 'unknown')}")
            print_info(f"生成时间: {config.get('generated_time', 'unknown')}")
            
            return True
            
        except Exception as e:
            print_error(f"加载配置文件失败: {e}")
            return False
    
    def generate_data_folder_readme(self, data_source_name=None):
        """
        生成data文件夹说明文档
        
        Args:
            data_source_name: 数据源名称
        """
        if data_source_name is None:
            data_source_name = "user_balance_table"
        
        readme_path = DATA_DIR / "README.md"
        
        try:
            # 获取数据文件列表
            data_files = list(DATA_DIR.glob("*.csv"))
            config_files = list(DATA_DIR.glob("*_preprocessing_config.json"))
            
            # 生成README内容
            readme_content = f"""# Data 文件夹说明

## 概述
data文件夹用于存放原始数据文件和相关的配置文件。

## 文件结构

### 原始数据文件
"""
            
            for data_file in data_files:
                file_size = data_file.stat().st_size / (1024 * 1024)  # MB
                readme_content += f"- **{data_file.name}**: {file_size:.2f} MB\n"
            
            readme_content += """
### 配置文件
"""
            
            for config_file in config_files:
                readme_content += f"- **{config_file.name}**: 数据处理配置文件\n"
            
            readme_content += f"""
## 数据预处理配置

### 当前数据源: {data_source_name}

系统支持自动生成和加载数据处理配置，配置文件包含以下内容：

#### 1. 数据预处理配置
- **缺失值处理**: 定义各字段的缺失值填充策略
- **异常值处理**: 异常值检测和处理方法
- **时间特征**: 时间特征提取配置

#### 2. 特征工程配置
- **基础特征**: 净资金流、总资金流、资金流比率等
- **时间特征**: 年、月、日、星期、季度等时间特征
- **统计特征**: 滚动窗口统计特征
- **用户特征**: 用户级别统计和分类特征
- **业务特征**: 交易活跃度、资金流类型等

#### 3. 数据转换配置
- **标准化**: 数值特征标准化方法
- **编码**: 分类特征编码方法
- **时间序列特征**: 滞后特征配置
- **特征选择**: 特征筛选策略

## 使用方法

### 1. 首次处理数据
1. 将原始数据文件放入data文件夹
2. 运行数据预处理功能
3. 系统自动生成配置文件

### 2. 修改处理参数
1. 编辑对应的配置文件（*_preprocessing_config.json）
2. 重新运行数据预处理功能
3. 系统将使用修改后的配置

### 3. 配置文件格式
配置文件采用JSON格式，包含详细的处理参数和说明。

## 注意事项

1. 配置文件与原始数据文件关联，修改数据文件后需要重新生成配置
2. 建议在修改配置前备份原配置文件
3. 配置文件包含处理日志，便于追踪处理过程
4. 所有配置文件使用UTF-8编码

## 文件命名规则

- **原始数据**: `{data_source_name}.csv`
- **预处理配置**: `{data_source_name}_preprocessing_config.json`
- **数据源配置**: `{data_source_name}_dispose.json`（由基础数据分析生成）

## 更新记录

- 配置文件生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 最后更新: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
            
            # 保存README文件
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            print_success(f"Data文件夹说明文档已生成: {readme_path}")
            return True
            
        except Exception as e:
            print_error(f"生成说明文档失败: {e}")
            return False


def run_data_processing():
    """运行数据处理功能"""
    print_header("数据处理模块", "数据清洗、特征工程、数据转换")
    
    # 检查是否有特化配置的数据源
    data_files = list(DATA_DIR.glob("*/*.csv"))
    if data_files:
        print_info("检测到特化配置的数据源，使用通用数据处理器")
        from src.data_processing_universal import run_universal_data_processing
        success = run_universal_data_processing()
    else:
        print_info("未检测到特化配置，使用传统数据处理器")
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