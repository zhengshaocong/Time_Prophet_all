# -*- coding: utf-8 -*-
"""
通用数据预处理主程序
支持数据源特化架构，主程序只负责流程控制，具体处理逻辑由特化模块实现
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, OUTPUT_DATA_DIR
from utils.interactive_utils import print_header, print_success, print_error, print_info, print_warning
from utils.file_utils import write_csv, write_json


class UniversalDataProcessor:
    """通用数据处理器"""
    
    def __init__(self):
        """初始化通用数据处理器"""
        self.data = None
        self.processed_data = None
        self.data_source_name = None
        self.config = None
        self.processing_log = []
        
    def detect_data_source(self, file_path):
        """
        检测数据源
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            str: 数据源名称
        """
        file_path = Path(file_path)
        data_source_name = file_path.stem
        
        # 检查是否存在对应的特化配置
        config_path = DATA_DIR / data_source_name / "config.json"
        if config_path.exists():
            print_success(f"检测到数据源: {data_source_name}")
            return data_source_name
        else:
            print_warning(f"未找到数据源 {data_source_name} 的特化配置")
            return None
    
    def load_specialized_config(self, data_source_name):
        """
        加载特化配置
        
        Args:
            data_source_name: 数据源名称
            
        Returns:
            bool: 是否成功加载配置
        """
        config_path = DATA_DIR / data_source_name / "config.json"
        
        if not config_path.exists():
            print_error(f"配置文件不存在: {config_path}")
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            print_success(f"已加载特化配置: {config_path}")
            print_info(f"数据源: {self.config.get('data_source_name', 'unknown')}")
            print_info(f"描述: {self.config.get('description', '无描述')}")
            
            return True
            
        except Exception as e:
            print_error(f"加载配置文件失败: {e}")
            return False
    
    def load_specialized_modules(self, data_source_name):
        """
        加载特化模块
        
        Args:
            data_source_name: 数据源名称
            
        Returns:
            tuple: (特征工程器, 异常检测器)
        """
        try:
            # 动态导入特化模块
            module_path = DATA_DIR / data_source_name
            
            # 添加模块路径到sys.path
            if str(module_path) not in sys.path:
                sys.path.insert(0, str(module_path))
            
            # 导入特化模块
            from features import FeatureEngineer
            from anomalies import AnomalyDetector
            
            # 创建特化处理器
            feature_engineer = FeatureEngineer(self.config)
            anomaly_detector = AnomalyDetector(self.config)
            
            print_success(f"已加载特化模块: {data_source_name}")
            return feature_engineer, anomaly_detector
            
        except ImportError as e:
            print_error(f"导入特化模块失败: {e}")
            print_info("请确保特化模块文件存在且格式正确")
            return None, None
        except Exception as e:
            print_error(f"加载特化模块失败: {e}")
            return None, None
    
    def load_data(self, file_path):
        """
        加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            bool: 是否成功
        """
        print_header("数据加载")
        
        try:
            # 检测数据源
            self.data_source_name = self.detect_data_source(file_path)
            if not self.data_source_name:
                return False
            
            # 加载特化配置
            if not self.load_specialized_config(self.data_source_name):
                return False
            
            # 加载数据
            print_info(f"加载数据文件: {file_path}")
            self.data = pd.read_csv(file_path)
            
            print_success(f"数据加载完成: {len(self.data):,} 条记录, {len(self.data.columns)} 个字段")
            
            # 记录处理日志
            self.processing_log.append({
                "步骤": "数据加载",
                "数据源": self.data_source_name,
                "记录数": len(self.data),
                "字段数": len(self.data.columns),
                "时间": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print_error(f"数据加载失败: {e}")
            return False
    
    def basic_data_cleaning(self):
        """
        基础数据清洗（通用逻辑）
        
        Returns:
            bool: 是否成功
        """
        print_header("基础数据清洗")
        
        if self.data is None:
            print_error("请先加载数据")
            return False
        
        try:
            data = self.data.copy()
            original_count = len(data)
            
            # 1. 处理缺失值
            print_info("处理缺失值...")
            missing_counts = data.isnull().sum()
            if missing_counts.sum() > 0:
                print(f"  发现缺失值:")
                for col, count in missing_counts[missing_counts > 0].items():
                    print(f"    {col}: {count} 个")
                
                # 数值字段用0填充，字符串字段用空字符串填充
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                string_columns = data.select_dtypes(include=['object']).columns
                
                data[numeric_columns] = data[numeric_columns].fillna(0)
                data[string_columns] = data[string_columns].fillna('')
                
                print("  已填充缺失值")
            else:
                print("  无缺失值")
            
            # 2. 数据类型转换
            print_info("转换数据类型...")
            field_mapping = self.config.get("field_mapping", {})
            
            # 转换时间字段
            time_field = field_mapping.get("时间字段")
            if time_field and time_field in data.columns:
                time_format = field_mapping.get("时间格式", "%Y%m%d")
                try:
                    data[time_field] = pd.to_datetime(data[time_field], format=time_format)
                    print(f"  已转换时间字段: {time_field}")
                except Exception as e:
                    print_warning(f"时间字段转换失败: {e}")
            
            # 3. 时间范围过滤
            print_info("应用时间范围过滤...")
            data = self._apply_time_range_filter(data)
            
            # 4. 转换数值字段
            numeric_fields = self.config.get("data_validation", {}).get("数值字段", [])
            for field in numeric_fields:
                if field in data.columns:
                    try:
                        data[field] = pd.to_numeric(data[field], errors='coerce')
                        data[field] = data[field].fillna(0)
                        print(f"  已转换数值字段: {field}")
                    except Exception as e:
                        print_warning(f"数值字段转换失败 {field}: {e}")
            
            # 5. 数据聚合
            data = self._aggregate_data(data)
            
            self.data = data
            cleaned_count = len(data)
            
            print_success(f"基础数据清洗完成: {original_count:,} -> {cleaned_count:,} 条")
            
            self.processing_log.append({
                "步骤": "基础数据清洗",
                "原始数据量": original_count,
                "清洗后数据量": cleaned_count,
                "时间": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print_error(f"基础数据清洗失败: {e}")
            return False
    
    def _apply_time_range_filter(self, data):
        """
        应用时间范围过滤
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        # 获取时间范围配置
        time_range_config = self.config.get("data_preprocessing", {}).get("时间范围限制", {})
        
        if not time_range_config.get("启用时间范围限制", False):
            print_info("时间范围限制未启用，跳过过滤")
            return data
        
        field_mapping = self.config.get("field_mapping", {})
        time_field = field_mapping.get("时间字段")
        
        if not time_field or time_field not in data.columns:
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
            config_start = pd.to_datetime(time_range_config.get("开始日期", "2014-01-01"))
            config_end = pd.to_datetime(time_range_config.get("结束日期", "2014-12-31"))
            
            # 检查配置的时间范围是否超出原始数据范围
            if time_range_config.get("超出范围警告", True):
                if config_start < original_start:
                    print_warning(f"配置的开始日期 {config_start.date()} 早于数据最早日期 {original_start.date()}")
                if config_end > original_end:
                    print_warning(f"配置的结束日期 {config_end.date()} 晚于数据最晚日期 {original_end.date()}")
            
            # 自动调整时间范围
            if time_range_config.get("自动调整", False):
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
    
    def _aggregate_data(self, data):
        """
        数据聚合
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 聚合后的数据
        """
        # 检查是否有聚合配置
        aggregate_config = self.config.get("data_preprocessing", {}).get("数据聚合", {})
        
        if not aggregate_config.get("启用聚合", False):
            print_info("跳过数据聚合 (聚合功能未启用)")
            print(f"  ⏭️  聚合功能已禁用，保持原始数据量: {len(data):,} 条")
            return data
        
        print_info("开始执行数据聚合...")
        
        # 获取聚合配置
        time_field = aggregate_config.get("时间字段", "auto")
        aggregate_method = aggregate_config.get("聚合方式", "daily")
        aggregate_function = aggregate_config.get("聚合函数", "sum")
        
        print(f"  🔄 聚合功能已启用")
        print(f"  📅 聚合方式: {aggregate_method}")
        print(f"  🧮 聚合函数: {aggregate_function}")
        print(f"  🕒 时间字段: {time_field}")
        
        # 自动检测时间字段
        if time_field == "auto":
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
            aggregate_columns = aggregate_config.get("聚合字段", "auto")
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
            
            # 聚合后的提示
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
    
    def specialized_processing(self):
        """
        特化处理（调用特化模块）
        
        Returns:
            bool: 是否成功
        """
        print_header("特化处理")
        
        if self.data is None:
            print_error("请先完成基础数据清洗")
            return False
        
        try:
            # 加载特化模块
            feature_engineer, anomaly_detector = self.load_specialized_modules(self.data_source_name)
            if feature_engineer is None or anomaly_detector is None:
                return False
            
            # 1. 异常检测
            print_info("执行异常检测...")
            clean_data, anomalies = anomaly_detector.detect_anomalies(self.data)
            
            # 保存异常检测结果到实例变量
            self.anomalies = anomalies
            
            # 显示异常检测结果
            anomaly_summary = anomaly_detector.get_anomaly_summary(anomalies)
            print(f"  异常检测完成: {anomaly_summary.get('总异常记录数', 0)} 条异常记录")
            
            # 2. 特征工程
            print_info("执行特征工程...")
            data_with_features = feature_engineer.engineer_features(clean_data)
            
            # 显示特征工程结果
            original_columns = len(self.data.columns)
            new_columns = len(data_with_features.columns)
            print(f"  特征工程完成: {original_columns} -> {new_columns} 列")
            print(f"  新增特征: {new_columns - original_columns} 个")
            
            self.processed_data = data_with_features
            
            # 记录处理日志
            self.processing_log.append({
                "步骤": "特化处理",
                "异常检测": {
                    "异常类型数": anomaly_summary.get("总异常类型数", 0),
                    "异常记录数": anomaly_summary.get("总异常记录数", 0)
                },
                "特征工程": {
                    "原始列数": original_columns,
                    "新列数": new_columns,
                    "新增特征数": new_columns - original_columns
                },
                "时间": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print_error(f"特化处理失败: {e}")
            return False
    
    def save_results(self, output_path=None):
        """
        保存处理结果
        
        Args:
            output_path: 输出路径
            
        Returns:
            bool: 是否成功
        """
        if self.processed_data is None:
            print_error("没有处理后的数据可保存")
            return False
        
        if output_path is None:
            # 使用覆盖模式，不添加时间戳
            output_path = OUTPUT_DATA_DIR / f"{self.data_source_name}_processed.csv"
        
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存数据
            self.processed_data.to_csv(output_path, index=False, encoding='utf-8')
            print_success(f"处理后的数据已保存: {output_path}")
            
            # 保存处理日志
            log_path = output_path.parent / f"{self.data_source_name}_processing_log.json"
            if write_json(self.processing_log, log_path):
                print_info(f"处理日志已保存: {log_path}")
            else:
                print_warning(f"处理日志保存失败: {log_path}")
            
            # 保存异常检测结果
            if hasattr(self, 'anomalies') and self.anomalies:
                anomaly_path = output_path.parent / f"{self.data_source_name}_anomalies.json"
                if write_json(self.anomalies, anomaly_path):
                    print_info(f"异常检测结果已保存: {anomaly_path}")
                else:
                    print_warning(f"异常检测结果保存失败: {anomaly_path}")
            
            return True
            
        except Exception as e:
            print_error(f"保存结果失败: {e}")
            return False
    
    def run_full_pipeline(self, file_path):
        """
        运行完整的数据处理流水线
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            bool: 是否成功
        """
        print_header("通用数据处理流水线", "数据加载 -> 基础清洗 -> 特化处理 -> 结果保存")
        
        # 1. 加载数据
        if not self.load_data(file_path):
            return False
        
        # 2. 基础数据清洗
        if not self.basic_data_cleaning():
            return False
        
        # 3. 特化处理
        if not self.specialized_processing():
            return False
        
        # 4. 保存结果
        if not self.save_results():
            return False
        
        print_success("完整数据处理流水线执行完成！")
        print(f"处理日志包含 {len(self.processing_log)} 个步骤")
        
        return True
    
    def get_processing_summary(self):
        """
        获取处理摘要
        
        Returns:
            dict: 处理摘要
        """
        if not self.processing_log:
            return {}
        
        return {
            "数据源": self.data_source_name,
            "处理步骤数": len(self.processing_log),
            "原始数据量": self.processing_log[0].get("记录数", 0) if self.processing_log else 0,
            "最终数据量": len(self.processed_data) if self.processed_data is not None else 0,
            "最终特征数": len(self.processed_data.columns) if self.processed_data is not None else 0,
            "处理时间": self.processing_log[-1].get("时间", "") if self.processing_log else ""
        }


def run_universal_data_processing(file_path=None):
    """运行通用数据处理功能"""
    print_header("通用数据处理模块", "支持数据源特化架构")
    
    if file_path is None:
        # 自动查找数据文件
        data_files = list(DATA_DIR.glob("*/*.csv"))
        if not data_files:
            print_error("未找到数据文件")
            return False
        
        # 使用第一个找到的数据文件
        file_path = data_files[0]
        print_info(f"自动选择数据文件: {file_path}")
    
    # 创建通用数据处理器
    processor = UniversalDataProcessor()
    
    # 运行完整流水线
    success = processor.run_full_pipeline(file_path)
    
    if success:
        # 显示处理摘要
        summary = processor.get_processing_summary()
        print("\n" + "="*50)
        print("处理摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("="*50)
    else:
        print_error("数据处理失败！")
    
    return success


if __name__ == "__main__":
    run_universal_data_processing() 