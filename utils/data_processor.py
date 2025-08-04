# -*- coding: utf-8 -*-
"""
数据处理工具函数
包含数据加载、预处理等可复用功能
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from config import DATA_DIR, CURRENT_DATA_SOURCE
from utils.config_utils import (
    get_data_field_mapping, get_field_name, 
    get_time_format, get_preprocessing_config,
    check_data_source_dispose_config, get_missing_dispose_config_message
)
from utils.interactive_utils import print_success, print_error, print_info, print_warning

class DataProcessor:
    """数据处理基类"""
    
    def __init__(self):
        """初始化数据处理器"""
        self.data = None
        self.data_info = {}
        
    def load_data(self, file_path=None, use_data_processing=False, module_name=None):
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径，默认为data/user_balance_table.csv
            use_data_processing: 是否使用数据处理管理器
            module_name: 模块名称，用于数据处理管理器
            
        Returns:
            bool: 是否加载成功
        """
        if file_path is None:
            file_path = DATA_DIR / "user_balance_table.csv"
        
        # 如果启用了数据处理管理器
        if use_data_processing and module_name:
            try:
                from utils.data_processing_manager import get_data_for_module, should_process_data
                
                if should_process_data(module_name):
                    print_info(f"模块 {module_name} 启用数据处理，正在处理数据...")
                    processed_data = get_data_for_module(module_name)
                    if processed_data is not None:
                        self.data = processed_data
                        print_success(f"使用处理后的数据，形状: {self.data.shape}")
                        return True
                    else:
                        print_warning("数据处理失败，使用原始数据")
                else:
                    print_info(f"模块 {module_name} 未启用数据处理，使用原始数据")
            except ImportError:
                print_warning("数据处理管理器导入失败，使用原始数据")
            except Exception as e:
                print_warning(f"数据处理管理器出错: {e}，使用原始数据")
        
        # 使用原始数据
        try:
            self.data = pd.read_csv(file_path)
            print_success(f"数据加载成功: {file_path}")
            print(f"数据形状: {self.data.shape}")
            return True
        except Exception as e:
            print_error(f"数据加载失败: {e}")
            return False
    
    def preprocess_data(self, data_source=None):
        """
        预处理数据
        
        Args:
            data_source: 数据源名称，如果为None则自动检测
            
        Returns:
            bool: 是否预处理成功
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
            
        try:
            # 获取字段映射配置（会自动检测数据源）
            field_mapping = get_data_field_mapping(data_source)
            preprocessing_config = get_preprocessing_config()
            
            # 检查必要的字段是否存在
            required_fields = ["时间字段", "申购金额字段", "赎回金额字段", "当前余额字段", "昨日余额字段"]
            missing_fields = []
            for field_type in required_fields:
                if field_type not in field_mapping:
                    missing_fields.append(field_type)
            
            if missing_fields:
                print_error(f"缺少必要的字段映射: {', '.join(missing_fields)}")
                print_error("请先运行基础数据分析生成数据源配置文件")
                return False
            
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
            # 自动检测数据源名称用于显示
            dispose_files = list(DATA_DIR.glob("*_dispose.json"))
            detected_source = dispose_files[0].stem.replace("_dispose", "") if dispose_files else "unknown"
            print(f"数据源: {data_source or detected_source}")
            print(f"总记录数: {len(self.data)}")
            print(f"有申购记录数: {len(self.data[self.data['Purchase_Amount'] > 0])}")
            print(f"有赎回记录数: {len(self.data[self.data['Redemption_Amount'] > 0])}")
            print(f"数据时间范围: {self.data[time_field].min()} 到 {self.data[time_field].max()}")
            
            return True
            
        except Exception as e:
            print_error(f"数据预处理失败: {e}")
            return False
    
    def analyze_data_structure(self, top_rows=5):
        """
        分析数据结构并解析字段
        
        Args:
            top_rows: 显示前几行数据
            
        Returns:
            bool: 是否分析成功
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
            
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