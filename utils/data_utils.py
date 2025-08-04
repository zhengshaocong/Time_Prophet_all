# -*- coding: utf-8 -*-
"""
数据处理工具模块
提供数据清洗、转换、验证等功能
"""

import pandas as pd
from typing import Any, Dict, List, Union, Optional
from pathlib import Path

def clean_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """清洗数据，移除空值和无效数据"""
    cleaned_data = []
    
    for item in data:
        # 移除所有值为None或空字符串的键
        cleaned_item = {k: v for k, v in item.items() 
                       if v is not None and v != ""}
        
        if cleaned_item:  # 只保留非空的数据项
            cleaned_data.append(cleaned_item)
    
    return cleaned_data

def validate_data(data: List[Dict[str, Any]], required_fields: List[str]) -> bool:
    """验证数据是否包含必需字段"""
    if not data:
        return False
    
    for item in data:
        for field in required_fields:
            if field not in item or item[field] is None:
                return False
    
    return True

def convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """将字典列表转换为DataFrame"""
    if not data:
        return pd.DataFrame()
    
    return pd.DataFrame(data)

def dataframe_to_dict_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """将DataFrame转换为字典列表"""
    if df.empty:
        return []
    
    return df.to_dict('records')

def filter_data(data: List[Dict[str, Any]], 
                field: str, 
                value: Any, 
                operator: str = "==") -> List[Dict[str, Any]]:
    """根据条件过滤数据"""
    filtered_data = []
    
    for item in data:
        if field not in item:
            continue
        
        item_value = item[field]
        
        # 根据操作符进行过滤
        if operator == "==" and item_value == value:
            filtered_data.append(item)
        elif operator == "!=" and item_value != value:
            filtered_data.append(item)
        elif operator == ">" and item_value > value:
            filtered_data.append(item)
        elif operator == "<" and item_value < value:
            filtered_data.append(item)
        elif operator == ">=" and item_value >= value:
            filtered_data.append(item)
        elif operator == "<=" and item_value <= value:
            filtered_data.append(item)
        elif operator == "in" and item_value in value:
            filtered_data.append(item)
        elif operator == "not in" and item_value not in value:
            filtered_data.append(item)
    
    return filtered_data

def sort_data(data: List[Dict[str, Any]], 
              field: str, 
              reverse: bool = False) -> List[Dict[str, Any]]:
    """根据字段排序数据"""
    if not data or field not in data[0]:
        return data
    
    return sorted(data, key=lambda x: x.get(field, ""), reverse=reverse)

def get_unique_values(data: List[Dict[str, Any]], field: str) -> List[Any]:
    """获取指定字段的唯一值"""
    if not data or field not in data[0]:
        return []
    
    unique_values = set()
    for item in data:
        if field in item and item[field] is not None:
            unique_values.add(item[field])
    
    return list(unique_values)

def count_by_field(data: List[Dict[str, Any]], field: str) -> Dict[Any, int]:
    """统计指定字段的值出现次数"""
    if not data or field not in data[0]:
        return {}
    
    count_dict = {}
    for item in data:
        if field in item and item[field] is not None:
            value = item[field]
            count_dict[value] = count_dict.get(value, 0) + 1
    
    return count_dict

def merge_data(data1: List[Dict[str, Any]], 
               data2: List[Dict[str, Any]], 
               key_field: str) -> List[Dict[str, Any]]:
    """根据关键字段合并两个数据集"""
    if not data1 or not data2:
        return data1 or data2
    
    # 创建第二个数据的查找字典
    data2_dict = {item[key_field]: item for item in data2 
                  if key_field in item}
    
    merged_data = []
    for item1 in data1:
        if key_field in item1 and item1[key_field] in data2_dict:
            # 合并数据，data2的字段会覆盖data1的同名字段
            merged_item = {**item1, **data2_dict[item1[key_field]]}
            merged_data.append(merged_item)
        else:
            merged_data.append(item1)
    
    return merged_data

def sample_data(data: List[Dict[str, Any]], 
                sample_size: int, 
                random_seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """随机采样数据"""
    if not data:
        return []
    
    import random
    if random_seed is not None:
        random.seed(random_seed)
    
    if sample_size >= len(data):
        return data
    
    return random.sample(data, sample_size)

def split_data(data: List[Dict[str, Any]], 
               split_ratio: float = 0.8) -> tuple:
    """分割数据集为训练集和测试集"""
    if not data:
        return [], []
    
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    return train_data, test_data 