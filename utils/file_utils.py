# -*- coding: utf-8 -*-
"""
文件操作工具模块
提供文件读写、路径处理等功能
"""

import os
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Union

def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """读取JSON文件"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """写入JSON文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 处理numpy数据类型
    def convert_numpy_types(obj):
        """递归转换numpy数据类型为Python原生类型"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # 转换数据类型
    converted_data = convert_numpy_types(data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=indent)

def read_csv(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    """读取CSV文件"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    return data

def write_csv(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """写入CSV文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data:
        return
    
    fieldnames = data[0].keys()
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def read_text(file_path: Union[str, Path]) -> str:
    """读取文本文件"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_text(content: str, file_path: Union[str, Path]) -> None:
    """写入文本文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def get_file_extension(file_path: Union[str, Path]) -> str:
    """获取文件扩展名"""
    return Path(file_path).suffix.lower()

def is_supported_format(file_path: Union[str, Path], supported_formats: List[str]) -> bool:
    """检查文件格式是否支持"""
    extension = get_file_extension(file_path)
    return extension in supported_formats

def clean_filename(filename: str) -> str:
    """清理文件名，移除非法字符"""
    import re
    # 移除或替换非法字符
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除多余的空格和点
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def get_file_size(file_path: Union[str, Path]) -> int:
    """获取文件大小（字节）"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    return file_path.stat().st_size

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}" 