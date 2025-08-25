# -*- coding: utf-8 -*-
"""
文件工具模块
提供文件读写、CSV处理、文件命名等工具函数
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from datetime import datetime


def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """读取JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return {}


def write_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """写入JSON文件"""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 先尝试序列化数据，确保没有序列化问题
        try:
            json_str = json.dumps(data, ensure_ascii=False, indent=indent, default=str)
        except Exception as serialize_error:
            print(f"JSON序列化失败: {serialize_error}")
            print(f"数据类型: {type(data)}")
            # 尝试使用默认序列化器
            try:
                json_str = json.dumps(data, ensure_ascii=False, indent=indent, default=str)
            except Exception as e2:
                print(f"使用默认序列化器也失败: {e2}")
                return False
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        # 验证写入是否成功
        if file_path.exists() and file_path.stat().st_size > 0:
            return True
        else:
            print(f"文件写入验证失败: {file_path}")
            return False
            
    except Exception as e:
        print(f"写入JSON文件失败: {e}")
        print(f"文件路径: {file_path}")
        print(f"数据类型: {type(data)}")
        return False


def read_csv(file_path: Union[str, Path], encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """读取CSV文件"""
    try:
        data = []
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return []


def write_csv(data: List[Dict[str, Any]], file_path: Union[str, Path], 
              encoding: str = 'utf-8', fieldnames: List[str] = None) -> bool:
    """写入CSV文件"""
    try:
        if not data:
            print("警告: 没有数据可写入")
            return False
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(file_path, 'w', encoding=encoding, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        return True
    except Exception as e:
        print(f"写入CSV文件失败: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


# 文件命名工具
class FileNamingManager:
    """文件命名管理器"""
    
    def __init__(self, mode: str = "overwrite"):
        self.mode = mode
    
    def get_filename(self, base_name: str, extension: str, 
                    directory: Optional[Union[str, Path]] = None,
                    prefix: str = "", suffix: str = "") -> Path:
        if self.mode == "overwrite":
            return self._get_overwrite_filename(base_name, extension, directory, prefix, suffix)
        elif self.mode == "version":
            return self._get_version_filename(base_name, extension, directory, prefix, suffix)
        elif self.mode == "timestamp":
            return self._get_timestamp_filename(base_name, extension, directory, prefix, suffix)
        else:
            raise ValueError(f"不支持的命名模式: {self.mode}")
    
    def _get_overwrite_filename(self, base_name: str, extension: str,
                               directory: Optional[Union[str, Path]] = None,
                               prefix: str = "", suffix: str = "") -> Path:
        filename = f"{prefix}{base_name}{suffix}{extension}"
        if directory:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            return directory / filename
        else:
            return Path(filename)
    
    def _get_version_filename(self, base_name: str, extension: str,
                             directory: Optional[Union[str, Path]] = None,
                             prefix: str = "", suffix: str = "") -> Path:
        if directory:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory = Path(".")
        
        pattern = f"{prefix}{base_name}{suffix}_v*{extension}"
        existing_files = list(directory.glob(pattern))
        
        if not existing_files:
            version = 1
        else:
            versions = []
            for file in existing_files:
                try:
                    name_parts = file.stem.split('_v')
                    if len(name_parts) > 1:
                        version_str = name_parts[-1]
                        if version_str.isdigit():
                            versions.append(int(version_str))
                except:
                    continue
            version = max(versions) + 1 if versions else 1
        
        filename = f"{prefix}{base_name}{suffix}_v{version}{extension}"
        return directory / filename
    
    def _get_timestamp_filename(self, base_name: str, extension: str,
                               directory: Optional[Union[str, Path]] = None,
                               prefix: str = "", suffix: str = "") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}{base_name}{suffix}_{timestamp}{extension}"
        
        if directory:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            return directory / filename
        else:
            return Path(filename)


# 全局文件命名管理器实例
file_manager = FileNamingManager(mode="overwrite")


def get_filename(base_name: str, extension: str, 
                directory: Optional[Union[str, Path]] = None,
                prefix: str = "", suffix: str = "",
                mode: str = "overwrite") -> Path:
    if mode != file_manager.mode:
        file_manager.mode = mode
    return file_manager.get_filename(base_name, extension, directory, prefix, suffix)


def set_naming_mode(mode: str):
    file_manager.mode = mode


def get_overwrite_filename(base_name: str, extension: str,
                          directory: Optional[Union[str, Path]] = None,
                          prefix: str = "", suffix: str = "") -> Path:
    return file_manager._get_overwrite_filename(base_name, extension, directory, prefix, suffix)


def get_version_filename(base_name: str, extension: str,
                        directory: Optional[Union[str, Path]] = None,
                        prefix: str = "", suffix: str = "") -> Path:
    return file_manager._get_version_filename(base_name, extension, directory, prefix, suffix)


def get_timestamp_filename(base_name: str, extension: str,
                          directory: Optional[Union[str, Path]] = None,
                          prefix: str = "", suffix: str = "") -> Path:
    return file_manager._get_timestamp_filename(base_name, extension, directory, prefix, suffix)
