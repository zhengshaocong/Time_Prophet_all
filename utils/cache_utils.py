# -*- coding: utf-8 -*-
"""
缓存工具模块
提供字段映射缓存的管理功能
"""

import json
from pathlib import Path
from config import CACHE_DIR, FIELD_MAPPING_CACHE_FILE


def save_field_mapping_cache(field_mapping):
    """
    保存字段映射到缓存文件
    
    Args:
        field_mapping: 字段映射字典
        
    Returns:
        bool: 是否保存成功
    """
    try:
        # 确保缓存目录存在
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 保存到缓存文件
        with open(FIELD_MAPPING_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(field_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 字段映射已保存到缓存: {FIELD_MAPPING_CACHE_FILE}")
        return True
    except Exception as e:
        print(f"✗ 保存字段映射缓存失败: {e}")
        return False


def load_field_mapping_cache():
    """
    从缓存文件加载字段映射
    
    Returns:
        dict: 字段映射字典，如果缓存不存在则返回None
    """
    try:
        if not FIELD_MAPPING_CACHE_FILE.exists():
            return None
        
        with open(FIELD_MAPPING_CACHE_FILE, 'r', encoding='utf-8') as f:
            field_mapping = json.load(f)
        
        print(f"✓ 已从缓存加载字段映射: {FIELD_MAPPING_CACHE_FILE}")
        return field_mapping
    except Exception as e:
        print(f"✗ 加载字段映射缓存失败: {e}")
        return None


def clear_field_mapping_cache():
    """
    清除字段映射缓存
    
    Returns:
        bool: 是否清除成功
    """
    try:
        if FIELD_MAPPING_CACHE_FILE.exists():
            FIELD_MAPPING_CACHE_FILE.unlink()
            print(f"✓ 已清除字段映射缓存: {FIELD_MAPPING_CACHE_FILE}")
        return True
    except Exception as e:
        print(f"✗ 清除字段映射缓存失败: {e}")
        return False


def has_field_mapping_cache():
    """
    检查是否存在字段映射缓存
    
    Returns:
        bool: 是否存在缓存
    """
    return FIELD_MAPPING_CACHE_FILE.exists()


def get_cache_info():
    """
    获取缓存信息
    
    Returns:
        dict: 缓存信息字典
    """
    cache_info = {
        "缓存目录": str(CACHE_DIR),
        "字段映射缓存文件": str(FIELD_MAPPING_CACHE_FILE),
        "字段映射缓存存在": has_field_mapping_cache()
    }
    
    if has_field_mapping_cache():
        try:
            field_mapping = load_field_mapping_cache()
            cache_info["字段映射字段数"] = len(field_mapping) if field_mapping else 0
        except:
            cache_info["字段映射字段数"] = 0
    
    return cache_info


def list_cache_files():
    """
    列出缓存目录中的所有文件
    
    Returns:
        list: 缓存文件列表
    """
    if not CACHE_DIR.exists():
        return []
    
    cache_files = []
    for file_path in CACHE_DIR.iterdir():
        if file_path.is_file():
            cache_files.append({
                "文件名": file_path.name,
                "文件路径": str(file_path),
                "文件大小": file_path.stat().st_size,
                "修改时间": file_path.stat().st_mtime
            })
    
    return cache_files


def clear_all_cache():
    """
    清除所有缓存文件
    
    Returns:
        bool: 是否清除成功
    """
    try:
        if not CACHE_DIR.exists():
            return True
        
        cleared_count = 0
        for file_path in CACHE_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
                cleared_count += 1
        
        print(f"✓ 已清除 {cleared_count} 个缓存文件")
        return True
    except Exception as e:
        print(f"✗ 清除缓存失败: {e}")
        return False 