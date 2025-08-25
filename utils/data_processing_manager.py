# -*- coding: utf-8 -*-
"""
数据处理管理器
用于在各个功能模块中统一管理数据处理
"""

import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

from config import (
    GLOBAL_DATA_PROCESSING_CONFIG, DATA_DIR, OUTPUT_DATA_DIR, CACHE_DIR,
    BASIC_ANALYSIS_CONFIG, MODULE_DATA_PREPROCESSING_CONFIG, 
    ARIMA_PREDICTION_CONFIG, ARIMA_TRAINING_CONFIG
)
from utils.interactive_utils import print_info, print_success, print_warning, print_error
from src.data_processing import DataProcessingPipeline


class DataProcessingManager:
    """数据处理管理器"""
    
    def __init__(self):
        """初始化数据处理管理器"""
        self.global_config = GLOBAL_DATA_PROCESSING_CONFIG
        self.cache_file = CACHE_DIR / "processed_data_cache.json"
        self.processed_data_cache = self._load_cache()
        self.pipeline = None
    
    def _load_cache(self):
        """加载处理数据缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print_warning(f"加载缓存失败: {e}")
        return {}
    
    def _save_cache(self):
        """保存处理数据缓存"""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_data_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print_warning(f"保存缓存失败: {e}")
    
    def get_module_config(self, module_name):
        """
        获取指定模块的数据处理配置
        
        Args:
            module_name: 模块名称
            
        Returns:
            dict: 模块配置
        """
        config_map = {
            "basic_analysis": BASIC_ANALYSIS_CONFIG,
            "data_preprocessing": MODULE_DATA_PREPROCESSING_CONFIG,
            "arima_prediction": ARIMA_PREDICTION_CONFIG,
            "classical_decomposition": __import__('config', fromlist=['modules']).modules.CLASSICAL_DECOMPOSITION_PREDICTION_CONFIG
        }
        
        return config_map.get(module_name, {})
    
    def should_process_data(self, module_name):
        """
        判断是否需要处理数据
        
        Args:
            module_name: 模块名称
            
        Returns:
            bool: 是否需要处理数据
        """
        # 检查全局配置
        if not self.global_config["启用数据处理"]:
            return False
        
        # 检查模块配置
        module_config = self.get_module_config(module_name)
        if not module_config.get("数据处理", {}).get("启用数据处理", True):
            return False
        
        return True
    
    def get_processed_data_path(self, module_name=None):
        """
        获取处理后数据的路径
        
        Args:
            module_name: 模块名称
            
        Returns:
            str: 处理后数据路径
        """
        # 检查缓存
        if module_name in self.processed_data_cache:
            cache_info = self.processed_data_cache[module_name]
            if cache_info.get("exists", False):
                return cache_info["path"]
        
        # 查找最新的处理后数据文件（注意：OUTPUT_DATA_DIR 已是 output/data）
        output_dir = OUTPUT_DATA_DIR
        if output_dir.exists():
            # 放宽匹配规则，兼容常见命名：processed_data_*.csv、*_processed_*.csv、*_processed.csv
            candidates = []
            candidates.extend(list(output_dir.glob("processed_data_*.csv")))
            candidates.extend(list(output_dir.glob("*_processed_*.csv")))
            candidates.extend(list(output_dir.glob("*_processed.csv")))
            # 进一步兼容任意包含 processed 的命名
            candidates.extend(list(output_dir.glob("*processed*.csv")))
            # 去重
            candidates = list({p: None for p in candidates}.keys())
            if candidates:
                latest_file = max(candidates, key=lambda x: x.stat().st_mtime)
                return str(latest_file)
        
        return None
    
    def check_data_quality(self, data, module_name):
        """
        检查数据质量
        
        Args:
            data: 数据
            module_name: 模块名称
            
        Returns:
            bool: 数据质量是否满足要求
        """
        module_config = self.get_module_config(module_name)
        data_quality_config = module_config.get("数据处理", {}).get("数据质量", {})
        
        if not data_quality_config:
            return True
        
        # 检查数据完整度
        min_completeness = data_quality_config.get("最小完整度", 0.8)
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        if completeness < min_completeness:
            print_warning(f"数据完整度不足: {completeness:.2f} < {min_completeness}")
            return False
        
        # 检查缺失值比例
        max_missing_ratio = data_quality_config.get("最大缺失值比例", 0.2)
        missing_ratio = data.isnull().sum().max() / len(data)
        if missing_ratio > max_missing_ratio:
            print_warning(f"缺失值比例过高: {missing_ratio:.2f} > {max_missing_ratio}")
            return False
        
        # 检查数据量
        min_data_size = data_quality_config.get("最小数据量", 100)
        if len(data) < min_data_size:
            print_warning(f"数据量不足: {len(data)} < {min_data_size}")
            return False
        
        print_success("数据质量检查通过")
        return True
    
    def process_data_for_module(self, module_name, data=None, force_reprocess=False):
        """
        为指定模块处理数据
        
        Args:
            module_name: 模块名称
            data: 输入数据，如果为None则从文件加载
            force_reprocess: 是否强制重新处理
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if not self.should_process_data(module_name):
            print_info(f"模块 {module_name} 未启用数据处理")
            return data
        
        module_config = self.get_module_config(module_name)
        processing_config = module_config.get("数据处理", {})
        
        # 检查是否使用处理后的数据
        use_processed_data = processing_config.get("使用处理后数据", True)
        
        if use_processed_data and not force_reprocess:
            # 尝试使用已处理的数据
            processed_data_path = self.get_processed_data_path(module_name)
            if processed_data_path and os.path.exists(processed_data_path):
                try:
                    print_info(f"使用已处理的数据: {processed_data_path}")
                    processed_data = pd.read_csv(processed_data_path)
                    
                    # 检查数据质量
                    if self.check_data_quality(processed_data, module_name):
                        return processed_data
                    else:
                        print_warning("已处理的数据质量不满足要求，重新处理")
                except Exception as e:
                    print_warning(f"加载已处理数据失败: {e}")
        
        # 创建数据处理流水线
        if self.pipeline is None:
            self.pipeline = DataProcessingPipeline()
        
        # 根据模块配置调整处理参数
        self._adjust_pipeline_for_module(module_name)
        
        # 处理数据
        if data is not None:
            # 使用提供的数据
            success = self.pipeline.run_full_pipeline(data=data)
        else:
            # 从文件加载数据
            success = self.pipeline.run_full_pipeline()
        
        if success:
            processed_data = self.pipeline.processed_data
            
            # 更新缓存
            self.processed_data_cache[module_name] = {
                "path": str(self.pipeline.get_processed_data_path()),
                "exists": True,
                "timestamp": datetime.now().isoformat(),
                "data_shape": processed_data.shape
            }
            self._save_cache()
            
            return processed_data
        else:
            print_error("数据处理失败")
            return None
    
    def _adjust_pipeline_for_module(self, module_name):
        """
        根据模块配置调整处理流水线参数
        
        Args:
            module_name: 模块名称
        """
        module_config = self.get_module_config(module_name)
        processing_config = module_config.get("数据处理", {})
        
        # 根据模块调整特征工程配置
        if module_name == "basic_analysis":
            # 基础分析通常不需要复杂的特征工程
            self.pipeline.feature_config["统计特征"]["启用"] = False
            self.pipeline.feature_config["用户特征"]["启用"] = False
            self.pipeline.feature_config["业务特征"]["启用"] = False
        
        elif module_name == "arima_prediction":
            # ARIMA预测需要时间序列特征
            self.pipeline.feature_config["用户特征"]["启用"] = False
            self.pipeline.feature_config["业务特征"]["启用"] = False
            self.pipeline.transformation_config["时间序列特征"]["启用"] = True
    
    def get_data_summary(self, module_name):
        """
        获取数据摘要信息
        
        Args:
            module_name: 模块名称
            
        Returns:
            dict: 数据摘要信息
        """
        processed_data_path = self.get_processed_data_path(module_name)
        if processed_data_path and os.path.exists(processed_data_path):
            try:
                data = pd.read_csv(processed_data_path)
                return {
                    "数据路径": processed_data_path,
                    "数据形状": data.shape,
                    "特征数量": len(data.columns),
                    "数据量": len(data),
                    "缺失值统计": data.isnull().sum().to_dict(),
                    "数据类型": data.dtypes.to_dict()
                }
            except Exception as e:
                print_error(f"获取数据摘要失败: {e}")
        
        return None
    
    def clear_cache(self):
        """清除缓存"""
        self.processed_data_cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print_success("缓存已清除")


# 全局数据处理管理器实例
data_processing_manager = DataProcessingManager()


def get_data_for_module(module_name, data=None, force_reprocess=False):
    """
    为指定模块获取数据（便捷函数）
    
    Args:
        module_name: 模块名称
        data: 输入数据
        force_reprocess: 是否强制重新处理
        
    Returns:
        pd.DataFrame: 处理后的数据
    """
    return data_processing_manager.process_data_for_module(
        module_name, data, force_reprocess
    )


def should_process_data(module_name):
    """
    判断是否需要处理数据（便捷函数）
    
    Args:
        module_name: 模块名称
        
    Returns:
        bool: 是否需要处理数据
    """
    return data_processing_manager.should_process_data(module_name)


def get_processed_data_path(module_name=None):
    """
    获取处理后数据路径（便捷函数）
    
    Args:
        module_name: 模块名称
        
    Returns:
        str: 处理后数据路径
    """
    return data_processing_manager.get_processed_data_path(module_name) 