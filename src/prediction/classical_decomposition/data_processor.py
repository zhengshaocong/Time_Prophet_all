# -*- coding: utf-8 -*-
"""
经典分解法数据预处理模块
Classical Decomposition Data Preprocessing Module

功能：
1. 读取前N段数据获取基础信息
2. 数据质量检查
3. 数据格式转换
4. 基础统计分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from utils.interactive_utils import print_header, print_info, print_success, print_warning, print_error
from utils.data_processing_manager import get_data_for_module, should_process_data


class ClassicalDecompositionDataProcessor:
    """经典分解法数据预处理器"""
    
    def __init__(self, config):
        """
        初始化数据预处理器
        
        Args:
            config: 经典分解法配置
        """
        self.config = config
        self.data = None
        self.basic_info = {}
        self.data_quality_report = {}
        
    def load_and_analyze_data(self, file_path, n_segments=5):
        """
        读取前N段数据并获取基础信息
        
        Args:
            file_path: 数据文件路径
            n_segments: 读取前N段数据
            
        Returns:
            bool: 是否成功
        """
        print_header("数据加载与基础信息分析")
        
        try:
            # 读取数据
            if not self._load_data(file_path):
                return False
            
            # 获取基础信息
            if not self._extract_basic_info(n_segments):
                return False
            
            # 数据质量检查
            if not self._check_data_quality():
                return False
            
            # 数据预处理
            if not self._preprocess_data():
                return False
            
            print_success("数据加载与基础信息分析完成")
            return True
            
        except Exception as e:
            print_error(f"数据加载与基础信息分析失败: {e}")
            return False
    
    def _load_data(self, file_path):
        """加载数据文件，优先复用全局数据预处理结果"""
        print_info("加载数据文件（优先复用预处理数据）...")
        
        # 1) 优先尝试通过数据处理管理器获取已处理数据
        try:
            if should_process_data("classical_decomposition"):
                processed = get_data_for_module("classical_decomposition")
                if processed is not None and len(processed) > 0:
                    self.data = processed
                    print_success(f"已复用处理后的数据，形状: {self.data.shape}")
                    return True
        except Exception as e:
            print_warning(f"复用预处理数据失败，回退到直接读取源数据: {e}")
        
        # 2) 回退：直接从文件读取
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print_error(f"数据文件不存在: {file_path}")
                return False
            
            if file_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            else:
                print_error(f"不支持的文件格式: {file_path.suffix}")
                return False
            
            print_success(f"数据加载成功，共{len(self.data)}行")
            return True
            
        except Exception as e:
            print_error(f"数据加载失败: {e}")
            return False
    
    def _extract_basic_info(self, n_segments):
        """提取基础信息"""
        print_info("提取基础信息...")
        
        if self.data is None or len(self.data) == 0:
            print_error("数据为空，无法提取基础信息")
            return False
        
        try:
            # 基本信息
            self.basic_info = {
                '总行数': len(self.data),
                '总列数': len(self.data.columns),
                '列名': list(self.data.columns),
                '数据类型': self.data.dtypes.to_dict(),
                '内存使用': self.data.memory_usage(deep=True).sum()
            }
            
            # 前N段数据统计
            if len(self.data) >= n_segments:
                sample_data = self.data.head(n_segments)
                self.basic_info['前N段数据'] = {
                    '行数': len(sample_data),
                    '数据预览': sample_data.to_dict('records')
                }
            
            # 数值列统计
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                numeric_stats = {}
                for col in numeric_columns:
                    numeric_stats[col] = {
                        '均值': self.data[col].mean(),
                        '标准差': self.data[col].std(),
                        '最小值': self.data[col].min(),
                        '最大值': self.data[col].max(),
                        '缺失值数量': self.data[col].isna().sum()
                    }
                self.basic_info['数值列统计'] = numeric_stats
            
            # 日期列处理
            date_columns = []
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    try:
                        pd.to_datetime(self.data[col].iloc[0])
                        date_columns.append(col)
                    except:
                        continue
            
            if date_columns:
                self.basic_info['日期列'] = date_columns
                # 转换日期列
                for col in date_columns:
                    self.data[col] = pd.to_datetime(self.data[col])
                
                # 日期范围
                for col in date_columns:
                    date_range = self.data[col].agg(['min', 'max'])
                    self.basic_info[f'{col}_日期范围'] = {
                        '开始日期': date_range['min'],
                        '结束日期': date_range['max'],
                        '总天数': (date_range['max'] - date_range['min']).days + 1
                    }
            
            print_success("基础信息提取完成")
            return True
            
        except Exception as e:
            print_error(f"基础信息提取失败: {e}")
            return False
    
    def _check_data_quality(self):
        """检查数据质量"""
        print_info("检查数据质量...")
        
        try:
            quality_report = {}
            
            # 缺失值检查
            missing_data = self.data.isnull().sum()
            missing_percentage = (missing_data / len(self.data)) * 100
            
            quality_report['缺失值'] = {
                '各列缺失值数量': missing_data.to_dict(),
                '各列缺失值比例': missing_percentage.to_dict(),
                '总缺失值比例': missing_data.sum() / (len(self.data) * len(self.data.columns))
            }
            
            # 重复值检查
            duplicate_rows = self.data.duplicated().sum()
            quality_report['重复值'] = {
                '重复行数': duplicate_rows,
                '重复行比例': duplicate_rows / len(self.data)
            }
            
            # 异常值检查（针对数值列）
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                outlier_report = {}
                for col in numeric_columns:
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                    outlier_report[col] = {
                        '异常值数量': len(outliers),
                        '异常值比例': len(outliers) / len(self.data),
                        '下界': lower_bound,
                        '上界': upper_bound
                    }
                
                quality_report['异常值'] = outlier_report
            
            # 数据完整性评估
            min_completeness = self.config.get('数据质量要求', {}).get('最小完整度', 0.9)
            max_missing_ratio = self.config.get('数据质量要求', {}).get('最大缺失值比例', 0.1)
            
            overall_completeness = 1 - quality_report['缺失值']['总缺失值比例']
            quality_report['数据完整性'] = {
                '整体完整度': overall_completeness,
                '是否满足要求': overall_completeness >= min_completeness,
                '要求完整度': min_completeness
            }
            
            self.data_quality_report = quality_report
            
            # 输出质量报告摘要
            print_info(f"数据质量检查完成:")
            print_info(f"  整体完整度: {overall_completeness:.2%}")
            print_info(f"  重复行比例: {quality_report['重复值']['重复行比例']:.2%}")
            
            if overall_completeness < min_completeness:
                print_warning(f"数据完整度({overall_completeness:.2%})低于要求({min_completeness:.2%})")
            
            return True
            
        except Exception as e:
            print_error(f"数据质量检查失败: {e}")
            return False
    
    def _preprocess_data(self):
        """数据预处理"""
        print_info("进行数据预处理...")
        
        try:
            # 处理缺失值
            if self.config.get('数据处理', {}).get('启用数据处理', True):
                self._handle_missing_values()
            
            # 处理异常值
            if self.config.get('数据处理', {}).get('数据质量要求', {}).get('异常值处理') == 'clip':
                self._handle_outliers()
            
            # 数据采样
            if self.config.get('基础数据信息', {}).get('数据采样', {}).get('启用采样', False):
                self._apply_sampling()
            
            print_success("数据预处理完成")
            return True
            
        except Exception as e:
            print_error(f"数据预处理失败: {e}")
            return False
    
    def _handle_missing_values(self):
        """处理缺失值"""
        # 对于数值列，使用前向填充
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.data[col].isna().sum() > 0:
                self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
        
        # 对于日期列，使用插值
        date_columns = self.data.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            if self.data[col].isna().sum() > 0:
                self.data[col] = self.data[col].interpolate(method='time')
    
    def _handle_outliers(self):
        """处理异常值（使用clipping方法）"""
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 将异常值限制在边界内
            self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
    
    def _apply_sampling(self):
        """应用数据采样"""
        sampling_config = self.config.get('基础数据信息', {}).get('数据采样', {})
        sampling_ratio = sampling_config.get('采样比例', 0.8)
        sampling_method = sampling_config.get('采样方式', 'random')
        
        if sampling_method == 'random':
            self.data = self.data.sample(frac=sampling_ratio, random_state=42)
        elif sampling_method == 'systematic':
            step = int(1 / sampling_ratio)
            self.data = self.data.iloc[::step]
        
        self.data = self.data.sort_index().reset_index(drop=True)
    
    def get_processed_data(self):
        """获取处理后的数据"""
        return self.data
    
    def get_basic_info(self):
        """获取基础信息"""
        return self.basic_info
    
    def get_quality_report(self):
        """获取质量报告"""
        return self.data_quality_report
    
    def prepare_for_prediction(self, date_column, value_column):
        """
        准备预测所需的数据格式
        
        Args:
            date_column: 日期列名
            value_column: 值列名
            
        Returns:
            tuple: (预测数据, 重命名后的列名映射)
        """
        print_info("准备预测数据...")
        
        if self.data is None:
            print_error("数据未加载")
            return None, None
        
        try:
            # 确保必要的列存在
            if date_column not in self.data.columns:
                print_error(f"日期列'{date_column}'不存在")
                return None, None
            
            if value_column not in self.data.columns:
                print_error(f"值列'{value_column}'不存在")
                return None, None
            
            # 创建预测数据（保留申购/赎回列以便后续分别预测）
            prediction_data = self.data[[date_column, value_column]].copy()
            prediction_data.columns = ['date', 'value']

            # 如果存在申购/赎回字段，则一并保留在预测数据中
            extra_cols = []
            for col in ['total_purchase_amt', 'total_redeem_amt']:
                if col in self.data.columns:
                    extra_cols.append(col)
            if extra_cols:
                extra_df = self.data[[date_column] + extra_cols].copy()
                extra_df = extra_df.rename(columns={date_column: 'date'})
                # 数值化并缺失填补
                for col in extra_cols:
                    try:
                        extra_df[col] = pd.to_numeric(extra_df[col], errors='coerce').fillna(0)
                    except Exception:
                        pass
                prediction_data = prediction_data.merge(extra_df, on='date', how='left')
            
            # 排序并重置索引
            prediction_data = prediction_data.sort_values('date').reset_index(drop=True)
            
            # 移除缺失值
            prediction_data = prediction_data.dropna()
            
            # 返回重命名后的列名映射
            column_mapping = {
                'date_column': 'date',
                'value_column': 'value'
            }
            
            print_success(f"预测数据准备完成，共{len(prediction_data)}行")
            return prediction_data, column_mapping
            
        except Exception as e:
            print_error(f"预测数据准备失败: {e}")
            return None, None
