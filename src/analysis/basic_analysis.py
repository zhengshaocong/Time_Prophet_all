# -*- coding: utf-8 -*-
"""
基础数据分析模块
专注于数据探索和基础可视化
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import pandas as pd

from utils.data_processor import DataProcessor
from utils.visualization_utils import (
    setup_matplotlib, create_time_series_plot, create_histogram,
    create_balance_change_plot, create_monthly_comparison_plot,
    save_plot, close_plot
)
from utils.interactive_utils import print_header, print_success, print_error, print_info
from utils.file_utils import write_json
from config import DATA_DIR, IMAGES_DIR, OUTPUT_DATA_DIR, BASIC_ANALYSIS_CONFIG
from utils.config_utils import get_field_name, get_data_source_dispose_file

class BasicDataAnalysis(DataProcessor):
    """基础数据分析类"""
    
    def __init__(self):
        """初始化基础数据分析"""
        super().__init__()
        self.module_name = "basic_analysis"
        self.analysis_data = {}  # 存储分析数据
        self.analysis_results = {}  # 存储分析结果
        
        # 加载配置
        self.config = BASIC_ANALYSIS_CONFIG
        print_info(f"加载基础数据分析配置: {self.config.get('数据处理', {}).get('处理模式', 'auto')} 模式")
    
    def load_data_for_analysis(self, data_file=None):
        """
        为分析目的加载数据（不进行数据处理）
        
        Args:
            data_file: 数据文件路径，如果为None则自动查找
            
        Returns:
            bool: 是否成功加载数据
        """
        if data_file is None:
            # 自动查找数据文件
            data_files = list(DATA_DIR.glob("*.csv"))
            if not data_files:
                print_error("数据目录中没有找到CSV文件")
                return False
            data_file = data_files[0]  # 使用第一个找到的文件
        
        try:
            print_header("数据加载", "读取原始数据")
            print(f"正在加载数据文件: {data_file}")
            
            # 直接读取原始数据，不进行任何处理
            self.data = pd.read_csv(data_file)
            self.current_data_file = str(data_file)
            
            print_success(f"数据加载成功: {data_file}")
            print(f"数据形状: {self.data.shape[0]} 行 × {self.data.shape[1]} 列")
            
            return True
            
        except Exception as e:
            print_error(f"数据加载失败: {e}")
            return False
    
    def save_analysis_data(self, output_dir=None):
        """
        保存分析数据到文件
        
        Args:
            output_dir: 输出目录，默认为output/data
        """
        if not self.analysis_data:
            print_error("没有分析数据可保存")
            return False
        
        if output_dir is None:
            output_dir = OUTPUT_DATA_DIR
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 保存分析数据为JSON格式
            analysis_file = output_dir / "basic_analysis_data.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_data, f, ensure_ascii=False, indent=2, default=str)
            
            # 保存分析结果统计
            results_file = output_dir / "basic_analysis_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
            
            print_success(f"分析数据已保存: {analysis_file}")
            print_success(f"分析结果已保存: {results_file}")
            return True
            
        except Exception as e:
            print_error(f"保存分析数据失败: {e}")
            return False
    
    def generate_detailed_analysis_report(self, output_file=None):
        """
        生成详细的数据分析报告
        
        Args:
            output_file: 输出文件路径，默认为原始数据同文件夹下的data_analysis_detailed.md
        """
        if not self.data_info:
            print_error("请先进行数据分析")
            return False
        
        if output_file is None:
            # 获取当前数据文件名，在同目录下生成报告
            if hasattr(self, 'current_data_file') and self.current_data_file:
                data_file_path = Path(self.current_data_file)
                output_file = data_file_path.parent / "data_analysis_detailed.md"
            else:
                output_file = DATA_DIR / "data_analysis_detailed.md"
        
        try:
            # 创建详细的Markdown格式分析报告
            md_content = f"""# 资金流数据详细分析报告

## 报告信息
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **数据文件**: {getattr(self, 'current_data_file', '未知')}
- **分析模块**: 基础数据分析

## 数据概览
- **数据形状**: {self.data_info['数据形状'][0]} 行 × {self.data_info['数据形状'][1]} 列
- **数据大小**: {self.data_info.get('数据大小', '未知')}
- **内存使用**: {self.data_info.get('内存使用', '未知')}
- **分析时间**: {self.data_info['分析时间']}

## 数据质量评估

### 完整性分析
"""
            
            # 添加完整性分析
            if '字段信息' in self.data_info:
                total_fields = len(self.data_info['字段信息'])
                complete_fields = 0
                for field, info in self.data_info['字段信息'].items():
                    if info.get('空值数量', 0) == 0:
                        complete_fields += 1
                
                completeness_rate = complete_fields / total_fields * 100
                md_content += f"- **完整字段数**: {complete_fields}/{total_fields} ({completeness_rate:.1f}%)\n"
                md_content += f"- **数据完整性**: {'优秀' if completeness_rate >= 95 else '良好' if completeness_rate >= 80 else '一般' if completeness_rate >= 60 else '较差'}\n"
            
            md_content += "\n### 数据分布特征\n"
            
            # 添加数据分布特征
            if '字段信息' in self.data_info:
                for field, info in self.data_info['字段信息'].items():
                    md_content += f"\n#### {field}\n"
                    md_content += f"- **数据类型**: {info.get('数据类型', '未知')}\n"
                    md_content += f"- **非空值数量**: {info.get('非空值数量', 0):,}\n"
                    md_content += f"- **空值数量**: {info.get('空值数量', 0):,}\n"
                    md_content += f"- **唯一值数量**: {info.get('唯一值数量', 0):,}\n"
                    
                    # 数值型字段的统计信息
                    if info.get('数据类型') in ['int64', 'float64', 'int32', 'float32']:
                        if '最小值' in info:
                            md_content += f"- **最小值**: {info['最小值']:,.2f}\n"
                        if '最大值' in info:
                            md_content += f"- **最大值**: {info['最大值']:,.2f}\n"
                        if '平均值' in info:
                            md_content += f"- **平均值**: {info['平均值']:,.2f}\n"
                        if '中位数' in info:
                            md_content += f"- **中位数**: {info['中位数']:,.2f}\n"
                        if '标准差' in info:
                            md_content += f"- **标准差**: {info['标准差']:,.2f}\n"
                    
                    # 示例值
                    if '示例值' in info:
                        md_content += f"- **示例值**: {info['示例值']}\n"
            
            # 添加业务字段分析
            md_content += "\n## 业务字段分析\n"
            
            # 分析时间字段
            time_field = get_field_name("时间字段")
            if time_field and time_field in self.data_info.get('字段信息', {}):
                time_info = self.data_info['字段信息'][time_field]
                md_content += f"\n### 时间字段 ({time_field})\n"
                if '最小值' in time_info and '最大值' in time_info:
                    start_date = str(int(time_info['最小值']))
                    end_date = str(int(time_info['最大值']))
                    md_content += f"- **时间范围**: {start_date} 至 {end_date}\n"
                    md_content += f"- **数据天数**: {time_info.get('唯一值数量', 0)} 天\n"
            
            # 分析用户字段
            user_field = get_field_name("用户ID字段")
            if user_field and user_field in self.data_info.get('字段信息', {}):
                user_info = self.data_info['字段信息'][user_field]
                md_content += f"\n### 用户字段 ({user_field})\n"
                md_content += f"- **用户总数**: {user_info.get('唯一值数量', 0):,}\n"
                md_content += f"- **平均每用户记录数**: {self.data_info['数据形状'][0] / user_info.get('唯一值数量', 1):.1f}\n"
            
            # 分析金额字段
            amount_fields = ['total_purchase_amt', 'total_redeem_amt', 'tBalance', 'yBalance']
            md_content += "\n### 金额字段分析\n"
            for field in amount_fields:
                if field in self.data_info.get('字段信息', {}):
                    info = self.data_info['字段信息'][field]
                    md_content += f"\n#### {field}\n"
                    if '平均值' in info:
                        md_content += f"- **平均金额**: {info['平均值']:,.2f}\n"
                    if '中位数' in info:
                        md_content += f"- **中位金额**: {info['中位数']:,.2f}\n"
                    if '最大值' in info:
                        md_content += f"- **最大金额**: {info['最大值']:,.2f}\n"
            
            # 添加数据样本
            md_content += "\n## 数据样本\n"
            md_content += "```\n"
            for record in self.data_info.get('前几行数据', [])[:5]:
                md_content += str(record) + "\n"
            md_content += "```\n"
            
            # 添加分析建议
            md_content += "\n## 分析建议\n"
            md_content += "1. **数据质量**: 建议检查异常值和缺失值\n"
            md_content += "2. **时间特征**: 建议提取更多时间特征（如星期、月份等）\n"
            md_content += "3. **用户行为**: 建议分析用户行为模式和趋势\n"
            md_content += "4. **资金流特征**: 建议分析资金流的周期性和季节性\n"
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            print_success(f"详细分析报告已保存: {output_file}")
            return True
            
        except Exception as e:
            print_error(f"生成详细分析报告失败: {e}")
            return False
    
    def generate_data_source_dispose_config(self, data_source_name=None):
        """
        生成数据源特定的dispose配置文件
        
        Args:
            data_source_name: 数据源名称，如果为None则从当前数据文件名推断
            
        Returns:
            bool: 是否成功生成配置文件
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
        
        # 如果没有指定数据源名称，从当前数据文件名推断
        if data_source_name is None:
            # 从数据处理器中获取当前数据文件名
            if hasattr(self, 'current_data_file') and self.current_data_file:
                data_source_name = Path(self.current_data_file).stem
            else:
                # 尝试从数据目录中找到匹配的文件
                data_files = list(DATA_DIR.glob("*.csv"))
                if len(data_files) == 1:
                    data_source_name = data_files[0].stem
                else:
                    print_error("无法确定数据源名称，请手动指定")
                    return False
        
        print_header("生成数据源配置文件", f"数据源: {data_source_name}")
        
        # 自动检测字段映射
        field_mapping = self.auto_detect_field_mapping()
        if not field_mapping:
            print_error("字段映射检测失败")
            return False
        
        # 构建完整的dispose配置
        dispose_config = {
            "data_source_name": data_source_name,
            "generated_time": datetime.now().isoformat(),
            "field_mapping": field_mapping,
            "data_info": {
                "total_rows": len(self.data),
                "total_columns": len(self.data.columns),
                "columns": list(self.data.columns),
                "data_types": {col: str(dtype) for col, dtype in self.data.dtypes.items()}
            },
            "preprocessing_config": {
                "missing_value_fill": {
                    "申购金额字段": 0,
                    "赎回金额字段": 0,
                    "消费金额字段": 0,
                    "转账金额字段": 0
                },
                "outlier_detection": {
                    "enabled": True,
                    "threshold": 3.0,
                    "method": "clip"
                }
            }
        }
        
        # 保存配置文件
        dispose_file = get_data_source_dispose_file(data_source_name)
        try:
            with open(dispose_file, 'w', encoding='utf-8') as f:
                json.dump(dispose_config, f, ensure_ascii=False, indent=2)
            
            print_success(f"数据源配置文件已生成: {dispose_file}")
            print_info("配置文件包含以下内容:")
            print_info(f"  - 字段映射: {len(field_mapping)} 个字段")
            print_info(f"  - 数据信息: {dispose_config['data_info']['total_rows']} 行 × {dispose_config['data_info']['total_columns']} 列")
            print_info(f"  - 预处理配置: 缺失值处理和异常值检测")
            
            return True
            
        except Exception as e:
            print_error(f"保存配置文件失败: {e}")
            return False
    
    def save_data_analysis(self, output_file=None):
        """
        保存数据分析结果到文件
        
        Args:
            output_file: 输出文件路径
        """
        if not self.data_info:
            print_error("请先进行数据分析")
            return False
            
        if output_file is None:
            output_file = DATA_DIR / "data_analysis.md"
            
        try:
            # 创建Markdown格式的分析报告
            md_content = f"""# 资金流数据字段分析报告

## 数据概览
- **数据形状**: {self.data_info['数据形状'][0]} 行 × {self.data_info['数据形状'][1]} 列
- **分析时间**: {self.data_info['分析时间']}

## 字段详细信息

"""
            
            # 添加字段信息
            for field, info in self.data_info['字段信息'].items():
                md_content += f"### {field}\n"
                for key, value in info.items():
                    md_content += f"- **{key}**: {value}\n"
                md_content += "\n"
            
            # 添加前几行数据
            md_content += "## 前几行数据\n"
            md_content += "```\n"
            for record in self.data_info['前几行数据']:
                md_content += str(record) + "\n"
            md_content += "```\n"
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            print_success(f"分析报告已保存: {output_file}")
            return True
            
        except Exception as e:
            print_error(f"保存分析报告失败: {e}")
            return False
    
    def explore_data(self):
        """
        探索数据并保存分析结果
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
        
        print_header("数据探索", "分析数据特征")
        
        try:
            # 基础数据信息
            data_info = {
                '数据形状': self.data.shape,
                '数据大小': f"{self.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                '内存使用': f"{self.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                '分析时间': datetime.now().isoformat(),
                '字段信息': {},
                '前几行数据': self.data.head().to_dict('records')
            }
            
            # 根据配置进行数据质量检查
            if self.config.get('数据处理', {}).get('数据质量检查', True):
                print_info("执行数据质量检查...")
                data_info['数据质量检查'] = self._perform_data_quality_check()
            
            # 根据配置进行数据完整性检查
            if self.config.get('数据处理', {}).get('数据完整性检查', True):
                print_info("执行数据完整性检查...")
                data_info['数据完整性检查'] = self._perform_data_integrity_check()
            
            # 分析每个字段
            for column in self.data.columns:
                field_info = {
                    '数据类型': str(self.data[column].dtype),
                    '非空值数量': int(self.data[column].count()),
                    '空值数量': int(self.data[column].isnull().sum()),
                    '唯一值数量': int(self.data[column].nunique()),
                    '示例值': str(self.data[column].iloc[0]) if len(self.data) > 0 else 'N/A'
                }
                
                # 数值型字段的统计信息
                if self.data[column].dtype in ['int64', 'float64', 'int32', 'float32']:
                    field_info.update({
                        '最小值': float(self.data[column].min()),
                        '最大值': float(self.data[column].max()),
                        '平均值': float(self.data[column].mean()),
                        '中位数': float(self.data[column].median()),
                        '标准差': float(self.data[column].std())
                    })
                
                data_info['字段信息'][column] = field_info
            
            self.data_info = data_info
            
            # 保存分析数据
            self.analysis_data = {
                'basic_stats': {
                    'total_rows': len(self.data),
                    'total_columns': len(self.data.columns),
                    'memory_usage': self.data.memory_usage(deep=True).sum(),
                    'missing_values': self.data.isnull().sum().to_dict(),
                    'data_types': {col: str(dtype) for col, dtype in self.data.dtypes.items()}
                },
                'field_analysis': data_info['字段信息'],
                'sample_data': self.data.head(10).to_dict('records'),
                'data_quality_check': data_info.get('数据质量检查', {}),
                'data_integrity_check': data_info.get('数据完整性检查', {})
            }
            
            # 保存分析结果
            self.analysis_results = {
                'analysis_time': datetime.now().isoformat(),
                'data_quality_score': self._calculate_data_quality_score(),
                'key_insights': self._generate_key_insights()
            }
            
            print_success("数据探索完成")
            return True
            
        except Exception as e:
            print_error(f"数据探索失败: {e}")
            return False
    
    def _calculate_data_quality_score(self):
        """
        计算数据质量评分
        
        Returns:
            float: 数据质量评分 (0-100)
        """
        if self.data is None:
            return 0
        
        total_cells = len(self.data) * len(self.data.columns)
        missing_cells = self.data.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells * 100
        
        # 检查数据类型一致性
        type_consistency = 100
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                # 检查是否应该转换为数值类型
                try:
                    pd.to_numeric(self.data[column], errors='raise')
                    type_consistency -= 10  # 可以转换但未转换
                except:
                    pass
        
        return min(100, max(0, (completeness + type_consistency) / 2))
    
    def _generate_key_insights(self):
        """
        生成关键洞察
        
        Returns:
            list: 关键洞察列表
        """
        insights = []
        
        if self.data is None:
            return insights
        
        # 数据量洞察
        total_rows = len(self.data)
        if total_rows > 1000000:
            insights.append("数据量很大，建议使用采样或分批处理")
        elif total_rows < 1000:
            insights.append("数据量较小，可能影响分析结果的可靠性")
        
        # 缺失值洞察
        missing_ratio = self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))
        if missing_ratio > 0.1:
            insights.append("数据缺失值较多，需要重点关注数据质量")
        elif missing_ratio < 0.01:
            insights.append("数据完整性很好")
        
        # 时间字段洞察
        time_field = get_field_name("时间字段")
        if time_field and time_field in self.data.columns:
            unique_dates = self.data[time_field].nunique()
            if unique_dates > 365:
                insights.append("数据时间跨度超过一年，适合进行长期趋势分析")
            elif unique_dates < 30:
                insights.append("数据时间跨度较短，建议收集更多历史数据")
        
        return insights
    
    def _perform_data_quality_check(self):
        """
        执行数据质量检查
        
        Returns:
            dict: 数据质量检查结果
        """
        quality_report = {
            '检查时间': datetime.now().isoformat(),
            '检查项目': {}
        }
        
        # 检查缺失值
        missing_data = self.data.isnull().sum()
        quality_report['检查项目']['缺失值检查'] = {
            '总缺失值数量': int(missing_data.sum()),
            '缺失值比例': float(missing_data.sum() / (len(self.data) * len(self.data.columns))),
            '各字段缺失值': missing_data.to_dict()
        }
        
        # 检查重复值
        duplicate_rows = self.data.duplicated().sum()
        quality_report['检查项目']['重复值检查'] = {
            '重复行数量': int(duplicate_rows),
            '重复行比例': float(duplicate_rows / len(self.data))
        }
        
        # 检查数据类型
        data_types = self.data.dtypes.to_dict()
        quality_report['检查项目']['数据类型检查'] = {
            '数据类型分布': {str(dtype): list(data_types.values()).count(dtype) for dtype in set(data_types.values())}
        }
        
        return quality_report
    
    def _perform_data_integrity_check(self):
        """
        执行数据完整性检查
        
        Returns:
            dict: 数据完整性检查结果
        """
        integrity_report = {
            '检查时间': datetime.now().isoformat(),
            '检查项目': {}
        }
        
        # 检查关键字段的完整性
        key_fields = ['user_id', 'report_date', 'tBalance', 'yBalance']
        for field in key_fields:
            if field in self.data.columns:
                field_data = self.data[field]
                integrity_report['检查项目'][f'{field}_完整性'] = {
                    '非空值数量': int(field_data.count()),
                    '空值数量': int(field_data.isnull().sum()),
                    '完整性比例': float(field_data.count() / len(field_data)),
                    '唯一值数量': int(field_data.nunique())
                }
        
        # 检查数据一致性
        if 'tBalance' in self.data.columns and 'yBalance' in self.data.columns:
            # 检查余额字段的逻辑一致性
            balance_consistency = (self.data['tBalance'] >= 0).sum()
            integrity_report['检查项目']['余额一致性'] = {
                '非负余额记录数': int(balance_consistency),
                '负余额记录数': int(len(self.data) - balance_consistency),
                '一致性比例': float(balance_consistency / len(self.data))
            }
        
        return integrity_report
    
    def auto_detect_field_mapping(self):
        """
        自动检测数据字段映射
        
        Returns:
            dict: 字段映射字典
        """
        if self.data is None:
            print_error("请先加载数据")
            return None
        
        print("正在自动检测字段映射...")
        
        # 获取所有列名
        columns = list(self.data.columns)
        field_mapping = {}
        
        # 检测时间字段
        time_candidates = ['report_date', 'date', 'datetime', 'time', 'timestamp']
        time_field = None
        for candidate in time_candidates:
            if candidate in columns:
                time_field = candidate
                break
        
        if time_field:
            field_mapping["时间字段"] = time_field
            # 尝试检测时间格式
            try:
                sample_value = str(self.data[time_field].iloc[0])
                if len(sample_value) == 8 and sample_value.isdigit():
                    field_mapping["时间格式"] = "%Y%m%d"
                elif '-' in sample_value and ':' in sample_value:
                    field_mapping["时间格式"] = "%d-%m-%Y %H:%M"
                else:
                    field_mapping["时间格式"] = "%Y%m%d"  # 默认格式
            except:
                field_mapping["时间格式"] = "%Y%m%d"
        
        # 检测用户ID字段
        user_id_candidates = ['user_id', 'userid', 'id', 'user']
        for candidate in user_id_candidates:
            if candidate in columns:
                field_mapping["用户ID字段"] = candidate
                break
        
        # 检测余额相关字段
        balance_candidates = ['tbalance', 't_balance', 'balance', 'current_balance', 'tBalance']
        for candidate in balance_candidates:
            if candidate in columns:
                field_mapping["当前余额字段"] = candidate
                break
        
        ybalance_candidates = ['ybalance', 'y_balance', 'previous_balance', 'yesterday_balance', 'yBalance']
        for candidate in ybalance_candidates:
            if candidate in columns:
                field_mapping["昨日余额字段"] = candidate
                break
        
        # 检测申购赎回字段
        purchase_candidates = ['total_purchase_amt', 'purchase_amt', 'purchase_amount', 'buy_amount']
        for candidate in purchase_candidates:
            if candidate in columns:
                field_mapping["申购金额字段"] = candidate
                break
        
        redeem_candidates = ['total_redeem_amt', 'redeem_amt', 'redemption_amount', 'sell_amount']
        for candidate in redeem_candidates:
            if candidate in columns:
                field_mapping["赎回金额字段"] = candidate
                break
        
        # 检测其他字段
        consume_candidates = ['consume_amt', 'consume_amount', 'consumption']
        for candidate in consume_candidates:
            if candidate in columns:
                field_mapping["消费金额字段"] = candidate
                break
        
        transfer_candidates = ['transfer_amt', 'transfer_amount', 'transfer']
        for candidate in transfer_candidates:
            if candidate in columns:
                field_mapping["转账金额字段"] = candidate
                break
        
        # 检测分类字段
        category_fields = [col for col in columns if 'category' in col.lower()]
        if category_fields:
            field_mapping["分类字段"] = category_fields
        
        # 打印检测结果
        print("字段映射检测结果:")
        for field_type, field_name in field_mapping.items():
            print(f"  {field_type}: {field_name}")
        
        return field_mapping

    def visualize_data(self, save_plot=True):
        """
        基础数据可视化（使用原始数据）
        
        Args:
            save_plot: 是否保存图片
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
            
        print_header("基础数据可视化", "生成图表")
        
        # 设置matplotlib
        setup_matplotlib()
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('资金流数据可视化分析', fontsize=16, fontweight='bold')
        
        try:
            # 1. 时间序列图 - 使用原始字段
            time_field = get_field_name("时间字段")
            purchase_field = get_field_name("申购金额字段")
            redeem_field = get_field_name("赎回金额字段")
            
            if time_field and purchase_field and redeem_field:
                # 按日期聚合数据
                daily_data = self.data.groupby(time_field).agg({
                    purchase_field: 'sum',
                    redeem_field: 'sum'
                }).reset_index()
                
                # 转换时间格式为可读的日期
                try:
                    # 假设时间字段是YYYYMMDD格式
                    daily_data['date_readable'] = pd.to_datetime(daily_data[time_field], format='%Y%m%d')
                    x_data = daily_data['date_readable']
                except:
                    # 如果转换失败，使用原始数据
                    x_data = daily_data[time_field]
                
                # 创建时间序列图
                axes[0, 0].plot(x_data, daily_data[purchase_field], 
                               label='申购金额', color='green', linewidth=1, alpha=0.8)
                axes[0, 0].plot(x_data, daily_data[redeem_field], 
                               label='赎回金额', color='red', linewidth=1, alpha=0.8)
                axes[0, 0].set_title('资金流时间序列')
                axes[0, 0].set_xlabel('日期')
                axes[0, 0].set_ylabel('金额')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # 格式化x轴日期
                if 'date_readable' in daily_data.columns:
                    axes[0, 0].tick_params(axis='x', rotation=45)
            else:
                axes[0, 0].text(0.5, 0.5, '缺少时间或金额字段', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('资金流时间序列')
            
            # 2. 申购金额分布直方图
            if purchase_field:
                purchase_data = self.data[purchase_field]
                # 过滤掉0值，只显示有申购的记录
                non_zero_purchase = purchase_data[purchase_data > 0]
                if len(non_zero_purchase) > 0:
                    # 使用对数刻度来更好地显示分布
                    axes[0, 1].hist(non_zero_purchase, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
                    axes[0, 1].set_title('申购金额分布（非零值）')
                    axes[0, 1].set_xlabel('申购金额')
                    axes[0, 1].set_ylabel('频次')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # 添加统计信息
                    mean_val = non_zero_purchase.mean()
                    median_val = non_zero_purchase.median()
                    axes[0, 1].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'均值: {mean_val:,.0f}')
                    axes[0, 1].axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'中位数: {median_val:,.0f}')
                    axes[0, 1].legend()
                else:
                    axes[0, 1].text(0.5, 0.5, '无申购数据', 
                                   ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title('申购金额分布')
            else:
                axes[0, 1].text(0.5, 0.5, '缺少申购字段', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('申购金额分布')
            
            # 3. 余额分布图
            balance_field = get_field_name("当前余额字段")
            if balance_field:
                balance_data = self.data[balance_field]
                # 过滤掉0值，只显示有余额的记录
                non_zero_balance = balance_data[balance_data > 0]
                if len(non_zero_balance) > 0:
                    axes[1, 0].hist(non_zero_balance, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title('当前余额分布（非零值）')
                    axes[1, 0].set_xlabel('余额')
                    axes[1, 0].set_ylabel('频次')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # 添加统计信息
                    mean_val = non_zero_balance.mean()
                    median_val = non_zero_balance.median()
                    axes[1, 0].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'均值: {mean_val:,.0f}')
                    axes[1, 0].axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'中位数: {median_val:,.0f}')
                    axes[1, 0].legend()
                else:
                    axes[1, 0].text(0.5, 0.5, '无余额数据', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('当前余额分布')
            else:
                axes[1, 0].text(0.5, 0.5, '缺少余额字段', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('当前余额分布')
            
            # 4. 用户活跃度分析
            user_field = get_field_name("用户ID字段")
            if user_field:
                # 统计每个用户的记录数
                user_activity = self.data[user_field].value_counts()
                # 取前20个最活跃用户
                top_users = user_activity.head(20)
                
                # 创建用户标签
                user_labels = [f'用户{i+1}' for i in range(len(top_users))]
                
                # 绘制柱状图
                bars = axes[1, 1].bar(range(len(top_users)), top_users.values, color='orange', alpha=0.7)
                axes[1, 1].set_title('用户活跃度（前20名）')
                axes[1, 1].set_xlabel('用户排名')
                axes[1, 1].set_ylabel('记录数')
                axes[1, 1].grid(True, alpha=0.3)
                
                # 设置x轴标签
                axes[1, 1].set_xticks(range(len(top_users)))
                axes[1, 1].set_xticklabels(user_labels, rotation=45)
                
                # 在柱子上添加数值标签
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}', ha='center', va='bottom', fontsize=8)
                
                # 添加统计信息
                mean_activity = top_users.mean()
                axes[1, 1].axhline(mean_activity, color='red', linestyle='--', alpha=0.8, 
                                  label=f'平均活跃度: {mean_activity:.0f}')
                axes[1, 1].legend()
            else:
                axes[1, 1].text(0.5, 0.5, '缺少用户字段', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('用户活跃度分析')
            
            plt.tight_layout()
            
            if save_plot:
                # 确保图片保存目录存在
                IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                
                # 保存图片
                plot_file = IMAGES_DIR / "basic_data_analysis.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                print_success(f"分析图表已保存: {plot_file}")
            
            # 关闭图形以释放内存
            close_plot(fig)
            return True
            
        except Exception as e:
            print_error(f"数据可视化失败: {e}")
            close_plot(fig)
            return False

def run_basic_data_analysis():
    """
    运行基础数据分析功能
    """
    print_header("资金流预测系统", "基础数据分析")
    
    # 创建基础数据分析实例
    analysis = BasicDataAnalysis()
    
    # 1.1 读取原始数据（不进行数据处理）
    print("\n=== 步骤 1.1: 数据加载 ===")
    if analysis.load_data_for_analysis():
        print_success("数据加载成功")
        
        # 1.2 基础数据探索
        print("\n=== 步骤 1.2: 基础数据探索 ===")
        if analysis.explore_data():
            print_success("数据探索完成")
            
            # 1.3 保存分析数据
            print("\n=== 步骤 1.3: 保存分析数据 ===")
            if analysis.save_analysis_data():
                print_success("分析数据保存完成")
            
            # 1.4 保存分析结果
            print("\n=== 步骤 1.4: 保存分析结果 ===")
            if analysis.save_data_analysis():
                print_success("分析结果保存完成")
            
            # 1.5 生成详细分析报告
            print("\n=== 步骤 1.5: 生成详细分析报告 ===")
            if analysis.generate_detailed_analysis_report():
                print_success("详细分析报告生成完成")
            
            # 1.6 生成数据源dispose配置文件
            print("\n=== 步骤 1.6: 生成数据源配置文件 ===")
            if analysis.generate_data_source_dispose_config():
                print_success("数据源配置文件生成完成")
            else:
                print_error("数据源配置文件生成失败")
            
            # 1.7 数据可视化
            print("\n=== 步骤 1.7: 数据可视化 ===")
            if analysis.visualize_data(save_plot=True):
                print_success("数据可视化完成")
            else:
                print_error("数据可视化失败")
        else:
            print_error("数据探索失败")
    else:
        print_error("数据加载失败")
        return False
    
    print_success("基础数据分析完成！")
    return True

if __name__ == "__main__":
    run_basic_data_analysis() 