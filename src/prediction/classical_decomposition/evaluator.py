# -*- coding: utf-8 -*-
"""
经典分解法性能评估模块
Classical Decomposition Performance Evaluator Module

功能：
1. 模型性能评估
2. 基准模型比较
3. 预测精度分析
4. 评估指标计算
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

from utils.interactive_utils import print_header, print_info, print_success, print_warning


class ClassicalDecompositionEvaluator:
    """经典分解法性能评估器"""
    
    def __init__(self, config):
        """
        初始化性能评估器
        
        Args:
            config: 经典分解法配置
        """
        self.config = config
        self.evaluation_results = {}
        self.baseline_results = {}
        
    def evaluate_model_performance(self, actual_values, predicted_values, model_name="Classical_Decomposition"):
        """
        评估模型性能
        
        Args:
            actual_values: 实际值
            predicted_values: 预测值
            model_name: 模型名称
            
        Returns:
            dict: 评估结果
        """
        print_header(f"评估{model_name}模型性能")
        
        try:
            # 确保数据长度一致
            if len(actual_values) != len(predicted_values):
                print_warning("实际值和预测值长度不一致，将截取到较短的长度")
                min_length = min(len(actual_values), len(predicted_values))
                actual_values = actual_values[:min_length]
                predicted_values = predicted_values[:min_length]
            
            # 移除NaN值
            valid_mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
            actual_clean = actual_values[valid_mask]
            predicted_clean = predicted_values[valid_mask]
            
            if len(actual_clean) == 0:
                print_warning("没有有效的数据用于评估")
                return {}
            
            # 计算评估指标
            evaluation_metrics = self._calculate_metrics(actual_clean, predicted_clean)
            
            # 存储评估结果
            self.evaluation_results[model_name] = {
                'metrics': evaluation_metrics,
                'data_length': len(actual_clean),
                'actual_values': actual_clean,
                'predicted_values': predicted_clean
            }
            
            # 输出评估结果
            print_success(f"{model_name}模型评估完成:")
            for metric, value in evaluation_metrics.items():
                print_info(f"  {metric}: {value:.4f}")
            
            return evaluation_metrics
            
        except Exception as e:
            print_warning(f"模型性能评估失败: {e}")
            return {}

    def evaluate_interval_quality(self, actual_values, predicted_values, model_name="Classical_Decomposition", confidence_level: float = 0.95):
        """
        评估预测区间质量（基于样本内回测）：
        - PICP: 区间覆盖率 (Prediction Interval Coverage Probability)
        - MPIW: 平均区间宽度 (Mean Prediction Interval Width)
        - NMPIW: 归一化平均区间宽度（除以样本范围）
        - ACE: 覆盖误差 |PICP - 置信水平|
        说明：使用实际值与点预测残差的标准差构造对称区间。
        """
        try:
            actual_values = np.asarray(actual_values)
            predicted_values = np.asarray(predicted_values)
            # 对齐长度
            min_len = min(len(actual_values), len(predicted_values))
            actual_values = actual_values[:min_len]
            predicted_values = predicted_values[:min_len]

            # 清理 NaN
            mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
            actual = actual_values[mask]
            pred = predicted_values[mask]
            if actual.size == 0:
                print_warning("没有有效数据用于区间评估")
                return {}

            residuals = actual - pred
            std = float(np.nanstd(residuals, ddof=1))
            if not np.isfinite(std):
                std = 0.0
            z = 1.96 if abs(confidence_level - 0.95) < 1e-6 else 1.64 if abs(confidence_level - 0.90) < 1e-6 else 2.58 if abs(confidence_level - 0.99) < 1e-6 else 1.96

            lower = pred - z * std
            upper = pred + z * std
            cover = (actual >= lower) & (actual <= upper)
            picp = float(np.mean(cover)) if cover.size > 0 else np.nan
            mpiw = float(np.mean(upper - lower)) if upper.size > 0 else np.nan
            rng = float(np.nanmax(actual) - np.nanmin(actual)) if actual.size > 0 else np.nan
            nmpiw = float(mpiw / rng) if rng and np.isfinite(rng) and rng != 0 else np.nan
            ace = float(abs(picp - confidence_level)) if np.isfinite(picp) else np.nan

            interval_metrics = {
                'PICP': picp,
                'MPIW': mpiw,
                'NMPIW': nmpiw,
                'ACE': ace,
                '置信水平': confidence_level
            }

            # 存入结果
            if model_name not in self.evaluation_results:
                self.evaluation_results[model_name] = {}
            self.evaluation_results[model_name]['interval_quality'] = interval_metrics

            # 打印并与其他内容隔开
            print_header("区间质量评估")
            for k, v in interval_metrics.items():
                if isinstance(v, float) and np.isfinite(v):
                    print_info(f"  {k}: {v:.4f}")
                else:
                    print_info(f"  {k}: {v}")

            return interval_metrics
        except Exception as e:
            print_warning(f"区间质量评估失败: {e}")
            return {}
    
    def _calculate_metrics(self, actual, predicted):
        """
        计算评估指标
        
        Args:
            actual: 实际值
            predicted: 预测值
            
        Returns:
            dict: 评估指标
        """
        metrics = {}
        
        # 获取配置的评估指标
        evaluation_metrics = self.config.get('性能评估', {}).get('评估指标', ['MAE', 'MSE', 'RMSE', 'MAPE'])
        
        if 'MAE' in evaluation_metrics:
            metrics['MAE'] = mean_absolute_error(actual, predicted)
        
        if 'MSE' in evaluation_metrics:
            metrics['MSE'] = mean_squared_error(actual, predicted)
        
        if 'RMSE' in evaluation_metrics:
            metrics['RMSE'] = np.sqrt(mean_squared_error(actual, predicted))
        
        if 'MAPE' in evaluation_metrics:
            # 避免除零错误
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                mape = mean_absolute_percentage_error(actual[non_zero_mask], predicted[non_zero_mask])
                metrics['MAPE'] = mape
            else:
                metrics['MAPE'] = np.nan
        
        # 计算R²
        if 'R2' in evaluation_metrics or 'R²' in evaluation_metrics:
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            if ss_tot != 0:
                r2 = 1 - (ss_res / ss_tot)
                metrics['R²'] = r2
            else:
                metrics['R²'] = np.nan
        
        # 计算平均绝对百分比误差
        if 'MAPE' not in metrics:
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
                metrics['MAPE(%)'] = mape
            else:
                metrics['MAPE(%)'] = np.nan
        
        return metrics
    
    def create_baseline_model(self, actual_values, baseline_type="naive"):
        """
        创建基准模型
        
        Args:
            actual_values: 实际值
            baseline_type: 基准模型类型
            
        Returns:
            np.array: 基准模型预测值
        """
        print_info(f"创建{baseline_type}基准模型...")
        
        try:
            if baseline_type == "naive":
                # 朴素预测：使用最后一个值
                baseline_predictions = np.full_like(actual_values, actual_values[-1])
                
            elif baseline_type == "seasonal_naive":
                # 季节性朴素预测：使用上一个周期的值
                # 假设周期为7（周度）
                period = 7
                baseline_predictions = np.array(actual_values)
                for i in range(period, len(actual_values)):
                    baseline_predictions[i] = actual_values[i - period]
                    
            elif baseline_type == "drift":
                # 漂移预测：基于线性趋势
                baseline_predictions = np.array(actual_values)
                if len(actual_values) > 1:
                    # 计算线性趋势
                    x = np.arange(len(actual_values))
                    slope = (actual_values[-1] - actual_values[0]) / (len(actual_values) - 1)
                    baseline_predictions = actual_values[0] + slope * x
                    
            else:
                print_warning(f"不支持的基准模型类型: {baseline_type}")
                return None
            
            self.baseline_results[baseline_type] = {
                'predictions': baseline_predictions,
                'actual_values': actual_values
            }
            
            print_success(f"{baseline_type}基准模型创建完成")
            return baseline_predictions
            
        except Exception as e:
            print_warning(f"创建基准模型失败: {e}")
            return None
    
    def compare_with_baseline(self, model_name="Classical_Decomposition"):
        """
        与基准模型比较
        
        Args:
            model_name: 模型名称
            
        Returns:
            dict: 比较结果
        """
        print_info("与基准模型比较...")
        
        if model_name not in self.evaluation_results:
            print_warning(f"模型{model_name}的评估结果不存在")
            return {}
        
        comparison_results = {}
        
        try:
            # 获取模型评估结果
            model_results = self.evaluation_results[model_name]
            
            # 与各种基准模型比较
            for baseline_type, baseline_data in self.baseline_results.items():
                # 评估基准模型
                baseline_metrics = self._calculate_metrics(
                    baseline_data['actual_values'], 
                    baseline_data['predictions']
                )
                
                # 计算改进比例
                improvements = {}
                for metric in model_results['metrics'].keys():
                    if metric in baseline_metrics:
                        model_value = model_results['metrics'][metric]
                        baseline_value = baseline_metrics[metric]
                        
                        if baseline_value != 0:
                            if metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'MAPE(%)']:
                                # 越小越好的指标
                                improvement = (baseline_value - model_value) / baseline_value * 100
                            else:
                                # 越大越好的指标（如R²）
                                improvement = (model_value - baseline_value) / baseline_value * 100
                            
                            improvements[metric] = improvement
                
                comparison_results[baseline_type] = {
                    'baseline_metrics': baseline_metrics,
                    'improvements': improvements
                }
            
            # 输出比较结果
            print_success("模型比较完成:")
            for baseline_type, comparison in comparison_results.items():
                print_info(f"  与{baseline_type}比较:")
                for metric, improvement in comparison['improvements'].items():
                    if improvement > 0:
                        print_info(f"    {metric}: 改进 {improvement:.2f}%")
                    else:
                        print_info(f"    {metric}: 下降 {abs(improvement):.2f}%")
            
            return comparison_results
            
        except Exception as e:
            print_warning(f"模型比较失败: {e}")
            return {}
    
    def generate_evaluation_report(self, model_name="Classical_Decomposition"):
        """
        生成评估报告
        
        Args:
            model_name: 模型名称
            
        Returns:
            dict: 完整评估报告
        """
        print_info("生成评估报告...")
        
        if model_name not in self.evaluation_results:
            print_warning(f"模型{model_name}的评估结果不存在")
            return {}
        
        try:
            model_results = self.evaluation_results[model_name]
            
            # 创建评估报告
            report = {
                '模型名称': model_name,
                '评估时间': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                '数据信息': {
                    '数据长度': model_results['data_length'],
                    '评估指标数量': len(model_results['metrics'])
                },
                '性能指标': model_results['metrics'],
                '基准模型比较': self.compare_with_baseline(model_name)
            }
            if 'interval_quality' in model_results:
                report['区间质量'] = model_results['interval_quality']
            
            # 性能评级
            report['性能评级'] = self._rate_performance(model_results['metrics'])
            
            print_success("评估报告生成完成")
            return report
            
        except Exception as e:
            print_warning(f"生成评估报告失败: {e}")
            return {}
    
    def _rate_performance(self, metrics):
        """
        对性能进行评级
        
        Args:
            metrics: 性能指标
            
        Returns:
            str: 性能评级
        """
        # 简单的评级逻辑，可以根据需要调整
        if 'MAPE' in metrics:
            mape = metrics['MAPE']
            if mape < 0.1:  # 10%
                return "优秀"
            elif mape < 0.2:  # 20%
                return "良好"
            elif mape < 0.3:  # 30%
                return "一般"
            else:
                return "需要改进"
        
        if 'R²' in metrics:
            r2 = metrics['R²']
            if r2 > 0.8:
                return "优秀"
            elif r2 > 0.6:
                return "良好"
            elif r2 > 0.4:
                return "一般"
            else:
                return "需要改进"
        
        return "无法评级"
    
    def save_evaluation_results(self, output_path=None):
        """
        保存评估结果
        
        Args:
            output_path: 输出路径
            
        Returns:
            str: 保存的文件路径
        """
        if not output_path:
            output_path = "output/classical_decomposition/evaluation_results.json"
        
        try:
            import json
            from pathlib import Path
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换numpy类型为Python原生类型
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                return obj
            
            # 递归转换
            def recursive_convert(data):
                if isinstance(data, dict):
                    return {k: recursive_convert(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [recursive_convert(item) for item in data]
                else:
                    return convert_numpy(data)
            
            converted_results = recursive_convert(self.evaluation_results)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, ensure_ascii=False, indent=2)
            
            print_success(f"评估结果已保存: {output_file}")
            return str(output_file)
            
        except Exception as e:
            print_warning(f"保存评估结果失败: {e}")
            return None
