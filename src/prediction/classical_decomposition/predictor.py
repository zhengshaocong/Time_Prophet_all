# -*- coding: utf-8 -*-
"""
经典分解法主预测器模块
Classical Decomposition Main Predictor Module

功能：
1. 整合所有子模块
2. 实现完整的经典分解法预测流程
3. 提供统一的预测接口
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .core import ClassicalDecompositionCore
from .data_processor import ClassicalDecompositionDataProcessor
from .visualization import ClassicalDecompositionVisualization
from .evaluator import ClassicalDecompositionEvaluator

from utils.interactive_utils import print_header, print_info, print_success, print_warning, print_error


class ClassicalDecompositionPredictor:
    """经典分解法主预测器"""
    
    def __init__(self, config):
        """
        初始化经典分解法预测器
        
        Args:
            config: 经典分解法配置
        """
        self.config = config
        
        # 初始化各个模块
        self.data_processor = ClassicalDecompositionDataProcessor(config)
        self.core = ClassicalDecompositionCore(config)
        self.visualization = ClassicalDecompositionVisualization(config)
        self.evaluator = ClassicalDecompositionEvaluator(config)
        
        # 存储结果
        self.data = None
        self.basic_info = {}
        self.periodic_factors = None
        self.base_values = None
        self.prediction_result = None
        self.decomposition_result = None
        
    def run_prediction_pipeline(self, data_file_path, date_column, value_column, 
                               period_type="weekday", remove_periodic_effect=True, 
                               smooth_window=3, forecast_steps=30, confidence_level=0.95):
        """
        运行完整的经典分解法预测流程
        
        Args:
            data_file_path: 数据文件路径
            date_column: 日期列名
            value_column: 值列名
            period_type: 周期类型，默认weekday
            remove_periodic_effect: 是否去除周期影响，默认True
            smooth_window: 平滑窗口大小，默认3
            forecast_steps: 预测步数，默认30
            confidence_level: 置信水平，默认0.95
            
        Returns:
            bool: 是否成功
        """
        print_header("经典分解法预测流程")
        
        # 保存参数供后续步骤使用
        self.prediction_params = {
            'date_column': date_column,
            'value_column': value_column,
            'period_type': period_type,
            'remove_periodic_effect': remove_periodic_effect,
            'smooth_window': smooth_window,
            'forecast_steps': forecast_steps,
            'confidence_level': confidence_level
        }
        
        try:
            # 步骤1: 读取数据的前五段数据，获取基础数据信息
            if not self._step1_load_and_analyze_data(data_file_path, date_column, value_column):
                return False
            
            # 步骤2: 获取周期因子（weekday）
            if not self._step2_calculate_periodic_factors():
                return False
            
            # 步骤3: 计算base（去除周期影响）
            if not self._step3_calculate_base_values():
                return False
            
            # 步骤4: 使用base和周期因子进行预测
            if not self._step4_predict_future():
                return False
            
            # 步骤5: 创建可视化图表
            if not self._step5_create_visualizations():
                return False
            
            # 步骤5.5: 保存预测结果为CSV格式
            if not self._step5_5_save_prediction_csv():
                print_warning("CSV保存失败，但继续执行")
            
            # 步骤6: 性能评估
            if not self._step6_evaluate_performance():
                return False
            
            print_success("经典分解法预测流程完成")
            return True
            
        except Exception as e:
            print_error(f"经典分解法预测流程失败: {e}")
            return False
    
    def _step1_load_and_analyze_data(self, data_file_path, date_column, value_column):
        """步骤1: 读取数据的前五段数据，获取基础数据信息"""
        print_info("步骤1: 读取数据的前五段数据，获取基础数据信息")
        
        # 获取配置的前N段数据数量
        n_segments = self.config.get('基础数据信息', {}).get('读取前N段数据', 5)
        
        # 加载和分析数据
        if not self.data_processor.load_and_analyze_data(data_file_path, n_segments):
            return False
        
        # 获取处理后的数据
        self.data = self.data_processor.get_processed_data()
        self.basic_info = self.data_processor.get_basic_info()
        
        # 准备预测数据
        prediction_data, column_mapping = self.data_processor.prepare_for_prediction(date_column, value_column)
        if prediction_data is None:
            return False
        
        # 更新数据为预测格式
        self.data = prediction_data
        
        # 更新列名映射
        if column_mapping:
            self.prediction_params['date_column'] = column_mapping['date_column']
            self.prediction_params['value_column'] = column_mapping['value_column']
        
        print_success(f"数据加载完成，共{len(self.data)}行数据")
        print_info(f"数据列名: {list(self.data.columns)}")
        print_info(f"数据前5行: {self.data.head().to_dict('records')}")
        return True
    
    def _step2_calculate_periodic_factors(self):
        """步骤2: 获取周期因子（weekday）"""
        print_info("步骤2: 获取周期因子（weekday）")
        
        try:
            # 使用传递的参数
            period_type = self.prediction_params.get('period_type', 'weekday')
            date_column = self.prediction_params.get('date_column', 'date')
            value_column = self.prediction_params.get('value_column', 'value')
            
            # 计算周期因子

            self.periodic_factors = self.core.calculate_periodic_factors(
                self.data, 
                period_type=period_type,
                date_column=date_column,
                value_column=value_column
            )
            
            if self.periodic_factors is None:
                print_error("周期因子计算失败")
                return False
            
            print_success(f"周期因子计算完成，类型: {period_type}")
            return True
            
        except Exception as e:
            print_error(f"周期因子计算失败: {e}")
            return False
    
    def _step3_calculate_base_values(self):
        """步骤3: 计算base（去除周期影响）"""
        print_info("步骤3: 计算base（去除周期影响）")
        
        try:
            # 使用传递的参数
            remove_periodic_effect = self.prediction_params.get('remove_periodic_effect', True)
            smooth_window = self.prediction_params.get('smooth_window', 3)
            date_column = self.prediction_params.get('date_column', 'date')
            value_column = self.prediction_params.get('value_column', 'value')
            
            # 计算Base值
            self.base_values = self.core.calculate_base_values(
                self.data, 
                remove_periodic_effect=remove_periodic_effect,
                smooth_window=smooth_window,
                date_column=date_column,
                value_column=value_column
            )
            
            if self.base_values is None:
                print_error("Base值计算失败")
                return False
            
            print_success("Base值计算完成")
            return True
            
        except Exception as e:
            print_error(f"Base值计算失败: {e}")
            return False
    
    def _step4_predict_future(self):
        """步骤4: 使用base和周期因子进行预测"""
        print_info("步骤4: 使用base和周期因子进行预测")
        
        try:
            # 使用传递的参数
            forecast_steps = self.prediction_params.get('forecast_steps', 30)
            confidence_level = self.prediction_params.get('confidence_level', 0.95)
            date_column = self.prediction_params.get('date_column', 'date')
            value_column = self.prediction_params.get('value_column', 'value')
            
            # 检查是否有申购和赎回数据
            has_purchase_redeem = self._check_purchase_redeem_data()
            
            # 获取历史数据的最后日期
            last_history_date = self._get_last_history_date(date_column)
            
            # 进行预测
            self.prediction_result = self.core.predict_future(
                self.base_values,
                self.periodic_factors,
                forecast_steps=forecast_steps,
                confidence_level=confidence_level,
                last_history_date=last_history_date,
                history_dates=self.data[date_column],
                hist_original_values=self.data[value_column]
            )
            
            if self.prediction_result is None:
                print_error("预测失败")
                return False
            
            # 如果有申购和赎回数据，进行分解分析
            if has_purchase_redeem:
                self.decomposition_result = self._perform_decomposition_analysis(
                    date_column, value_column
                )
            
            print_success(f"预测完成，共{len(self.prediction_result['dates'])}步")
            if has_purchase_redeem:
                print_info("申购和赎回预测已启用")
            
            return True
            
        except Exception as e:
            print_error(f"预测失败: {e}")
            return False
    
    def _check_purchase_redeem_data(self):
        """检查是否有申购和赎回数据"""
        try:
            # 检查配置是否启用申购赎回预测
            purchase_redeem_config = self.config.get('申购赎回预测配置', {})
            if not purchase_redeem_config.get('启用', False):
                return False
            
            # 检查数据中是否有申购和赎回字段
            purchase_field = purchase_redeem_config.get('申购预测', {}).get('字段名', 'total_purchase_amt')
            redeem_field = purchase_redeem_config.get('赎回预测', {}).get('字段名', 'total_redeem_amt')
            
            has_purchase = purchase_field in self.data.columns
            has_redeem = redeem_field in self.data.columns
            
            if has_purchase and has_redeem:
                print_info(f"检测到申购字段: {purchase_field}")
                print_info(f"检测到赎回字段: {redeem_field}")
                return True
            else:
                print_warning(f"申购或赎回字段缺失: 申购={has_purchase}, 赎回={has_redeem}")
                return False
                
        except Exception as e:
            print_warning(f"检查申购赎回数据失败: {e}")
            return False
    
    def _get_last_history_date(self, date_column):
        """获取历史数据的最后日期"""
        try:
            if date_column in self.data.columns:
                # 确保日期列是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
                    self.data[date_column] = pd.to_datetime(self.data[date_column])
                
                last_date = self.data[date_column].max()
                print_info(f"历史数据最后日期: {last_date}")
                return last_date
            else:
                print_warning(f"未找到日期列: {date_column}")
                return None
                
        except Exception as e:
            print_warning(f"获取最后日期失败: {e}")
            return None
    
    def _perform_decomposition_analysis(self, date_column, value_column):
        """执行时间序列分解分析"""
        try:
            print_info("执行时间序列分解分析...")
            
            # 对净资金流进行分解
            if 'Net_Flow' in self.data.columns:
                decompose_column = 'Net_Flow'
            else:
                decompose_column = value_column
            
            decomposition_result = self.core.decompose_series(
                self.data,
                period_type="weekday",
                date_column=date_column,
                value_column=decompose_column
            )
            
            if decomposition_result:
                print_success("时间序列分解分析完成")
                return decomposition_result
            else:
                print_warning("时间序列分解分析失败")
                return None
                
        except Exception as e:
            print_warning(f"执行分解分析失败: {e}")
            return None
    
    def _step5_create_visualizations(self):
        """步骤5: 创建可视化图表"""
        print_info("步骤5: 创建可视化图表")
        
        try:
            # 检查是否启用可视化
            visualization_config = self.config.get('可视化配置', {})
            if not visualization_config.get('保存配置', {}).get('保存基础信息图', True):
                print_info("可视化功能已禁用")
                return True
            
            # 创建基础信息图
            if visualization_config.get('保存配置', {}).get('保存基础信息图', True):
                self.visualization.create_basic_info_plot(
                    self.data, self.basic_info, save_plot=True
                )
            
            # 创建周期因子图
            if visualization_config.get('保存配置', {}).get('保存周期因子图', True):
                self.visualization.create_periodic_factors_plot(
                    self.periodic_factors, save_plot=True
                )
            
            # 创建Base分析图
            if visualization_config.get('保存配置', {}).get('保存Base分析图', True):
                self.visualization.create_base_analysis_plot(
                    self.data, self.base_values, save_plot=True
                )
            
            # 创建预测结果图
            if visualization_config.get('保存配置', {}).get('保存预测结果图', True):
                self.visualization.create_prediction_plot(
                    self.data, self.prediction_result, save_plot=True
                )
            
            # 创建分解分析图
            if visualization_config.get('保存配置', {}).get('保存分解分析图', True):
                if self.decomposition_result:
                    self.visualization.create_decomposition_plot(
                        self.decomposition_result, save_plot=True
                    )
            
            # 创建申购赎回预测图（如果启用）
            if visualization_config.get('保存配置', {}).get('保存申购赎回预测图', True):
                if self.prediction_result and self.prediction_result.get('has_purchase_redeem', False):
                    # 申购赎回预测图
                    self.visualization.create_purchase_redeem_prediction_plot(
                        self.data, self.prediction_result, save_plot=True
                    )
                    
                    # 申购赎回对比分析图
                    self.visualization.create_purchase_redeem_comparison_plot(
                        self.data, self.prediction_result, save_plot=True
                    )
                    
                    print_success("申购赎回预测图表已创建")
                else:
                    print_info("未检测到申购赎回数据，跳过申购赎回图表创建")
            
            print_success("可视化图表创建完成")
            return True
            
        except Exception as e:
            print_error(f"创建可视化图表失败: {e}")
            return False
    
    def _step5_5_save_prediction_csv(self):
        """步骤5.5: 保存预测结果为CSV格式"""
        print_info("步骤5.5: 保存预测结果为CSV格式")
        
        try:
            if self.prediction_result is None:
                print_warning("没有预测结果可保存")
                return False
            
            # 创建输出目录
            output_dir = Path("output/classical_decomposition")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 准备保存数据
            save_data = []
            dates = self.prediction_result['dates']
            predictions = self.prediction_result['predictions']
            confidence_intervals = self.prediction_result['confidence_intervals']
            base_values = self.prediction_result['base_values']
            
            # 检查是否有申购和赎回数据
            has_purchase_redeem = self.prediction_result.get('has_purchase_redeem', False)
            purchase_predictions = self.prediction_result.get('purchase_predictions', [])
            redeem_predictions = self.prediction_result.get('redeem_predictions', [])
            
            for i, date in enumerate(dates):
                row_data = {
                    '预测日期': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    '预测值': predictions[i] if i < len(predictions) else None,
                    '置信区间下界': confidence_intervals[i][0] if i < len(confidence_intervals) else None,
                    '置信区间上界': confidence_intervals[i][1] if i < len(confidence_intervals) else None,
                    'Base值': base_values[i] if i < len(base_values) else None
                }
                
                # 如果有申购和赎回数据，添加相应字段
                if has_purchase_redeem:
                    row_data['申购预测'] = purchase_predictions[i] if i < len(purchase_predictions) else None
                    row_data['赎回预测'] = redeem_predictions[i] if i < len(redeem_predictions) else None
                    row_data['净资金流预测'] = (purchase_predictions[i] - redeem_predictions[i]) if i < len(purchase_predictions) and i < len(redeem_predictions) else None
                
                save_data.append(row_data)
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(save_data)
            
            # 保存主预测结果
            main_file_path = output_dir / "prediction_results.csv"
            df.to_csv(main_file_path, index=False, encoding='utf-8-sig')
            print_success(f"主预测结果已保存: {main_file_path}")
            
            # 如果有申购和赎回数据，单独保存详细结果
            if has_purchase_redeem:
                detailed_file_path = output_dir / "purchase_redeem_predictions.csv"
                
                # 创建详细的申购赎回预测结果
                detailed_data = []
                for i, date in enumerate(dates):
                    detailed_row = {
                        '预测日期': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        '申购预测': purchase_predictions[i] if i < len(purchase_predictions) else None,
                        '赎回预测': redeem_predictions[i] if i < len(redeem_predictions) else None,
                        '净资金流预测': (purchase_predictions[i] - redeem_predictions[i]) if i < len(purchase_predictions) and i < len(redeem_predictions) else None,
                        '申购赎回比例': (purchase_predictions[i] / redeem_predictions[i]) if i < len(purchase_predictions) and i < len(redeem_predictions) and redeem_predictions[i] != 0 else None,
                        '累计净资金流': sum([(p - r) for p, r in zip(purchase_predictions[:i+1], redeem_predictions[:i+1])]) if i < len(purchase_predictions) and i < len(redeem_predictions) else None
                    }
                    detailed_data.append(detailed_row)
                
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_csv(detailed_file_path, index=False, encoding='utf-8-sig')
                print_success(f"申购赎回详细预测结果已保存: {detailed_file_path}")

            # 额外输出一份与 ARIMA 一致的CSV（report_date,purchase,redeem，YYYYMMDD，UTF-8-SIG）
            try:
                from config import OUTPUT_DIR as _OUTPUT_DIR
                # 将日期统一为YYYYMMDD格式
                def _fmt_date(d):
                    try:
                        return d.strftime('%Y%m%d')
                    except Exception:
                        return pd.to_datetime(d).strftime('%Y%m%d')

                report_dates = [ _fmt_date(d) for d in dates ]

                if has_purchase_redeem and purchase_predictions and redeem_predictions:
                    purchase_vals = purchase_predictions
                    redeem_vals = redeem_predictions
                else:
                    # 若无申购/赎回预测，则使用主预测作为 purchase，占位 redeem=0，保证格式一致
                    purchase_vals = predictions
                    redeem_vals = [0 for _ in predictions]

                matched_df = pd.DataFrame({
                    'report_date': report_dates,
                    'purchase': purchase_vals,
                    'redeem': redeem_vals
                })

                matched_out_dir = _OUTPUT_DIR / "data"
                matched_out_dir.mkdir(parents=True, exist_ok=True)
                matched_file = matched_out_dir / "classical_forecast_201409.csv"
                matched_df.to_csv(matched_file, index=False, encoding='utf-8-sig')
                print_success(f"已额外输出与ARIMA格式一致的CSV: {matched_file}")
            except Exception as e:
                print_warning(f"输出与ARIMA一致格式CSV失败: {e}")
            
            return True
            
        except Exception as e:
            print_error(f"保存预测结果CSV失败: {e}")
            return False
    
    def _step6_evaluate_performance(self):
        """步骤6: 性能评估"""
        print_info("步骤6: 性能评估")
        
        try:
            # 创建基准模型
            baseline_types = self.config.get('性能评估', {}).get('基准模型', ['naive'])
            # 若为字符串，转为列表
            if isinstance(baseline_types, str):
                baseline_types = [baseline_types]
            value_column = self.prediction_params.get('value_column', 'value')
            for baseline_type in baseline_types:
                self.evaluator.create_baseline_model(
                    self.data[value_column].values, 
                    baseline_type=baseline_type
                )
            
            # 评估经典分解法模型
            # 使用历史数据进行回测评估
            if self.decomposition_result:
                actual_values = self.decomposition_result['original'].values
                predicted_values = (self.decomposition_result['trend'] + 
                                 (self.decomposition_result['seasonal'] - 1) * 
                                 self.decomposition_result['base']).values
                
                self.evaluator.evaluate_model_performance(
                    actual_values, 
                    predicted_values, 
                    "Classical_Decomposition"
                )
                
                # 区间质量评估（与其他内容隔开打印）
                try:
                    conf_level = self.prediction_params.get('confidence_level', 0.95)
                    self.evaluator.evaluate_interval_quality(
                        actual_values,
                        predicted_values,
                        model_name="Classical_Decomposition",
                        confidence_level=conf_level
                    )
                except Exception as _:
                    pass
                
                # 与基准模型比较
                self.evaluator.compare_with_baseline("Classical_Decomposition")
                
                # 生成评估报告
                evaluation_report = self.evaluator.generate_evaluation_report("Classical_Decomposition")
                
                # 保存评估结果
                self.evaluator.save_evaluation_results()
            
            print_success("性能评估完成")
            return True
            
        except Exception as e:
            print_error(f"性能评估失败: {e}")
            return False
    
    def get_prediction_results(self):
        """获取预测结果"""
        return {
            'prediction_result': self.prediction_result,
            'decomposition_result': self.decomposition_result,
            'periodic_factors': self.periodic_factors,
            'base_values': self.base_values,
            'basic_info': self.basic_info
        }
    
    def save_prediction_results(self, output_path=None):
        """保存预测结果"""
        if not output_path:
            output_path = "output/classical_decomposition/prediction_results.json"
        
        try:
            import json
            from pathlib import Path
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备保存的数据
            save_data = {
                '预测时间': datetime.now().isoformat(),
                '配置信息': self.config,
                '基础信息': self.basic_info,
                '周期因子': self.periodic_factors,
                '预测结果': self.prediction_result
            }
            
            # 转换numpy类型
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Series):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif obj is pd.NaT:
                    return None
                elif hasattr(obj, 'isoformat'):
                    # 处理原生datetime等
                    try:
                        return obj.isoformat()
                    except Exception:
                        pass
                # np.dtype 等对象
                try:
                    import numpy as _np
                    if isinstance(obj, _np.dtype):
                        return str(obj)
                except Exception:
                    pass
                return obj
            
            # 递归转换
            def recursive_convert(data):
                if isinstance(data, dict):
                    return {k: recursive_convert(v) for k, v in data.items()}
                elif isinstance(data, (list, tuple, set)):
                    return [recursive_convert(item) for item in list(data)]
                else:
                    return convert_numpy(data)
            
            converted_data = recursive_convert(save_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            
            print_success(f"预测结果已保存: {output_file}")
            return str(output_file)
            
        except Exception as e:
            print_error(f"保存预测结果失败: {e}")
            return None
    
    def print_summary(self):
        """打印预测结果摘要"""
        print_header("经典分解法预测结果摘要")
        
        try:
            if self.prediction_result is None:
                print_warning("没有预测结果可显示")
                return
            
            # 基本信息
            print_info("=== 预测基本信息 ===")
            print_info(f"预测步数: {len(self.prediction_result['dates'])}")
            print_info(f"预测日期范围: {self.prediction_result['dates'][0]} 至 {self.prediction_result['dates'][-1]}")
            
            # 主预测结果统计
            predictions = self.prediction_result['predictions']
            if predictions:
                print_info("=== 主预测结果统计 ===")
                print_info(f"预测值范围: {min(predictions):.2f} 至 {max(predictions):.2f}")
                print_info(f"预测值均值: {np.mean(predictions):.2f}")
                print_info(f"预测值标准差: {np.std(predictions):.2f}")
            
            # 申购和赎回预测结果（如果有）
            if self.prediction_result.get('has_purchase_redeem', False):
                print_info("=== 申购和赎回预测结果 ===")
                
                purchase_predictions = self.prediction_result.get('purchase_predictions', [])
                redeem_predictions = self.prediction_result.get('redeem_predictions', [])
                
                if purchase_predictions:
                    print_info("申购预测统计:")
                    print_info(f"  预测值范围: {min(purchase_predictions):.2f} 至 {max(purchase_predictions):.2f}")
                    print_info(f"  预测值均值: {np.mean(purchase_predictions):.2f}")
                    print_info(f"  预测值标准差: {np.std(purchase_predictions):.2f}")
                
                if redeem_predictions:
                    print_info("赎回预测统计:")
                    print_info(f"  预测值范围: {min(redeem_predictions):.2f} 至 {max(redeem_predictions):.2f}")
                    print_info(f"  预测值均值: {np.mean(redeem_predictions):.2f}")
                    print_info(f"  预测值标准差: {np.std(redeem_predictions):.2f}")
                
                # 净资金流分析
                if purchase_predictions and redeem_predictions:
                    net_flow_predictions = [p - r for p, r in zip(purchase_predictions, redeem_predictions)]
                    positive_days = sum(1 for nf in net_flow_predictions if nf > 0)
                    negative_days = sum(1 for nf in net_flow_predictions if nf < 0)
                    
                    print_info("净资金流分析:")
                    print_info(f"  净流入天数: {positive_days} ({positive_days/len(net_flow_predictions)*100:.1f}%)")
                    print_info(f"  净流出天数: {negative_days} ({negative_days/len(net_flow_predictions)*100:.1f}%)")
                    print_info(f"  净资金流范围: {min(net_flow_predictions):.2f} 至 {max(net_flow_predictions):.2f}")
                    print_info(f"  累计净资金流: {sum(net_flow_predictions):.2f}")
            
            # 置信区间信息
            confidence_intervals = self.prediction_result.get('confidence_intervals', [])
            if confidence_intervals:
                print_info("=== 置信区间信息 ===")
                avg_interval_width = np.mean([upper - lower for lower, upper in confidence_intervals])
                print_info(f"平均置信区间宽度: {avg_interval_width:.2f}")
            
            # 周期因子信息
            if self.periodic_factors:
                print_info("=== 周期因子信息 ===")
                for period_type, factors in self.periodic_factors.items():
                    if isinstance(factors, dict):
                        factor_values = list(factors.values())
                        print_info(f"{period_type}周期因子范围: {min(factor_values):.4f} 至 {max(factor_values):.4f}")
                        print_info(f"{period_type}周期因子标准差: {np.std(factor_values):.4f}")
            
            print_success("预测结果摘要打印完成")
            
        except Exception as e:
            print_error(f"打印摘要失败: {e}")
