#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经典分解法参数优化脚本
通过网格搜索找到最优参数组合，提高预测分数
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入经典分解法相关模块
from src.prediction.classical_decomposition.core import ClassicalDecompositionCore
from src.prediction.classical_decomposition.data_processor import ClassicalDecompositionDataProcessor

def load_data():
    """加载数据"""
    # 自动检测最新的预处理数据文件
    output_dir = "output/data"
    if not os.path.exists(output_dir):
        print(f"错误：输出目录不存在: {output_dir}")
        return None
    
    processed_files = list(Path(output_dir).glob("*processed*.csv"))
    if not processed_files:
        print(f"错误：未找到预处理数据文件")
        return None
    
    # 选择最新的文件
    latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
    print(f"自动检测到最新数据文件: {latest_file}")
    
    try:
        data = pd.read_csv(latest_file, encoding='utf-8')
        print(f"数据加载成功: {len(data)} 行")
        print(f"数据列: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def prepare_data(data):
    """准备经典分解法数据"""
    # 检查是否有申购和赎回字段
    if 'total_purchase_amt' in data.columns and 'total_redeem_amt' in data.columns:
        # 准备申购数据
        purchase_data = data[['report_date', 'total_purchase_amt']].copy()
        purchase_data['report_date'] = pd.to_datetime(purchase_data['report_date'])
        purchase_data = purchase_data.rename(columns={'report_date': 'date', 'total_purchase_amt': 'value'})
        
        # 准备赎回数据
        redeem_data = data[['report_date', 'total_redeem_amt']].copy()
        redeem_data['report_date'] = pd.to_datetime(redeem_data['report_date'])
        redeem_data = redeem_data.rename(columns={'report_date': 'date', 'total_redeem_amt': 'value'})
        
        return purchase_data, redeem_data
    else:
        print("错误：数据中缺少申购或赎回字段")
        return None, None

def evaluate_params(data, params, field_name):
    """评估单个参数组合"""
    try:
        # 创建数据处理器
        processor = ClassicalDecompositionDataProcessor(config={})
        processor.data = data
        
        # 创建经典分解核心
        core = ClassicalDecompositionCore(config=params)
        
        # 执行分解分析
        processor.load_and_analyze_data("temp_data.csv", n_segments=5)
        
        # 计算周期因子
        periodic_factors = core.calculate_periodic_factors(
            data, 
            period_type=params.get("周期类型", "weekday"),
            date_column='date',
            value_column='value'
        )
        
        if periodic_factors is None:
            return None
        
        # 计算base值（简化版本，使用移动平均）
        window_size = params.get("平滑窗口", 3)
        base_values = data['value'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        
        # 生成预测（使用较小的预测步数进行快速评估）
        forecast_steps = min(30, len(data) // 3)  # 使用数据量的1/3作为预测步数
        predictions = core.predict_future(
            base_values=base_values,
            periodic_factors=periodic_factors,
            forecast_steps=forecast_steps,
            hist_original_values=data
        )
        
        if predictions is None:
            return None
        
        # 确保predictions是列表或数组
        try:
            if hasattr(predictions, '__len__'):
                predictions_len = len(predictions)
            else:
                predictions_len = 0
        except:
            predictions_len = 0
            
        if predictions_len == 0:
            return None
        
        # 从返回的字典中提取预测值
        try:
            if isinstance(predictions, dict) and 'predictions' in predictions:
                predictions_list = predictions['predictions']
            elif isinstance(predictions, list):
                predictions_list = predictions
            else:
                print(f"无法识别的predictions类型: {type(predictions)}")
                return None
            
            # 将predictions转换为numpy数组以便处理
            if isinstance(predictions_list, list):
                predictions_array = np.array(predictions_list)
            elif hasattr(predictions_list, 'values'):
                predictions_array = predictions_list.values
            else:
                predictions_array = np.array(predictions_list)
        except Exception as e:
            print(f"无法转换predictions为数组: {type(predictions)}, 错误: {e}")
            return None
        
        # 计算历史拟合误差（使用最后一部分数据作为验证集）
        validation_size = min(20, len(data) // 4)
        if validation_size > 0:
            actual = data['value'].iloc[-validation_size:].values
            predicted = predictions_array[:validation_size]
            
            # 确保长度匹配
            min_len = min(len(actual), len(predicted))
            if min_len > 0:
                actual = actual[:min_len]
                predicted = predicted[:min_len]
                
                # 计算性能指标
                mae = np.mean(np.abs(actual - predicted))
                mse = np.mean((actual - predicted) ** 2)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                
                return {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape,
                    'params': params
                }
        
        return None
        
    except Exception as e:
        print(f"参数评估失败: {e}")
        return None

def optimize_classical_decomposition_params():
    """优化经典分解法参数"""
    print("=" * 60)
    print("经典分解法参数优化 - 寻找最优参数组合")
    print("=" * 60)
    
    # 加载数据
    data = load_data()
    if data is None:
        return
    
    # 准备申购和赎回数据
    purchase_data, redeem_data = prepare_data(data)
    if purchase_data is None or redeem_data is None:
        return
    
    print(f"申购数据: {len(purchase_data)} 行")
    print(f"赎回数据: {len(redeem_data)} 行")
    
    # 定义参数网格（基于当前配置和优化经验）
    param_grid = {
        "周期类型": ["weekday", "monthly"],
        "周期长度": [7, 14, 30],
        "平滑窗口": [3, 5, 7],
        "权重计算方式": ["frequency", "average"],
        "平滑方法": ["moving_average", "exponential"],
        "趋势阻尼系数": [0.85, 0.9, 0.95, 0.98],
        "均值回归强度": [0.03, 0.05, 0.08, 0.1],
        "均值回归窗口": [20, 30, 45],
        "因子扰动比例": [0.01, 0.02, 0.05, 0.08],
        "残差噪声比例": [0.05, 0.08, 0.1, 0.15],
        "起点对齐衰减": [0.03, 0.05, 0.07, 0.1],
        "启用随机性": [True, False]
    }
    
    # 生成所有参数组合（为了避免组合过多，我们选择一些关键参数）
    key_param_grid = {
        "周期类型": ["weekday", "monthly"],
        "平滑窗口": [3, 5, 7],
        "趋势阻尼系数": [0.9, 0.95, 0.98],
        "均值回归强度": [0.03, 0.05, 0.08],
        "因子扰动比例": [0.02, 0.05, 0.08],
        "残差噪声比例": [0.05, 0.08, 0.1],
        "启用随机性": [True, False]
    }
    
    # 生成参数组合
    param_combinations = []
    for values in product(*key_param_grid.values()):
        param_dict = dict(zip(key_param_grid.keys(), values))
        # 添加固定参数
        param_dict.update({
            "周期长度": 7 if param_dict["周期类型"] == "weekday" else 30,
            "权重计算方式": "frequency",
            "平滑方法": "moving_average",
            "均值回归窗口": 30,
            "起点对齐衰减": 0.05,
            "随机种子": 42
        })
        param_combinations.append(param_dict)
    
    print(f"总共 {len(param_combinations)} 个参数组合需要测试")
    
    # 测试申购模型参数
    print("\n" + "=" * 40)
    print("优化申购模型参数...")
    print("=" * 40)
    
    purchase_results = []
    for i, params in enumerate(param_combinations):
        print(f"测试申购参数组合 {i+1}/{len(param_combinations)}: {params}")
        result = evaluate_params(purchase_data, params, "申购")
        if result:
            purchase_results.append(result)
            print(f"  结果: MAE={result['mae']:.2f}, MAPE={result['mape']:.2f}%")
    
    # 测试赎回模型参数
    print("\n" + "=" * 40)
    print("优化赎回模型参数...")
    print("=" * 40)
    
    redeem_results = []
    for i, params in enumerate(param_combinations):
        print(f"测试赎回参数组合 {i+1}/{len(param_combinations)}: {params}")
        result = evaluate_params(redeem_data, params, "赎回")
        if result:
            redeem_results.append(result)
            print(f"  结果: MAE={result['mae']:.2f}, MAPE={result['mape']:.2f}%")
    
    # 分析结果
    print("\n" + "=" * 60)
    print("参数优化结果分析")
    print("=" * 60)
    
    if purchase_results:
        # 按MAPE排序（越低越好）
        purchase_results.sort(key=lambda x: x['mape'])
        print("\n申购模型最优参数组合（按MAPE排序）:")
        for i, result in enumerate(purchase_results[:5]):
            print(f"{i+1}. MAPE: {result['mape']:.2f}%, MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}")
            print(f"   参数: {result['params']}")
    
    if redeem_results:
        # 按MAPE排序（越低越好）
        redeem_results.sort(key=lambda x: x['mape'])
        print("\n赎回模型最优参数组合（按MAPE排序）:")
        for i, result in enumerate(redeem_results[:5]):
            print(f"{i+1}. MAPE: {result['mape']:.2f}%, MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}")
            print(f"   参数: {result['params']}")
    
    # 保存结果
    output_dir = "output/classical_decomposition"
    os.makedirs(output_dir, exist_ok=True)
    
    if purchase_results:
        purchase_df_results = pd.DataFrame(purchase_results)
        purchase_df_results.to_csv(f"{output_dir}/purchase_optimization_results.csv", index=False)
        print(f"\n申购优化结果已保存: {output_dir}/purchase_optimization_results.csv")
    
    if redeem_results:
        redeem_df_results = pd.DataFrame(redeem_results)
        redeem_df_results.to_csv(f"{output_dir}/redeem_optimization_results.csv", index=False)
        print(f"赎回优化结果已保存: {output_dir}/redeem_optimization_results.csv")
    
    # 推荐最优参数
    print("\n" + "=" * 60)
    print("推荐的最优参数配置")
    print("=" * 60)
    
    if purchase_results and redeem_results:
        best_purchase = purchase_results[0]
        best_redeem = redeem_results[0]
        
        print(f"申购模型推荐参数: {best_purchase['params']}")
        print(f"申购模型预期MAPE: {best_purchase['mape']:.2f}%")
        print(f"\n赎回模型推荐参数: {best_redeem['params']}")
        print(f"赎回模型预期MAPE: {best_redeem['mape']:.2f}%")
        
        # 生成配置文件
        config_content = f"""# 经典分解法优化后的参数配置
CLASSICAL_DECOMPOSITION_OPTIMIZED_PARAMS = {{
    "申购模型": {best_purchase['params']},
    "赎回模型": {best_redeem['params']},
    "预期性能": {{
        "申购MAPE": {best_purchase['mape']:.2f},
        "赎回MAPE": {best_redeem['mape']:.2f}
    }}
}}
"""
        
        with open(f"{output_dir}/optimized_params_config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        print(f"\n优化后的参数配置已保存: {output_dir}/optimized_params_config.py")

if __name__ == "__main__":
    optimize_classical_decomposition_params()
