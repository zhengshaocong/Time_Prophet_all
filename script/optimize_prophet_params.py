#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prophet 参数优化脚本
通过网格搜索找到最优参数组合，提高预测分数
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from itertools import product
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

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

def prepare_data(data, field_name):
    """准备Prophet数据格式"""
    df = data[['report_date', field_name]].copy()
    df['report_date'] = pd.to_datetime(df['report_date'])
    df = df.rename(columns={'report_date': 'ds', field_name: 'y'})
    df = df.sort_values('ds')
    
    # 去除异常值
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['y'] = df['y'].clip(lower_bound, upper_bound)
    
    return df

def evaluate_params(df, params):
    """评估单个参数组合"""
    try:
        # 创建模型
        model = Prophet(**params)
        model.fit(df)
        
        # 交叉验证
        df_cv = cross_validation(
            model, 
            initial="60 days",  # 初始训练期
            period="15 days",   # 切分间隔
            horizon="7 days"    # 预测期
        )
        
        # 计算性能指标
        df_p = performance_metrics(df_cv)
        
        # 返回关键指标
        return {
            'mae': df_p['mae'].mean(),
            'rmse': df_p['rmse'].mean(),
            'mape': df_p['mape'].mean(),
            'params': params
        }
    except Exception as e:
        print(f"参数评估失败: {e}")
        return None

def optimize_prophet_params():
    """优化Prophet参数"""
    print("=" * 60)
    print("Prophet 参数优化 - 寻找最优参数组合")
    print("=" * 60)
    
    # 加载数据
    data = load_data()
    if data is None:
        return
    
    # 准备申购和赎回数据
    purchase_df = prepare_data(data, 'total_purchase_amt')
    redeem_df = prepare_data(data, 'total_redeem_amt')
    
    print(f"申购数据: {len(purchase_df)} 行")
    print(f"赎回数据: {len(redeem_df)} 行")
    
    # 定义参数网格
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.2],
        'seasonality_prior_scale': [5, 8, 10, 12],
        'holidays_prior_scale': [5, 8, 10, 12],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_range': [0.7, 0.8, 0.85]
    }
    
    # 生成所有参数组合
    param_combinations = []
    for values in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), values))
        param_combinations.append(param_dict)
    
    print(f"总共 {len(param_combinations)} 个参数组合需要测试")
    
    # 测试申购模型参数
    print("\n" + "=" * 40)
    print("优化申购模型参数...")
    print("=" * 40)
    
    purchase_results = []
    for i, params in enumerate(param_combinations):
        print(f"测试申购参数组合 {i+1}/{len(param_combinations)}: {params}")
        result = evaluate_params(purchase_df, params)
        if result:
            purchase_results.append(result)
    
    # 测试赎回模型参数
    print("\n" + "=" * 40)
    print("优化赎回模型参数...")
    print("=" * 40)
    
    redeem_results = []
    for i, params in enumerate(param_combinations):
        print(f"测试赎回参数组合 {i+1}/{len(param_combinations)}: {params}")
        result = evaluate_params(redeem_df, params)
        if result:
            redeem_results.append(result)
    
    # 分析结果
    print("\n" + "=" * 60)
    print("参数优化结果分析")
    print("=" * 60)
    
    if purchase_results:
        # 按MAPE排序（越低越好）
        purchase_results.sort(key=lambda x: x['mape'])
        print("\n申购模型最优参数组合（按MAPE排序）:")
        for i, result in enumerate(purchase_results[:5]):
            print(f"{i+1}. MAPE: {result['mape']:.4f}, MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}")
            print(f"   参数: {result['params']}")
    
    if redeem_results:
        # 按MAPE排序（越低越好）
        redeem_results.sort(key=lambda x: x['mape'])
        print("\n赎回模型最优参数组合（按MAPE排序）:")
        for i, result in enumerate(redeem_results[:5]):
            print(f"{i+1}. MAPE: {result['mape']:.4f}, MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}")
            print(f"   参数: {result['params']}")
    
    # 保存结果
    output_dir = "output/prophet"
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
        print(f"申购模型预期MAPE: {best_purchase['mape']:.4f}")
        print(f"\n赎回模型推荐参数: {best_redeem['params']}")
        print(f"赎回模型预期MAPE: {best_redeem['mape']:.4f}")
        
        # 生成配置文件
        config_content = f"""# Prophet 优化后的参数配置
PROPHET_OPTIMIZED_PARAMS = {{
    "申购模型": {best_purchase['params']},
    "赎回模型": {best_redeem['params']},
    "预期性能": {{
        "申购MAPE": {best_purchase['mape']:.4f},
        "赎回MAPE": {best_redeem['mape']:.4f}
    }}
}}
"""
        
        with open(f"{output_dir}/optimized_params_config.py", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        print(f"\n优化后的参数配置已保存: {output_dir}/optimized_params_config.py")

if __name__ == "__main__":
    optimize_prophet_params()
