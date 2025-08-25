#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用优化参数的经典分解法运行脚本
Classical Decomposition with Optimized Parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入经典分解法相关模块
from src.prediction.classical_decomposition.predictor import ClassicalDecompositionPredictor
from config.classical_decomposition_config import CLASSICAL_DECOMPOSITION_CONFIG

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
    latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
    print(f"使用数据文件: {latest_file}")
    
    try:
        data = pd.read_csv(latest_file)
        print(f"数据加载成功，共 {len(data)} 行")
        print(f"列名: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("经典分解法预测（使用优化参数）")
    print("=" * 60)
    
    # 加载数据
    data = load_data()
    if data is None:
        return
    
    # 检查数据列
    required_columns = ['report_date', 'total_purchase_amt', 'total_redeem_amt']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"错误：缺少必要的列: {missing_columns}")
        return
    
    # 数据预处理
    try:
        # 转换日期列
        data['report_date'] = pd.to_datetime(data['report_date'])
        data = data.sort_values('report_date').reset_index(drop=True)
        
        # 添加value列（用于兼容性）
        data['value'] = data['total_purchase_amt'] - data['total_redeem_amt']
        
        print(f"数据预处理完成")
        print(f"日期范围: {data['report_date'].min()} 到 {data['report_date'].max()}")
        print(f"申购金额范围: {data['total_purchase_amt'].min():.2f} 到 {data['total_purchase_amt'].max():.2f}")
        print(f"赎回金额范围: {data['total_redeem_amt'].min():.2f} 到 {data['total_redeem_amt'].max():.2f}")
        
    except Exception as e:
        print(f"数据预处理失败: {e}")
        return
    
    # 创建预测器
    try:
        print("\n创建经典分解法预测器...")
        predictor = ClassicalDecompositionPredictor(config=CLASSICAL_DECOMPOSITION_CONFIG)
        print("预测器创建成功")
        
    except Exception as e:
        print(f"预测器创建失败: {e}")
        return
    
    # 保存临时数据文件
    temp_data_file = "temp/classical_decomposition_temp_data.csv"
    os.makedirs("temp", exist_ok=True)
    data.to_csv(temp_data_file, index=False)
    print(f"临时数据文件已保存: {temp_data_file}")
    
    # 执行预测
    try:
        print("\n开始执行经典分解法预测...")
        result = predictor.run_prediction_pipeline(
            data_file_path=temp_data_file,
            date_column='report_date',
            value_column='Net_Flow',  # 使用Net_Flow列作为value列
            period_type='weekday',
            remove_periodic_effect=True,
            smooth_window=3,
            forecast_steps=30,
            confidence_level=0.95
        )
        
        if result:
            print("预测完成！")
            print(f"结果保存在: output/classical_decomposition/")
            
            # 检查生成的文件
            output_dir = Path("output/classical_decomposition")
            if output_dir.exists():
                files = list(output_dir.glob("*"))
                print(f"\n生成的文件:")
                for file in files:
                    if file.is_file():
                        size = file.stat().st_size / 1024  # KB
                        print(f"  - {file.name} ({size:.1f} KB)")
        else:
            print("预测失败")
            
    except Exception as e:
        print(f"预测执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理临时文件
    if os.path.exists(temp_data_file):
        os.remove(temp_data_file)
        print(f"临时文件已清理: {temp_data_file}")

if __name__ == "__main__":
    main()
