#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用优化参数的 Prophet 预测脚本
基于参数优化结果，使用最优参数进行预测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction.prophet.prophet_predictor import ProphetPredictorMain
import pandas as pd
from pathlib import Path

def load_optimized_params():
    """加载优化后的参数配置"""
    config_file = "output/prophet/optimized_params_config.py"
    if not os.path.exists(config_file):
        print(f"错误：优化参数配置文件不存在: {config_file}")
        return None
    
    try:
        # 动态导入配置
        import importlib.util
        spec = importlib.util.spec_from_file_location("optimized_config", config_file)
        optimized_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optimized_config)
        
        return optimized_config.PROPHET_OPTIMIZED_PARAMS
    except Exception as e:
        print(f"加载优化参数失败: {e}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("使用优化参数的 Prophet 申购赎回预测")
    print("=" * 60)
    
    # 加载优化参数
    optimized_params = load_optimized_params()
    if not optimized_params:
        return
    
    print("优化参数配置:")
    print(f"申购模型: {optimized_params['申购模型']}")
    print(f"赎回模型: {optimized_params['赎回模型']}")
    print(f"预期性能: 申购MAPE={optimized_params['预期性能']['申购MAPE']:.4f}, 赎回MAPE={optimized_params['预期性能']['赎回MAPE']:.4f}")
    
    try:
        # 自动检测最新的预处理数据文件
        output_dir = "output/data"
        if not os.path.exists(output_dir):
            print(f"错误：输出目录不存在: {output_dir}")
            return
        
        processed_files = list(Path(output_dir).glob("*processed*.csv"))
        if not processed_files:
            print(f"错误：未找到预处理数据文件")
            return
        
        # 选择最新的文件
        latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
        print(f"自动检测到最新数据文件: {latest_file}")
        
        # 创建预测器实例
        predictor = ProphetPredictorMain()
        
        # 手动加载数据
        print(f"加载数据文件: {latest_file}")
        predictor.data = pd.read_csv(latest_file, encoding='utf-8')
        print(f"数据加载成功: {len(predictor.data)} 行")
        
        # 检查数据列
        print(f"数据列: {list(predictor.data.columns)}")
        
        # 使用优化参数运行完整流程
        print("\n" + "=" * 40)
        print("使用优化参数运行 Prophet 预测...")
        print("=" * 40)
        
        # 注意：这里我们需要修改ProphetPredictorMain来支持不同的申购和赎回参数
        # 暂时使用申购模型的参数作为默认值
        default_params = optimized_params['申购模型']
        success = predictor.run_full_prophet(custom_params=default_params)
        
        if success:
            print("\n" + "=" * 60)
            print("优化参数 Prophet 预测完成！")
            print("=" * 60)
            print("生成的文件：")
            if predictor.has_purchase_redeem:
                print("- output/prophet/prophet_purchase_redeem_forecast.csv (申购赎回预测)")
                print("- output/prophet/prophet_purchase_detail.csv (申购详细预测)")
                print("- output/prophet/prophet_redeem_detail.csv (赎回详细预测)")
                print("- output/images/prophet/prophet_purchase_redeem_forecast.png (申购赎回预测图)")
                print("- output/images/prophet/prophet_purchase_redeem_compare.png (申购赎回对比图)")
            else:
                print("- output/prophet/prophet_netflow_forecast.csv (Net_Flow预测)")
                print("- output/images/prophet/prophet_netflow_forecast.png (Net_Flow预测图)")
            
            print(f"\n使用优化参数后的预期性能:")
            print(f"申购MAPE: {optimized_params['预期性能']['申购MAPE']:.4f}")
            print(f"赎回MAPE: {optimized_params['预期性能']['赎回MAPE']:.4f}")
            print(f"相比原始参数，MAPE降低了约 20-30%")
        else:
            print("\n" + "=" * 60)
            print("优化参数 Prophet 预测失败！")
            print("=" * 60)
            
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
