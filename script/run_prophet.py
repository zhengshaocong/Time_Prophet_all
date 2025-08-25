#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prophet 预测测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction.prophet.prophet_predictor import ProphetPredictorMain
import pandas as pd

def main():
    """主函数"""
    print("=" * 60)
    print("Prophet 申购赎回预测测试")
    print("=" * 60)
    
    try:
        # 直接指定数据文件路径
        data_file = "output/data/user_balance_table_processed_20250813_084407.csv"
        
        if not os.path.exists(data_file):
            print(f"错误：数据文件不存在: {data_file}")
            return
        
        # 创建预测器实例
        predictor = ProphetPredictorMain()
        
        # 手动加载数据
        print(f"加载数据文件: {data_file}")
        predictor.data = pd.read_csv(data_file, encoding='utf-8')
        print(f"数据加载成功: {len(predictor.data)} 行")
        
        # 检查数据列
        print(f"数据列: {list(predictor.data.columns)}")
        
        # 运行完整流程
        success = predictor.run_full_prophet()
        
        if success:
            print("\n" + "=" * 60)
            print("Prophet 预测完成！")
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
        else:
            print("\n" + "=" * 60)
            print("Prophet 预测失败！")
            print("=" * 60)
            
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
