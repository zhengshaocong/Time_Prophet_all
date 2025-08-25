# -*- coding: utf-8 -*-
"""
经典分解法预测运行脚本
Classical Decomposition Prediction Runner Script

使用方法：
python script/run_classical_decomposition.py [数据文件路径] [日期列名] [值列名]

示例：
python script/run_classical_decomposition.py data/sample_data.csv date value
"""

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Added for np.round

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.classical_decomposition_config import CLASSICAL_DECOMPOSITION_CONFIG
from src.prediction.classical_decomposition import ClassicalDecompositionPredictor
from utils.interactive_utils import print_header, print_info, print_success, print_error


def main():
    """主函数"""
    print_header("经典分解法预测程序")
    
    # 获取命令行参数
    if len(sys.argv) < 2:
        print_error("使用方法: python script/run_classical_decomposition.py [数据文件路径]")
        print_info("示例: python script/run_classical_decomposition.py data/sample_data.csv")
        return False
    
    data_file_path = sys.argv[1]
    # 检查数据文件是否存在
    if not os.path.exists(data_file_path):
        print_error(f"数据文件不存在: {data_file_path}")
        return False
    
    # 读取数据，自动检测字段
    df = pd.read_csv(data_file_path)
    if 'report_date' in df.columns:
        date_column = 'report_date'
    elif 'date' in df.columns:
        date_column = 'date'
    else:
        print_error("未检测到日期列（report_date或date）")
        return False
    
    # 将日期列统一转换为datetime
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        df = df.sort_values(by=date_column)
    except Exception as e:
        print_error(f"日期列转换失败: {e}")
        return False
    
    # 检查申购和赎回字段
    purchase_col = 'total_purchase_amt' if 'total_purchase_amt' in df.columns else None
    redeem_col = 'total_redeem_amt' if 'total_redeem_amt' in df.columns else None
    if not purchase_col or not redeem_col:
        print_error("未检测到申购或赎回字段（total_purchase_amt, total_redeem_amt）")
        return False
    
    # 预测申购
    print_info("预测申购金额...")
    predictor_purchase = ClassicalDecompositionPredictor(CLASSICAL_DECOMPOSITION_CONFIG)
    ok_purchase = predictor_purchase.run_prediction_pipeline(
        data_file_path,
        date_column,
        purchase_col,
        period_type="weekday",
        remove_periodic_effect=True,
        smooth_window=3,
        forecast_steps=30,
        confidence_level=0.95
    )
    # 预测赎回
    print_info("预测赎回金额...")
    predictor_redeem = ClassicalDecompositionPredictor(CLASSICAL_DECOMPOSITION_CONFIG)
    ok_redeem = predictor_redeem.run_prediction_pipeline(
        data_file_path,
        date_column,
        redeem_col,
        period_type="weekday",
        remove_periodic_effect=True,
        smooth_window=3,
        forecast_steps=30,
        confidence_level=0.95
    )
    if not (ok_purchase and ok_redeem):
        print_error("申购或赎回预测失败")
        return False
    # 打印摘要
    predictor_purchase.print_summary()
    predictor_redeem.print_summary()
    # 保存结果
    predictor_purchase.save_prediction_results()
    predictor_redeem.save_prediction_results()
    
    # 可视化：画出两条预测线
    print_info("生成申购和赎回对比预测图...")
    plt.figure(figsize=(14, 8))
    
    # 历史曲线
    plt.plot(df[date_column], df[purchase_col], label='历史申购', color='blue', alpha=0.5)
    plt.plot(df[date_column], df[redeem_col], label='历史赎回', color='red', alpha=0.5)
    
    # 预测曲线（统一转为datetime）
    pred_dates = pd.to_datetime(predictor_purchase.prediction_result['dates'])
    purchase_pred = predictor_purchase.prediction_result['predictions']
    redeem_pred = predictor_redeem.prediction_result['predictions']
    plt.plot(pred_dates, purchase_pred, label='申购预测', color='blue', linestyle='--', linewidth=2)
    plt.plot(pred_dates, redeem_pred, label='赎回预测', color='red', linestyle='--', linewidth=2)
    
    plt.title('经典分解法-申购与赎回预测')
    plt.xlabel('日期')
    plt.ylabel('金额')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = 'output/classical_decomposition/purchase_redeem_compare.png'
    plt.savefig(out_path, dpi=150)
    print_success(f"申购赎回对比预测图已保存: {out_path}")

    # 导出合并CSV
    try:
        export_cfg = CLASSICAL_DECOMPOSITION_CONFIG.get('导出CSV', {})
        if export_cfg.get('启用', True):
            date_fmt = export_cfg.get('日期格式', '%Y%m%d')
            date_col_name = export_cfg.get('日期列名', 'report_date')
            pur_col_name = export_cfg.get('申购列名', 'purchase')
            red_col_name = export_cfg.get('赎回列名', 'redeem')
            decimals = int(export_cfg.get('小数位', 2))
            export_name = export_cfg.get('文件名', 'purchase_redeem_forecast.csv')

            out_dates = pd.to_datetime(predictor_purchase.prediction_result['dates'])
            out_dates_str = out_dates.strftime(date_fmt)
            export_df = pd.DataFrame({
                date_col_name: out_dates_str,
                pur_col_name: np.round(purchase_pred, decimals),
                red_col_name: np.round(redeem_pred, decimals)
            })
            csv_path = Path('output/classical_decomposition') / export_name
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            export_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print_success(f"合并CSV已导出: {csv_path}")
    except Exception as e:
        print_error(f"导出CSV失败: {e}")

    print_success("经典分解法申购赎回预测全部完成！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
