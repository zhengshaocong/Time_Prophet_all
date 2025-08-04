# -*- coding: utf-8 -*-
"""
启动文件
程序的主入口点
支持命令行参数选择功能
"""

import sys
import argparse
from main import main, show_main_menu
from src.analysis.basic_analysis import run_basic_data_analysis
from src.analysis.purchase_redemption_analysis import run_purchase_redemption_analysis
from src.prediction.cash_flow_predictor import run_prediction_analysis
from src.prediction.arima_predictor import run_arima_prediction
from utils.config_utils import config_management_menu
from utils.interactive_utils import print_header, print_success, print_error


def show_deep_analysis_placeholder():
    """深度分析功能占位符"""
    print_header("申购赎回深度分析", "模式识别和趋势预测")
    print("此功能正在开发中...")
    print_success("深度分析功能将在后续版本中实现")


def show_anomaly_detection_placeholder():
    """异常检测功能占位符"""
    print_header("申购赎回异常检测", "异常模式检测和异常值识别")
    print("此功能正在开发中...")
    print_success("异常检测功能将在后续版本中实现")


def show_data_preprocessing_placeholder():
    """数据预处理功能占位符"""
    print_header("数据预处理", "数据清洗和特征工程")
    print("此功能正在开发中...")
    print_success("预处理功能将在后续版本中实现")


def show_model_evaluation_placeholder():
    """模型评估功能占位符"""
    print_header("模型评估", "性能评估和准确性分析")
    print("此功能正在开发中...")
    print_success("评估功能将在后续版本中实现")


def run_function_by_name(function_name):
    """根据功能名称运行对应的函数"""
    function_map = {
        'basic': run_basic_data_analysis,
        'purchase': run_purchase_redemption_analysis,
        'purchase-basic': run_purchase_redemption_analysis,
        'purchase-deep': show_deep_analysis_placeholder,
        'purchase-anomaly': show_anomaly_detection_placeholder,
        'purchase-arima': run_arima_prediction,
        'config': config_management_menu,
        'prediction': run_prediction_analysis,
        'preprocess': show_data_preprocessing_placeholder,
        'evaluate': show_model_evaluation_placeholder
    }
    
    if function_name in function_map:
        try:
            function_map[function_name]()
            print_success(f"功能 '{function_name}' 执行完成")
        except Exception as e:
            print_error(f"执行功能 '{function_name}' 时出错: {e}")
    else:
        print_error(f"未知的功能名称: {function_name}")
        print("可用的功能:")
        for name, func in function_map.items():
            print(f"  --{name}: {func.__name__}")


def main_with_args():
    """带命令行参数的主函数"""
    parser = argparse.ArgumentParser(
        description='资金流预测系统 - Time Prophet v1.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python3 run.py                    # 启动交互式菜单
  python3 run.py --basic            # 运行基础数据分析
  python3 run.py --purchase         # 运行申购赎回分析（基础分析）
  python3 run.py --purchase-basic   # 运行申购赎回基础分析
  python3 run.py --purchase-deep    # 运行申购赎回深度分析
  python3 run.py --purchase-anomaly # 运行申购赎回异常检测
  python3 run.py --purchase-arima   # 运行ARIMA预测
  python3 run.py --config           # 运行配置管理
  python3 run.py --prediction       # 运行资金流预测
  python3 run.py --preprocess       # 运行数据预处理
  python3 run.py --evaluate         # 运行模型评估
        """
    )
    
    # 添加功能选择参数
    parser.add_argument('--basic', action='store_true', 
                       help='运行基础数据分析')
    parser.add_argument('--purchase', action='store_true', 
                       help='运行申购赎回分析（基础分析）')
    parser.add_argument('--purchase-basic', action='store_true', 
                       help='运行申购赎回基础分析')
    parser.add_argument('--purchase-deep', action='store_true', 
                       help='运行申购赎回深度分析')
    parser.add_argument('--purchase-anomaly', action='store_true', 
                       help='运行申购赎回异常检测')
    parser.add_argument('--purchase-arima', action='store_true', 
                       help='运行ARIMA预测')
    parser.add_argument('--config', action='store_true', 
                       help='运行配置管理')
    parser.add_argument('--prediction', action='store_true', 
                       help='运行资金流预测')
    parser.add_argument('--preprocess', action='store_true', 
                       help='运行数据预处理')
    parser.add_argument('--evaluate', action='store_true', 
                       help='运行模型评估')
    
    # 添加通用参数
    parser.add_argument('--version', action='version', version='Time Prophet v1.0')
    
    args = parser.parse_args()
    
    # 检查是否有任何功能参数被指定
    specified_functions = [
        args.basic, args.purchase, args.purchase_basic, args.purchase_deep, args.purchase_anomaly,
        args.purchase_arima, args.config, args.prediction, args.preprocess, args.evaluate
    ]
    
    if any(specified_functions):
        # 如果指定了功能参数，直接运行对应的功能
        print_header("资金流预测系统", "Time Prophet v1.0")
        
        if args.basic:
            run_function_by_name('basic')
        elif args.purchase or args.purchase_basic:
            run_function_by_name('purchase-basic')
        elif args.purchase_deep:
            run_function_by_name('purchase-deep')
        elif args.purchase_anomaly:
            run_function_by_name('purchase-anomaly')
        elif args.purchase_arima:
            run_function_by_name('purchase-arima')
        elif args.config:
            run_function_by_name('config')
        elif args.prediction:
            run_function_by_name('prediction')
        elif args.preprocess:
            run_function_by_name('preprocess')
        elif args.evaluate:
            run_function_by_name('evaluate')
    else:
        # 如果没有指定参数，启动交互式菜单
        main()


if __name__ == "__main__":
    main_with_args() 