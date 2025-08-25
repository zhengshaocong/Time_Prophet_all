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
from src.prediction.arima_predictor import run_arima_prediction
from src.prediction.prophet import run_prophet_prediction
from utils.config_utils import config_management_menu
from utils.interactive_utils import print_header, print_success, print_error
from src.data_processing import run_data_processing


def run_function_by_name(function_name):
    """根据功能名称运行对应的函数"""
    function_map = {
        'basic': run_basic_data_analysis,
        'preprocess': run_data_processing,
        'arima': run_arima_prediction,
        'prophet': run_prophet_prediction,
        'config': config_management_menu
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
  python3 run.py --preprocess       # 运行数据预处理
  python3 run.py --arima            # 运行ARIMA预测
  python3 run.py --prophet          # 运行Prophet预测
  python3 run.py --config           # 运行配置管理
        """
    )
    
    # 添加功能选择参数
    parser.add_argument('--basic', action='store_true', 
                       help='运行基础数据分析')
    parser.add_argument('--preprocess', action='store_true', 
                       help='运行数据预处理')
    parser.add_argument('--arima', action='store_true', 
                       help='运行ARIMA预测')
    parser.add_argument('--prophet', action='store_true', 
                       help='运行Prophet预测')
    parser.add_argument('--config', action='store_true', 
                       help='运行配置管理')
    
    args = parser.parse_args()
    
    # 检查是否有参数
    if not any([args.basic, args.preprocess, args.arima, args.prophet, args.config]):
        # 没有参数，启动交互式菜单
        main()
        return
    
    # 有参数，执行对应功能
    if args.basic:
        run_function_by_name('basic')
    elif args.preprocess:
        run_function_by_name('preprocess')
    elif args.arima:
        run_function_by_name('arima')
    elif args.prophet:
        run_function_by_name('prophet')
    elif args.config:
        run_function_by_name('config')


if __name__ == "__main__":
    main_with_args() 