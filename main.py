# -*- coding: utf-8 -*-
"""
资金流预测系统主程序
"""

from utils.config_utils import ensure_directories, validate_config
import pandas as pd
from utils.interactive_utils import print_header, print_success, print_error, show_menu, wait_for_key
from src.analysis.basic_analysis import run_basic_data_analysis
from src.prediction.arima_predictor import run_arima_prediction
from src.prediction.prophet import run_prophet_prediction
from src.prediction.classical_decomposition import ClassicalDecompositionPredictor
from src.prediction.fusion_predictor import run_fusion_prediction
from config.classical_decomposition_config import CLASSICAL_DECOMPOSITION_CONFIG
from utils.interactive_utils import print_info
from utils.data_processing_manager import get_processed_data_path
from src.data_processing import run_data_processing
from config import DATA_DIR
from utils.config_utils import config_management_menu
from utils.arima_config_manager import run_arima_config_manager


def run_classical_decomposition_prediction():
    """运行经典分解法预测"""
    print_header("经典分解法预测")
    try:
        # 自动使用最新数据预处理结果
        processed_path = get_processed_data_path("classical_decomposition")
        if not processed_path:
            print_error("未找到预处理结果，请先在主菜单运行【数据预处理】以生成处理后的数据")
            return

        # 读取预处理数据，仅用于列名检测
        df = pd.read_csv(processed_path)
        # 选择日期列与数值列（遵循系统标准字段）
        if 'report_date' in df.columns:
            date_col = 'report_date'
        elif 'date' in df.columns:
            date_col = 'date'
        else:
            print_error("预处理数据中未找到日期列（期望 'report_date' 或 'date'），请检查预处理配置")
            return

        if 'Net_Flow' in df.columns:
            value_col = 'Net_Flow'
        elif 'value' in df.columns:
            value_col = 'value'
        else:
            print_error("预处理数据中未找到数值列（期望 'Net_Flow' 或 'value'），请检查预处理流程")
            return

        predictor = ClassicalDecompositionPredictor(CLASSICAL_DECOMPOSITION_CONFIG)
        ok = predictor.run_prediction_pipeline(
            processed_path, 
            date_col, 
            value_col,
            period_type="weekday",
            remove_periodic_effect=True,
            smooth_window=3,
            forecast_steps=30,
            confidence_level=0.95
        )
        if ok:
            predictor.print_summary()
            predictor.save_prediction_results()
            print_success("经典分解法预测完成")
        else:
            print_error("经典分解法预测失败")
    except KeyboardInterrupt:
        print_info("已取消")
    except Exception as e:
        print_error(f"运行失败: {e}")


def main():
    """主程序入口"""
    print_header("资金流预测系统", "Time Prophet v1.0")
    
    # 确保目录存在
    ensure_directories()
    
    # 验证配置
    if not validate_config():
        print_error("配置验证失败，程序退出")
        return
    
    # 显示主菜单
    show_main_menu()


def show_main_menu():
    """显示主菜单"""
    menu_items = [
        {
            "name": "基础数据分析",
            "description": "读取数据、解析字段、生成可视化图表",
            "action": run_basic_data_analysis
        },
        {
            "name": "数据预处理",
            "description": "数据清洗、特征工程等预处理操作",
            "action": run_data_processing
        },
        {
            "name": "ARIMA预测（增强版）",
            "description": "集成周期性因子的ARIMA/SARIMA预测，自动检测季节性并优化模型",
            "action": run_arima_prediction
        },

        {
            "name": "Prophet预测",
            "description": "使用Prophet模型进行时间序列预测",
            "action": run_prophet_prediction
        },
        {
            "name": "经典分解法预测",
            "description": "使用经典分解法进行时间序列预测",
            "action": run_classical_decomposition_prediction
        },
        {
            "name": "融合预测",
            "description": "Prophet + ARIMA 加权融合预测（目标120分）",
            "action": run_fusion_prediction
        },
        {
            "name": "配置管理",
            "description": "查看和修改数据源配置",
            "action": config_management_menu
        },
        {
            "name": "退出程序",
            "description": "退出程序",
            "action": exit_program
        }
    ]
    
    while True:
        selected_action = show_menu(menu_items, "资金流预测系统主菜单")
        
        if selected_action is None:
            # 用户选择了退出
            exit_program()
        else:
            # 执行选中的功能
            try:
                selected_action()
                # 功能执行完成后等待用户确认
                wait_for_key("按回车键返回主菜单...")
            except Exception as e:
                print_error(f"执行功能时出错: {e}")
                wait_for_key("按回车键返回主菜单...")


def exit_program():
    """退出程序"""
    print_success("感谢使用资金流预测系统！")
    exit(0)


if __name__ == "__main__":
    main()