# -*- coding: utf-8 -*-
"""
资金流预测系统主程序
"""

from utils.config_utils import ensure_directories, validate_config
from utils.interactive_utils import print_header, print_success, print_error, show_menu, wait_for_key
from src.analysis.basic_analysis import run_basic_data_analysis
from src.prediction.arima_predictor import run_arima_prediction
from src.data_processing import run_data_processing
from config import DATA_DIR
from utils.config_utils import config_management_menu
from utils.arima_config_manager import run_arima_config_manager

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
            "name": "ARIMA预测",
            "description": "使用ARIMA模型进行时间序列预测",
            "action": run_arima_prediction
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