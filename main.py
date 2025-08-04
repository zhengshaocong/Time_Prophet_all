# -*- coding: utf-8 -*-
"""
资金流预测系统主程序
"""

from utils.config_utils import ensure_directories, validate_config
from utils.interactive_utils import print_header, print_success, print_error, show_menu, wait_for_key
from src.analysis.basic_analysis import run_basic_data_analysis
from src.analysis.purchase_redemption_analysis import run_purchase_redemption_analysis
from src.prediction.cash_flow_predictor import run_prediction_analysis
from src.prediction.arima_predictor import run_arima_prediction
from src.data_processing import run_data_processing
from utils.config_utils import config_management_menu

# 这个函数现在从模块中导入，不需要重复定义


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
            "name": "申购赎回分析",
            "description": "分析申购与赎回的时间序列模式",
            "action": show_purchase_redemption_submenu
        },
        {
            "name": "配置管理",
            "description": "查看和修改数据源配置",
            "action": config_management_menu
        },
        {
            "name": "资金流预测",
            "description": "训练预测模型并进行资金流预测",
            "action": run_prediction_analysis
        },
        {
            "name": "数据预处理",
            "description": "数据清洗、特征工程等预处理操作",
            "action": run_data_processing
        },
        {
            "name": "模型评估",
            "description": "评估预测模型的性能和准确性",
            "action": show_model_evaluation_placeholder
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


def show_purchase_redemption_submenu():
    """申购赎回分析二级菜单"""
    submenu_items = [
        {
            "name": "基础分析",
            "description": "申购与赎回的时间序列对比和分布分析",
            "action": run_purchase_redemption_analysis
        },
        {
            "name": "深度分析",
            "description": "申购赎回模式识别和趋势预测",
            "action": show_deep_analysis_placeholder
        },
        {
            "name": "异常检测",
            "description": "检测申购赎回异常模式和异常值",
            "action": show_anomaly_detection_placeholder
        },
        {
            "name": "ARIMA预测",
            "description": "使用ARIMA模型进行申购赎回趋势预测",
            "action": run_arima_prediction
        },
        {
            "name": "返回主菜单",
            "description": "返回主菜单",
            "action": "return"
        }
    ]
    
    while True:
        selected_action = show_menu(submenu_items, "申购赎回分析子菜单")
        
        if selected_action is None:
            # 用户选择了退出
            break
        elif selected_action == "return":
            # 用户选择了返回主菜单
            break
        else:
            # 执行选中的功能
            try:
                selected_action()
                # 功能执行完成后等待用户确认
                wait_for_key("按回车键返回申购赎回分析菜单...")
            except Exception as e:
                print_error(f"执行功能时出错: {e}")
                wait_for_key("按回车键返回申购赎回分析菜单...")


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


def exit_program():
    """退出程序"""
    print_success("感谢使用资金流预测系统！")
    exit(0)


if __name__ == "__main__":
    main() 