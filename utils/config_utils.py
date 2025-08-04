# -*- coding: utf-8 -*-
"""
配置工具模块
提供配置管理、验证、获取等功能
"""

import json
from pathlib import Path
from config import (
    DATA_DIR, CACHE_DIR, TEMP_DIR, OUTPUT_DIR, IMAGES_DIR, OUTPUT_DATA_DIR,
    UTILS_DIR, SRC_DIR, SCRIPT_DIR, TESTS_DIR, DEFAULT_FIELD_MAPPING,
    DATA_PREPROCESSING_CONFIG, ARIMA_TRAINING_CONFIG, CURRENT_DATA_SOURCE
)
from utils.interactive_utils import print_header, print_success, print_error, print_info


def ensure_directories():
    """确保所有必要的目录都存在"""
    directories = [
        DATA_DIR, CACHE_DIR, TEMP_DIR,
        OUTPUT_DIR, IMAGES_DIR, OUTPUT_DATA_DIR,
        UTILS_DIR, SRC_DIR, SCRIPT_DIR, TESTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"确保目录存在: {directory}")


def get_config_info():
    """获取配置信息"""
    return {
        "数据目录": str(DATA_DIR),
        "缓存目录": str(CACHE_DIR),
        "临时目录": str(TEMP_DIR),
        "输出目录": str(OUTPUT_DIR),
        "图片输出目录": str(IMAGES_DIR),
        "数据输出目录": str(OUTPUT_DATA_DIR),
        "工具目录": str(UTILS_DIR),
        "源码目录": str(SRC_DIR),
        "脚本目录": str(SCRIPT_DIR),
        "测试目录": str(TESTS_DIR)
    }


def get_data_source_dispose_file(data_source_name):
    """
    获取数据源的dispose配置文件路径
    
    Args:
        data_source_name: 数据源名称（不包含扩展名）
        
    Returns:
        Path: dispose配置文件路径
    """
    return DATA_DIR / f"{data_source_name}_dispose.json"


def load_data_source_dispose_config(data_source_name):
    """
    加载数据源的dispose配置文件
    
    Args:
        data_source_name: 数据源名称（不包含扩展名）
        
    Returns:
        dict: 配置字典，如果文件不存在返回None
    """
    dispose_file = get_data_source_dispose_file(data_source_name)
    
    if not dispose_file.exists():
        return None
    
    try:
        with open(dispose_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print_error(f"读取数据源配置文件失败: {e}")
        return None


def get_data_field_mapping(data_source=None):
    """
    获取数据字段映射配置
    
    Args:
        data_source: 数据源名称，如果为None则自动检测数据源
        
    Returns:
        dict: 字段映射配置
    """
    # 如果没有指定数据源，尝试自动检测
    if data_source is None:
        # 查找data目录中的dispose配置文件
        dispose_files = list(DATA_DIR.glob("*_dispose.json"))
        if dispose_files:
            # 使用第一个找到的配置文件
            data_source = dispose_files[0].stem.replace("_dispose", "")
            print(f"自动检测到数据源: {data_source}")
    
    # 尝试从数据源特定的dispose文件读取
    if data_source:
        dispose_config = load_data_source_dispose_config(data_source)
        if dispose_config and "field_mapping" in dispose_config:
            return dispose_config["field_mapping"]
    
    # 如果没有特定配置，返回默认配置
    print("⚠ 未找到数据源配置文件，请先运行基础数据分析生成配置")
    return DEFAULT_FIELD_MAPPING


def get_field_name(field_type, data_source=None):
    """
    获取指定类型的字段名
    
    Args:
        field_type: 字段类型（如"时间字段"、"申购金额字段"等）
        data_source: 数据源名称，如果为None则自动检测
        
    Returns:
        str: 字段名
    """
    mapping = get_data_field_mapping(data_source)
    return mapping.get(field_type)


def get_time_format(data_source=None):
    """
    获取时间格式
    
    Args:
        data_source: 数据源名称，如果为None则自动检测
        
    Returns:
        str: 时间格式字符串
    """
    mapping = get_data_field_mapping(data_source)
    return mapping.get("时间格式")


def check_data_source_dispose_config(data_source_name):
    """
    检查数据源是否有dispose配置文件
    
    Args:
        data_source_name: 数据源名称（不包含扩展名）
        
    Returns:
        bool: 是否存在配置文件
    """
    dispose_file = get_data_source_dispose_file(data_source_name)
    return dispose_file.exists()


def get_missing_dispose_config_message(data_source_name):
    """
    获取缺失dispose配置文件的提示信息
    
    Args:
        data_source_name: 数据源名称（不包含扩展名）
        
    Returns:
        str: 提示信息
    """
    return f"""
⚠ 未找到数据源配置文件: {data_source_name}_dispose.json

请先运行基础数据分析功能来生成数据源特定的配置。
基础数据分析将自动检测数据字段并生成相应的配置。

建议操作：
1. 运行基础数据分析功能
2. 系统将自动生成 {data_source_name}_dispose.json 配置文件
3. 然后重新运行当前功能
"""


def get_preprocessing_config():
    """
    获取数据预处理配置
    
    Returns:
        dict: 预处理配置
    """
    return DATA_PREPROCESSING_CONFIG


def get_arima_config():
    """
    获取ARIMA模型配置
    
    Returns:
        dict: ARIMA配置
    """
    return ARIMA_TRAINING_CONFIG


def get_arima_training_range():
    """
    获取ARIMA训练时间范围配置
    
    Returns:
        dict: 训练时间范围配置
    """
    # 重新导入config模块以确保获取最新配置
    import importlib
    import config
    importlib.reload(config)
    return config.ARIMA_TRAINING_CONFIG["训练时间范围"]


def get_arima_data_limits():
    """
    获取ARIMA数据极限配置
    
    Returns:
        dict: 数据极限配置
    """
    # 重新导入config模块以确保获取最新配置
    import importlib
    import config
    importlib.reload(config)
    return config.ARIMA_TRAINING_CONFIG["数据极限"]


def get_arima_model_params():
    """
    获取ARIMA模型参数配置
    
    Returns:
        dict: 模型参数配置
    """
    return ARIMA_TRAINING_CONFIG["模型参数"]


def get_arima_prediction_config():
    """
    获取ARIMA预测配置
    
    Returns:
        dict: 预测配置
    """
    return ARIMA_TRAINING_CONFIG["预测配置"]


def get_arima_evaluation_config():
    """
    获取ARIMA性能评估配置
    
    Returns:
        dict: 性能评估配置
    """
    return ARIMA_TRAINING_CONFIG["性能评估"]


def update_arima_training_range(start_date=None, end_date=None, auto_detect=None):
    """
    更新ARIMA训练时间范围
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        auto_detect: 是否自动检测
        
    Returns:
        bool: 是否更新成功
    """
    global ARIMA_TRAINING_CONFIG
    
    try:
        if start_date is not None:
            ARIMA_TRAINING_CONFIG["训练时间范围"]["开始日期"] = start_date
        if end_date is not None:
            ARIMA_TRAINING_CONFIG["训练时间范围"]["结束日期"] = end_date
        if auto_detect is not None:
            ARIMA_TRAINING_CONFIG["训练时间范围"]["自动检测"] = auto_detect
        
        print_success("ARIMA训练时间范围配置已更新")
        return True
    except Exception as e:
        print_error(f"更新ARIMA训练时间范围失败: {e}")
        return False


def update_arima_data_limits(min_data=None, max_data=None, sampling_enabled=None):
    """
    更新ARIMA数据极限配置
    
    Args:
        min_data: 最小数据量
        max_data: 最大数据量
        sampling_enabled: 是否启用采样
        
    Returns:
        bool: 是否更新成功
    """
    global ARIMA_TRAINING_CONFIG
    
    try:
        if min_data is not None:
            ARIMA_TRAINING_CONFIG["数据极限"]["最小数据量"] = min_data
        if max_data is not None:
            ARIMA_TRAINING_CONFIG["数据极限"]["最大数据量"] = max_data
        if sampling_enabled is not None:
            ARIMA_TRAINING_CONFIG["数据极限"]["数据采样"]["启用采样"] = sampling_enabled
        
        print_success("ARIMA数据极限配置已更新")
        return True
    except Exception as e:
        print_error(f"更新ARIMA数据极限失败: {e}")
        return False


def switch_data_source(new_data_source):
    """
    切换数据源配置
    
    Args:
        new_data_source: 新的数据源名称
        
    Returns:
        bool: 是否切换成功
    """
    global CURRENT_DATA_SOURCE
    
    if new_data_source not in ["auto_detected"]:
        print(f"错误: 未找到数据源配置 '{new_data_source}'")
        print(f"可用的数据源: {list_available_data_sources()}")
        return False
    
    CURRENT_DATA_SOURCE = new_data_source
    print(f"已切换到数据源: {new_data_source}")
    return True


def list_available_data_sources():
    """
    列出所有可用的数据源
    
    Returns:
        list: 数据源名称列表
    """
    # 现在只返回自动检测的数据源
    return ["auto_detected"]


def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查必要的目录是否可写
    for dir_name, dir_path in [
        ("数据目录", DATA_DIR),
        ("缓存目录", CACHE_DIR),
        ("输出目录", OUTPUT_DIR)
    ]:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            # 测试写入权限
            test_file = dir_path / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            errors.append(f"{dir_name} ({dir_path}) 不可写: {e}")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("配置验证通过")
    return True


# ==================== 配置管理菜单函数 ====================
def show_current_config():
    """显示当前配置"""
    print_header("当前配置信息")
    
    config_info = get_config_info()
    for key, value in config_info.items():
        print(f"  {key}: {value}")
    
    # 显示字段映射信息
    # 现在直接从数据源配置文件读取
    if check_data_source_dispose_config(CURRENT_DATA_SOURCE):
        dispose_config = load_data_source_dispose_config(CURRENT_DATA_SOURCE)
        if dispose_config and "field_mapping" in dispose_config:
            field_mapping = dispose_config["field_mapping"]
            print(f"\n字段映射: 已加载 ({len(field_mapping)} 个字段)")
            for field_type, field_name in field_mapping.items():
                print(f"  {field_type}: {field_name}")
        else:
            print("\n字段映射: 未找到，请先运行基础数据分析")
    else:
        print("\n字段映射: 未找到，请先运行基础数据分析")


def show_all_data_sources():
    """显示所有可用的数据源"""
    print_header("可用数据源")
    
    data_sources = list_available_data_sources()
    if data_sources:
        for i, source in enumerate(data_sources, 1):
            print(f"  {i}. {source}")
    else:
        print("  暂无可用数据源")


def change_data_source():
    """切换数据源"""
    print_header("切换数据源")
    
    data_sources = list_available_data_sources()
    if not data_sources:
        print_error("没有可用的数据源")
        return
    
    print("可用的数据源:")
    for i, source in enumerate(data_sources, 1):
        print(f"  {i}. {source}")
    
    try:
        choice = input(f"\n请选择数据源 (1-{len(data_sources)}): ").strip()
        if not choice:
            print_info("取消操作")
            return
        
        index = int(choice) - 1
        if 0 <= index < len(data_sources):
            new_source = data_sources[index]
            if switch_data_source(new_source):
                print_success(f"已切换到数据源: {new_source}")
            else:
                print_error("切换数据源失败")
        else:
            print_error("无效的选择")
    except ValueError:
        print_error("请输入有效的数字")


def add_new_data_source():
    """添加新的数据源"""
    print_header("添加新数据源")
    print_info("当前系统使用自动检测机制，无需手动添加数据源")
    print_info("请运行基础数据分析来自动检测和配置数据源")


def validate_data_structure():
    """验证数据结构"""
    print_header("验证数据结构")
    
    # 检查数据源配置文件
    if not check_data_source_dispose_config(CURRENT_DATA_SOURCE):
        print_error(f"未找到数据源配置文件: {CURRENT_DATA_SOURCE}_dispose.json")
        print_info("请先运行基础数据分析来生成数据源特定的配置")
        return False
    
    dispose_config = load_data_source_dispose_config(CURRENT_DATA_SOURCE)
    if not dispose_config:
        print_error("加载数据源配置失败")
        return False
    
    field_mapping = dispose_config.get("field_mapping")
    if not field_mapping:
        print_error("数据源配置中未找到字段映射")
        return False
    
    # 检查必要字段
    required_fields = ["时间字段", "申购金额字段", "赎回金额字段"]
    missing_fields = []
    
    for field in required_fields:
        if field not in field_mapping or not field_mapping[field]:
            missing_fields.append(field)
    
    if missing_fields:
        print_error(f"缺少必要字段: {', '.join(missing_fields)}")
        print_info("请重新运行基础数据分析")
        return False
    
    print_success("数据结构验证通过")
    return True


def arima_config_menu():
    """ARIMA模型配置菜单"""
    while True:
        print_header("ARIMA模型配置", "选择操作")
        print("1. 查看ARIMA配置")
        print("2. 修改训练时间范围")
        print("3. 修改数据极限")
        print("4. 修改模型参数")
        print("5. 修改预测配置")
        print("6. 修改评估配置")
        print("0. 返回配置管理")
        
        try:
            choice = input("\n请选择操作 (0-6): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                show_arima_config()
            elif choice == "2":
                modify_arima_training_range()
            elif choice == "3":
                modify_arima_data_limits()
            elif choice == "4":
                modify_arima_model_params()
            elif choice == "5":
                modify_arima_prediction_config()
            elif choice == "6":
                modify_arima_evaluation_config()
            else:
                print_error("无效的选择")
        
        except KeyboardInterrupt:
            print("\n操作已取消")
            break
        except Exception as e:
            print_error(f"操作失败: {e}")
    
    input("\n按回车键继续...")


def show_arima_config():
    """显示ARIMA配置"""
    print_header("ARIMA模型配置")
    
    config = get_arima_config()
    
    # 显示训练时间范围
    print("=== 训练时间范围 ===")
    training_range = config["训练时间范围"]
    print(f"  开始日期: {training_range['开始日期']}")
    print(f"  结束日期: {training_range['结束日期']}")
    print(f"  时间格式: {training_range['时间格式']}")
    print(f"  自动检测: {training_range['自动检测']}")
    print(f"  最小训练天数: {training_range['最小训练天数']}")
    print(f"  最大训练天数: {training_range['最大训练天数']}")
        
    # 显示数据极限
    print("\n=== 数据极限 ===")
    data_limits = config["数据极限"]
    print(f"  最小数据量: {data_limits['最小数据量']}")
    print(f"  最大数据量: {data_limits['最大数据量']}")
    print(f"  启用采样: {data_limits['数据采样']['启用采样']}")
    print(f"  采样比例: {data_limits['数据采样']['采样比例']}")
    print(f"  采样方式: {data_limits['数据采样']['采样方式']}")
    print(f"  最小完整度: {data_limits['数据质量']['最小完整度']}")
    print(f"  最大缺失值比例: {data_limits['数据质量']['最大缺失值比例']}")
    print(f"  异常值处理: {data_limits['数据质量']['异常值处理']}")
    
    # 显示模型参数
    print("\n=== 模型参数 ===")
    model_params = config["模型参数"]
    print(f"  自动检测差分: {model_params['差分阶数']['自动检测']}")
    print(f"  最大差分: {model_params['差分阶数']['最大差分']}")
    print(f"  季节性差分: {model_params['差分阶数']['季节性差分']}")
    print(f"  AR阶数范围: {model_params['ARIMA参数']['p_range']}")
    print(f"  差分阶数范围: {model_params['ARIMA参数']['d_range']}")
    print(f"  MA阶数范围: {model_params['ARIMA参数']['q_range']}")
    print(f"  信息准则: {model_params['模型选择']['信息准则']}")
    print(f"  启用交叉验证: {model_params['模型选择']['交叉验证']['启用']}")
    print(f"  交叉验证折数: {model_params['模型选择']['交叉验证']['折数']}")
        
    # 显示预测配置
    print("\n=== 预测配置 ===")
    prediction_config = config["预测配置"]
    print(f"  预测步数: {prediction_config['预测步数']}")
    print(f"  置信区间: {prediction_config['置信区间']}")
    print(f"  预测频率: {prediction_config['预测频率']}")
    print(f"  包含置信区间: {prediction_config['输出格式']['包含置信区间']}")
    print(f"  包含历史数据: {prediction_config['输出格式']['包含历史数据']}")
    print(f"  格式化输出: {prediction_config['输出格式']['格式化输出']}")
        
    # 显示性能评估
    print("\n=== 性能评估 ===")
    evaluation_config = config["性能评估"]
    print(f"  评估指标: {evaluation_config['评估指标']}")
    print(f"  基准模型: {evaluation_config['基准模型']}")
    print(f"  启用模型比较: {evaluation_config['模型比较']['启用']}")
    print(f"  比较模型: {evaluation_config['模型比较']['比较模型']}")
    print(f"  可视化: {evaluation_config['模型比较']['可视化']}")


def modify_arima_training_range():
    """修改ARIMA训练时间范围"""
    print_header("修改ARIMA训练时间范围")
    
    current_config = get_arima_training_range()
    print(f"当前配置:")
    print(f"  开始日期: {current_config['开始日期']}")
    print(f"  结束日期: {current_config['结束日期']}")
    print(f"  自动检测: {current_config['自动检测']}")
    
    try:
        # 修改开始日期
        new_start = input(f"\n新的开始日期 (当前: {current_config['开始日期']}, 回车跳过): ").strip()
        if not new_start:
            new_start = None
        
        # 修改结束日期
        new_end = input(f"新的结束日期 (当前: {current_config['结束日期']}, 回车跳过): ").strip()
        if not new_end:
            new_end = None
        
        # 修改自动检测
        auto_detect_input = input(f"是否自动检测 (当前: {current_config['自动检测']}, y/n, 回车跳过): ").strip().lower()
        if auto_detect_input == 'y':
            new_auto_detect = True
        elif auto_detect_input == 'n':
            new_auto_detect = False
        else:
            new_auto_detect = None
        
        if update_arima_training_range(new_start, new_end, new_auto_detect):
            print_success("训练时间范围配置已更新")
        else:
            print_error("更新失败")
            
    except Exception as e:
        print_error(f"修改失败: {e}")


def modify_arima_data_limits():
    """修改ARIMA数据极限"""
    print_header("修改ARIMA数据极限")
    
    current_config = get_arima_data_limits()
    print(f"当前配置:")
    print(f"  最小数据量: {current_config['最小数据量']}")
    print(f"  最大数据量: {current_config['最大数据量']}")
    print(f"  启用采样: {current_config['数据采样']['启用采样']}")
    
    try:
        # 修改最小数据量
        min_data_input = input(f"\n新的最小数据量 (当前: {current_config['最小数据量']}, 回车跳过): ").strip()
        if min_data_input:
            new_min_data = int(min_data_input)
        else:
            new_min_data = None
        
        # 修改最大数据量
        max_data_input = input(f"新的最大数据量 (当前: {current_config['最大数据量']}, 回车跳过): ").strip()
        if max_data_input:
            new_max_data = int(max_data_input)
        else:
            new_max_data = None
        
        # 修改采样设置
        sampling_input = input(f"是否启用采样 (当前: {current_config['数据采样']['启用采样']}, y/n, 回车跳过): ").strip().lower()
        if sampling_input == 'y':
            new_sampling = True
        elif sampling_input == 'n':
            new_sampling = False
        else:
            new_sampling = None
        
        if update_arima_data_limits(new_min_data, new_max_data, new_sampling):
            print_success("数据极限配置已更新")
        else:
            print_error("更新失败")
        
    except Exception as e:
        print_error(f"修改失败: {e}")


def modify_arima_model_params():
    """修改ARIMA模型参数"""
    print_header("修改ARIMA模型参数")
    print_info("模型参数修改功能正在开发中...")
    print_info("请直接修改config.py文件中的ARIMA_TRAINING_CONFIG配置")


def modify_arima_prediction_config():
    """修改ARIMA预测配置"""
    print_header("修改ARIMA预测配置")
    print_info("预测配置修改功能正在开发中...")
    print_info("请直接修改config.py文件中的ARIMA_TRAINING_CONFIG配置")


def modify_arima_evaluation_config():
    """修改ARIMA评估配置"""
    print_header("修改ARIMA评估配置")
    print_info("评估配置修改功能正在开发中...")
    print_info("请直接修改config.py文件中的ARIMA_TRAINING_CONFIG配置")


def config_management_menu():
    """配置管理菜单"""
    while True:
        print_header("配置管理", "选择操作")
        print("1. 查看当前配置")
        print("2. 查看所有数据源")
        print("3. 切换数据源")
        print("4. 添加新数据源")
        print("5. 验证数据结构")
        print("6. ARIMA模型配置")
        print("0. 返回主菜单")
        
        try:
            choice = input("\n请选择操作 (0-5): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                show_current_config()
            elif choice == "2":
                show_all_data_sources()
            elif choice == "3":
                change_data_source()
            elif choice == "4":
                add_new_data_source()
            elif choice == "5":
                validate_data_structure()
            elif choice == "6":
                arima_config_menu()
            else:
                print_error("无效的选择")
                
        except KeyboardInterrupt:
            print("\n操作已取消")
            break
        except Exception as e:
            print_error(f"操作失败: {e}")
        
        input("\n按回车键继续...")


if __name__ == "__main__":
    config_management_menu() 