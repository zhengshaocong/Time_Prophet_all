# -*- coding: utf-8 -*-
"""
项目设置脚本
用于初始化项目环境和创建必要的文件
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """创建项目目录结构"""
    print("创建项目目录结构...")
    
    # 项目根目录
    root_dir = Path(__file__).parent.parent
    
    # 需要创建的目录
    directories = [
        "data",
        "cache", 
        "temp",
        "output",
        "output/images",
        "output/data",
        "utils",
        "src",
        "script",
        "tests"
    ]
    
    for directory in directories:
        dir_path = root_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ 创建目录: {directory}")
    
    # 创建.gitkeep文件保持空目录
    gitkeep_dirs = ["data", "output/images", "output/data", "src"]
    for dir_name in gitkeep_dirs:
        gitkeep_file = root_dir / dir_name / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            print(f"  ✓ 创建文件: {dir_name}/.gitkeep")

def create_sample_data():
    """创建示例数据文件"""
    print("\n创建示例数据文件...")
    
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    
    # 示例CSV数据
    sample_csv_data = """姓名,年龄,城市,职业
张三,25,北京,工程师
李四,30,上海,设计师
王五,28,广州,产品经理
赵六,35,深圳,数据分析师
钱七,27,杭州,前端开发
"""
    
    csv_file = data_dir / "sample_data.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(sample_csv_data)
    print(f"  ✓ 创建文件: data/sample_data.csv")
    
    # 示例JSON数据
    sample_json_data = {
        "项目信息": {
            "名称": "Python项目模板",
            "版本": "1.0.0",
            "描述": "这是一个标准化的Python项目模板",
            "作者": "开发者",
            "创建时间": "2024-01-01"
        },
        "配置": {
            "数据目录": "data/",
            "输出目录": "output/",
            "缓存目录": "cache/"
        },
        "功能": [
            "文件操作工具",
            "数据处理工具", 
            "缓存管理工具",
            "配置管理"
        ]
    }
    
    import json
    json_file = data_dir / "sample_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(sample_json_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 创建文件: data/sample_data.json")

def create_sample_scripts():
    """创建示例脚本"""
    print("\n创建示例脚本...")
    
    root_dir = Path(__file__).parent.parent
    script_dir = root_dir / "script"
    
    # 数据清理脚本
    cleanup_script = '''# -*- coding: utf-8 -*-
"""
数据清理脚本示例
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_utils import clean_data, validate_data
from utils.file_utils import read_csv, write_csv
from config import DATA_DIR, OUTPUT_DATA_DIR

def main():
    """主函数"""
    print("开始数据清理...")
    
    # 读取原始数据
    input_file = DATA_DIR / "sample_data.csv"
    if not input_file.exists():
        print(f"输入文件不存在: {input_file}")
        return
    
    data = read_csv(input_file)
    print(f"读取到 {len(data)} 条数据")
    
    # 清理数据
    cleaned_data = clean_data(data)
    print(f"清理后剩余 {len(cleaned_data)} 条数据")
    
    # 验证数据
    required_fields = ["姓名", "年龄", "城市"]
    if validate_data(cleaned_data, required_fields):
        print("数据验证通过")
    else:
        print("数据验证失败")
        return
    
    # 保存清理后的数据
    output_file = OUTPUT_DATA_DIR / "cleaned_data.csv"
    write_csv(cleaned_data, output_file)
    print(f"清理后的数据已保存到: {output_file}")

if __name__ == "__main__":
    main()
'''
    
    cleanup_file = script_dir / "cleanup_data.py"
    with open(cleanup_file, 'w', encoding='utf-8') as f:
        f.write(cleanup_script)
    print(f"  ✓ 创建文件: script/cleanup_data.py")

def main():
    """主函数"""
    print("=== Python项目模板设置 ===")
    
    # 创建项目结构
    create_project_structure()
    
    # 创建示例数据
    create_sample_data()
    
    # 创建示例脚本
    create_sample_scripts()
    
    print("\n✓ 项目设置完成！")
    print("\n下一步:")
    print("1. 运行 'python start.py' 启动项目")
    print("2. 运行 'python script/cleanup_data.py' 测试数据清理")
    print("3. 运行 'python -m pytest tests/' 运行测试")

if __name__ == "__main__":
    main() 