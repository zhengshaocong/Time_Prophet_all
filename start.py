# -*- coding: utf-8 -*-
"""
启动脚本
自动检查虚拟环境，安装依赖并启动程序
"""

import os
import sys
import subprocess
from pathlib import Path

def check_virtual_environment():
    """检查是否在虚拟环境中"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def install_requirements():
    """安装依赖包"""
    print("正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖包安装完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖包安装失败: {e}")
        return False

def main():
    """主函数"""
    print("=== Python项目模板启动器 ===")
    
    # 检查虚拟环境
    if not check_virtual_environment():
        print("警告: 建议在虚拟环境中运行程序")
        print("可以使用以下命令创建虚拟环境:")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # Linux/Mac")
        print("  venv\\Scripts\\activate     # Windows")
        print()
    
    # 检查requirements.txt是否存在
    if Path("requirements.txt").exists():
        # 安装依赖
        if not install_requirements():
            print("依赖安装失败，程序退出")
            return
    else:
        print("未找到requirements.txt文件，跳过依赖安装")
    
    print("\n启动主程序...")
    print("-" * 50)
    
    # 启动主程序
    try:
        subprocess.run([sys.executable, "run.py"])
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main() 