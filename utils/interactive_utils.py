# -*- coding: utf-8 -*-
"""
交互式操作界面工具模块
提供终端交互式操作的各种功能
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

# 跨平台键盘输入支持
try:
    import tty
    import termios
    PLATFORM_SUPPORT = True
except ImportError:
    PLATFORM_SUPPORT = False

class InteractiveUI:
    """交互式用户界面类"""
    
    def __init__(self):
        """初始化交互式界面"""
        self.clear_screen()
    
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str, subtitle: str = ""):
        """打印标题"""
        print("=" * 60)
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print("=" * 60)
        print()
    
    def print_success(self, message: str):
        """打印成功消息"""
        print(f"✓ {message}")
    
    def print_error(self, message: str):
        """打印错误消息"""
        print(f"✗ {message}")
    
    def print_warning(self, message: str):
        """打印警告消息"""
        print(f"⚠ {message}")
    
    def print_info(self, message: str):
        """打印信息消息"""
        print(f"ℹ {message}")
    
    def print_progress(self, current: int, total: int, description: str = ""):
        """打印进度条"""
        percentage = int((current / total) * 100)
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        if description:
            print(f"\r{description}: |{bar}| {percentage}% ({current}/{total})", end='')
        else:
            print(f"\r|{bar}| {percentage}% ({current}/{total})", end='')
        
        if current == total:
            print()
    
    def confirm(self, message: str, default: bool = True) -> bool:
        """确认操作（传统方式）"""
        default_text = "Y/n" if default else "y/N"
        while True:
            response = input(f"{message} ({default_text}): ").strip().lower()
            if response == "":
                return default
            elif response in ['y', 'yes', '是']:
                return True
            elif response in ['n', 'no', '否']:
                return False
            else:
                print("请输入 y/yes/是 或 n/no/否")
    
    def confirm_interactive(self, message: str, default: bool = True) -> bool:
        """交互式确认操作（支持方向键）"""
        if not PLATFORM_SUPPORT:
            # 如果不支持tty，使用传统确认方式
            return self.confirm(message, default)
        
        current_selection = 0 if default else 1
        options = ["确定", "取消"]
        
        while True:
            self.clear_screen()
            self.print_header("确认操作")
            print(f"  {message}")
            print()
            
            # 显示选项
            for i, option in enumerate(options):
                if i == current_selection:
                    print(f"  ▶ 【{option}】")
                else:
                    print(f"    【{option}】")
            
            print()
            print("使用 ↑↓ 键选择，回车确认")
            
            # 获取用户输入
            key = self.get_key()
            
            if key == 'UP':
                current_selection = (current_selection - 1) % len(options)
            elif key == 'DOWN':
                current_selection = (current_selection + 1) % len(options)
            elif key == 'ENTER':
                return current_selection == 0  # 0=确定, 1=取消
            elif key == 'QUIT' or key == 'CTRL_C':
                return False  # 取消
    
    def select_option(self, options: List[str], title: str = "请选择选项") -> int:
        """选择选项"""
        print(f"\n{title}:")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        
        while True:
            try:
                choice = input(f"\n请输入选项 (1-{len(options)}): ").strip()
                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    return choice_num - 1
                else:
                    print(f"请输入 1-{len(options)} 之间的数字")
            except ValueError:
                print("请输入有效的数字")
    
    def input_text(self, prompt: str, default: str = "", required: bool = True) -> str:
        """输入文本"""
        while True:
            if default:
                text = input(f"{prompt} (默认: {default}): ").strip()
                if text == "":
                    text = default
            else:
                text = input(f"{prompt}: ").strip()
            
            if not required or text:
                return text
            else:
                print("此字段不能为空，请重新输入")
    
    def input_number(self, prompt: str, min_val: float = None, max_val: float = None) -> float:
        """输入数字"""
        while True:
            try:
                text = input(f"{prompt}: ").strip()
                number = float(text)
                
                if min_val is not None and number < min_val:
                    print(f"数值不能小于 {min_val}")
                    continue
                
                if max_val is not None and number > max_val:
                    print(f"数值不能大于 {max_val}")
                    continue
                
                return number
            except ValueError:
                print("请输入有效的数字")
    
    def input_path(self, prompt: str, must_exist: bool = False, is_dir: bool = False) -> Path:
        """输入路径"""
        while True:
            path_text = input(f"{prompt}: ").strip()
            path = Path(path_text)
            
            if must_exist and not path.exists():
                print("路径不存在，请重新输入")
                continue
            
            if is_dir and path.exists() and not path.is_dir():
                print("请输入目录路径")
                continue
            
            return path
    
    def get_key(self):
        """获取键盘输入"""
        if not PLATFORM_SUPPORT:
            # 如果不支持tty，回退到传统输入方式
            return input("请选择: ").strip()
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # ESC键
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':  # 上箭头
                        return 'UP'
                    elif ch3 == 'B':  # 下箭头
                        return 'DOWN'
                    elif ch3 == 'C':  # 右箭头
                        return 'RIGHT'
                    elif ch3 == 'D':  # 左箭头
                        return 'LEFT'
            elif ch == '\r' or ch == '\n':  # 回车键
                return 'ENTER'
            elif ch == '\x03':  # Ctrl+C
                return 'CTRL_C'
            elif ch == 'q' or ch == 'Q':  # Q键退出
                return 'QUIT'
            else:
                return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def show_interactive_menu(self, menu_items: List[Dict[str, Any]], title: str = "主菜单") -> Optional[Callable]:
        """显示交互式菜单（支持方向键）"""
        if not PLATFORM_SUPPORT:
            # 如果不支持tty，使用传统菜单
            return self.show_traditional_menu(menu_items, title)
        
        current_selection = 0
        total_items = len(menu_items)
        
        while True:
            self.clear_screen()
            self.print_header(title)
            
            # 显示菜单项
            for i, item in enumerate(menu_items):
                if i == current_selection:
                    # 当前选中项
                    print(f"  ▶ 【{item['name']}】")
                    if 'description' in item:
                        print(f"     {item['description']}")
                else:
                    # 未选中项
                    print(f"    【{item['name']}】")
                    if 'description' in item:
                        print(f"     {item['description']}")
                print()
            
            # 显示退出选项
            if current_selection == total_items:
                print("  ▶ 【退出】")
            else:
                print("    【退出】")
            
            print()
            print("使用 ↑↓ 键选择，回车确认，Q键退出")
            
            # 获取用户输入
            key = self.get_key()
            
            if key == 'UP':
                current_selection = (current_selection - 1) % (total_items + 1)
            elif key == 'DOWN':
                current_selection = (current_selection + 1) % (total_items + 1)
            elif key == 'ENTER':
                if current_selection == total_items:
                    return None  # 退出
                else:
                    return menu_items[current_selection]['action']
            elif key == 'QUIT' or key == 'CTRL_C':
                return None  # 退出
    
    def show_traditional_menu(self, menu_items: List[Dict[str, Any]], title: str = "主菜单") -> Optional[Callable]:
        """显示传统菜单（数字选择）"""
        while True:
            self.clear_screen()
            self.print_header(title)
            
            for i, item in enumerate(menu_items, 1):
                print(f"  {i}. 【{item['name']}】")
                if 'description' in item:
                    print(f"     {item['description']}")
                print()
            
            print(f"  0. 【退出】")
            print()
            
            try:
                choice = input("请选择操作: ").strip()
                if choice == "0":
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(menu_items):
                    return menu_items[choice_num - 1]['action']
                else:
                    print(f"请输入 0-{len(menu_items)} 之间的数字")
                    time.sleep(1)
            except ValueError:
                print("请输入有效的数字")
                time.sleep(1)
    
    def show_menu(self, menu_items: List[Dict[str, Any]], title: str = "主菜单") -> Optional[Callable]:
        """显示菜单（兼容旧版本）"""
        return self.show_interactive_menu(menu_items, title)
    
    def show_table(self, data: List[Dict[str, Any]], title: str = ""):
        """显示表格"""
        if not data:
            print("没有数据可显示")
            return
        
        if title:
            print(f"\n{title}")
        
        # 获取所有列名
        columns = list(data[0].keys())
        
        # 计算每列的最大宽度
        col_widths = {}
        for col in columns:
            max_width = len(col)
            for row in data:
                cell_width = len(str(row.get(col, '')))
                max_width = max(max_width, cell_width)
            col_widths[col] = max_width
        
        # 打印表头
        header = "|"
        separator = "|"
        for col in columns:
            header += f" {col:<{col_widths[col]}} |"
            separator += "-" * (col_widths[col] + 2) + "|"
        
        print(separator)
        print(header)
        print(separator)
        
        # 打印数据行
        for row in data:
            row_str = "|"
            for col in columns:
                value = str(row.get(col, ''))
                row_str += f" {value:<{col_widths[col]}} |"
            print(row_str)
        
        print(separator)
    
    def wait_for_key(self, message: str = "按回车键继续..."):
        """等待按键"""
        input(message)
    
    def countdown(self, seconds: int, message: str = "倒计时"):
        """倒计时"""
        for i in range(seconds, 0, -1):
            print(f"\r{message}: {i}秒", end='')
            time.sleep(1)
        print(f"\r{message}: 完成!")
    
    def loading_animation(self, duration: float, message: str = "处理中"):
        """加载动画"""
        import threading
        import time
        
        def animate():
            chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            i = 0
            start_time = time.time()
            
            while time.time() - start_time < duration:
                print(f"\r{chars[i]} {message}", end='')
                i = (i + 1) % len(chars)
                time.sleep(0.1)
            
            print(f"\r✓ {message} 完成!")
        
        animate()

# 全局UI实例
ui = InteractiveUI()

# 便捷函数
def print_header(title: str, subtitle: str = ""):
    """打印标题"""
    ui.print_header(title, subtitle)

def print_success(message: str):
    """打印成功消息"""
    ui.print_success(message)

def print_error(message: str):
    """打印错误消息"""
    ui.print_error(message)

def print_warning(message: str):
    """打印警告消息"""
    ui.print_warning(message)

def print_info(message: str):
    """打印信息消息"""
    ui.print_info(message)

def confirm(message: str, default: bool = True) -> bool:
    """确认操作（传统方式）"""
    return ui.confirm(message, default)

def confirm_interactive(message: str, default: bool = True) -> bool:
    """交互式确认操作（支持方向键）"""
    return ui.confirm_interactive(message, default)

def select_option(options: List[str], title: str = "请选择选项") -> int:
    """选择选项"""
    return ui.select_option(options, title)

def input_text(prompt: str, default: str = "", required: bool = True) -> str:
    """输入文本"""
    return ui.input_text(prompt, default, required)

def input_number(prompt: str, min_val: float = None, max_val: float = None) -> float:
    """输入数字"""
    return ui.input_number(prompt, min_val, max_val)

def input_path(prompt: str, must_exist: bool = False, is_dir: bool = False) -> Path:
    """输入路径"""
    return ui.input_path(prompt, must_exist, is_dir)

def show_menu(menu_items: List[Dict[str, Any]], title: str = "主菜单") -> Optional[Callable]:
    """显示菜单"""
    return ui.show_menu(menu_items, title)

def show_table(data: List[Dict[str, Any]], title: str = ""):
    """显示表格"""
    ui.show_table(data, title)

def wait_for_key(message: str = "按回车键继续..."):
    """等待按键"""
    ui.wait_for_key(message)

def countdown(seconds: int, message: str = "倒计时"):
    """倒计时"""
    ui.countdown(seconds, message)

def loading_animation(duration: float, message: str = "处理中"):
    """加载动画"""
    ui.loading_animation(duration, message) 