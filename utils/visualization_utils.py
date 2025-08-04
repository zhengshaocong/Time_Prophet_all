# -*- coding: utf-8 -*-
"""
可视化工具函数
包含通用的图表生成功能
"""

import matplotlib.pyplot as plt
import numpy as np
from config import IMAGES_DIR
from utils.config_utils import get_field_name

def setup_matplotlib():
    """设置matplotlib中文字体"""
    import matplotlib.font_manager as fm
    import platform
    import os
    
    # 根据操作系统设置不同的字体
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # macOS 系统字体路径
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Arial Unicode MS.ttf'
        ]
        font_list = ['PingFang SC', 'STHeiti', 'Hiragino Sans GB', 'Arial Unicode MS']
    elif system == "Windows":
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc'   # 宋体
        ]
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        ]
        font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    
    # 首先尝试添加字体文件
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
                print(f"✓ 成功加载字体文件: {font_path}")
            except Exception as e:
                continue
    
    # 然后尝试设置字体
    font_found = False
    for font in font_list:
        try:
            # 检查字体是否可用
            font_prop = fm.FontProperties(family=font)
            font_path = fm.findfont(font_prop)
            
            # 如果找到了字体文件
            if font_path and font_path != fm.rcParams['font.sans-serif']:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 成功设置字体: {font}")
                font_found = True
                break
        except Exception as e:
            continue
    
    # 如果还是没有找到合适的中文字体，使用简单的设置
    if not font_found:
        print("⚠ 未找到合适的中文字体，使用基本设置")
        # 使用基本的字体设置
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

def create_time_series_plot(data, time_field, value_fields, labels, colors, title, ax):
    """
    创建时间序列图
    
    Args:
        data: 数据框
        time_field: 时间字段名
        value_fields: 值字段名列表
        labels: 标签列表
        colors: 颜色列表
        title: 图表标题
        ax: matplotlib轴对象
    """
    try:
        for field, label, color in zip(value_fields, labels, colors):
            ax.plot(data[time_field], data[field], 
                   linewidth=1, alpha=0.7, color=color, label=label)
        
        ax.set_title(title)
        ax.set_xlabel('时间')
        ax.set_ylabel('金额')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        return True
    except Exception as e:
        print(f"时间序列图生成失败: {e}")
        return False

def create_histogram(data, field, bins=30, color='skyblue', title='', ax=None):
    """
    创建直方图
    
    Args:
        data: 数据框
        field: 字段名
        bins: 直方图箱数
        color: 颜色
        title: 标题
        ax: matplotlib轴对象
    """
    try:
        if ax is None:
            ax = plt.gca()
        
        ax.hist(data[field], bins=bins, alpha=0.7, color=color, edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel(field)
        ax.set_ylabel('频次')
        return True
    except Exception as e:
        print(f"直方图生成失败: {e}")
        return False

def create_balance_change_plot(data, title='余额变化分布（排除异常值）', ax=None):
    """
    创建余额变化分布图（处理异常值）
    
    Args:
        data: 数据框
        title: 标题
        ax: matplotlib轴对象
    """
    try:
        if ax is None:
            ax = plt.gca()
        
        balance_change = data['Balance_Change']
        
        # 计算分位数来识别异常值
        q1 = balance_change.quantile(0.25)
        q3 = balance_change.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 过滤异常值用于可视化
        normal_balance_change = balance_change[(balance_change >= lower_bound) & (balance_change <= upper_bound)]
        
        # 绘制直方图
        ax.hist(normal_balance_change, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('余额变化')
        ax.set_ylabel('频次')
        
        # 添加统计信息
        stats_text = f"正常范围: {lower_bound:.0f} ~ {upper_bound:.0f}\n"
        stats_text += f"异常值数量: {len(balance_change) - len(normal_balance_change)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return True
    except Exception as e:
        print(f"余额变化分布图生成失败: {e}")
        return False

def create_monthly_comparison_plot(data, title='月度资金流分布', ax=None):
    """
    创建月度对比图
    
    Args:
        data: 数据框
        title: 标题
        ax: matplotlib轴对象
    """
    try:
        if ax is None:
            ax = plt.gca()
        
        # 按月份聚合数据
        monthly_data = data.groupby('Month').agg({
            'Net_Flow': 'mean',
            'Purchase_Amount': 'mean',
            'Redemption_Amount': 'mean'
        }).reset_index()
        
        x = range(len(monthly_data))
        width = 0.25
        
        ax.bar([i - width for i in x], monthly_data['Purchase_Amount'], 
               width, label='申购金额', color='green', alpha=0.7)
        ax.bar([i + width for i in x], monthly_data['Redemption_Amount'], 
               width, label='赎回金额', color='red', alpha=0.7)
        ax.plot(x, monthly_data['Net_Flow'], 'o-', label='净资金流', 
               color='blue', linewidth=2, markersize=6)
        
        ax.set_title(title)
        ax.set_xlabel('月份')
        ax.set_ylabel('平均金额')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{m}月' for m in monthly_data['Month']])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return True
    except Exception as e:
        print(f"月度对比图生成失败: {e}")
        return False

def save_plot(fig, filename, dpi=300):
    """
    保存图表
    
    Args:
        fig: matplotlib图形对象
        filename: 文件名
        dpi: 分辨率
    """
    try:
        plot_path = IMAGES_DIR / filename
        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存: {plot_path}")
        return True
    except Exception as e:
        print(f"图表保存失败: {e}")
        return False

def close_plot(fig):
    """
    关闭图表释放内存
    
    Args:
        fig: matplotlib图形对象
    """
    try:
        plt.close(fig)
    except Exception as e:
        print(f"关闭图表失败: {e}") 