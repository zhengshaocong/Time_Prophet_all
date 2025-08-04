# -*- coding: utf-8 -*-
"""
特征工程配置文件
定义特征工程和数据转换相关的配置参数
"""

# ==================== 特征工程配置 ====================
FEATURE_ENGINEERING_CONFIG = {
    "基础特征": {
        "启用": True,              # 是否启用基础特征计算
        "净资金流": True,          # 计算净资金流
        "总资金流": True,          # 计算总资金流
        "资金流比率": True,        # 计算资金流比率
        "余额变化": True,          # 计算余额变化
        "余额变化率": True         # 计算余额变化率
    },
    "时间特征": {
        "启用": True,              # 是否启用时间特征
        "年": True,                # 提取年份
        "月": True,                # 提取月份
        "日": True,                # 提取日期
        "星期": True,              # 提取星期
        "季度": True,              # 提取季度
        "年中天数": True,          # 提取年中天数
        "年中周数": True,          # 提取年中周数
        "月初月末": True,          # 是否为月初月末
        "周末": True,              # 是否为周末
        "节假日": True             # 是否为节假日
    },
    "统计特征": {
        "启用": True,              # 是否启用统计特征
        "滚动窗口": [7, 30],       # 滚动窗口大小（天）
        "统计函数": ["mean", "std", "max", "min"],  # 统计函数
        "特征列表": ["Net_Flow", "Total_Flow", "Balance_Change"]  # 计算统计特征的特征列表
    },
    "用户特征": {
        "启用": True,              # 是否启用用户特征
        "用户统计": True,          # 用户级别统计
        "交易频率": True,          # 交易频率特征
        "用户分类": True           # 用户分类特征
    },
    "业务特征": {
        "启用": True,              # 是否启用业务特征
        "交易活跃度": True,        # 交易活跃度分类
        "资金流类型": True,        # 资金流类型分类
        "余额水平": True           # 余额水平分类
    }
}

# ==================== 数据转换配置 ====================
DATA_TRANSFORMATION_CONFIG = {
    "标准化": {
        "启用": True,              # 是否启用标准化
        "方法": "zscore",          # 标准化方法: zscore, minmax, robust
        "特征列表": [              # 需要标准化的特征列表
            "Net_Flow", "Total_Flow", "Balance_Change", "Balance_Change_Rate",
            "User_NetFlow_Mean", "User_NetFlow_Std", "User_TotalFlow_Mean",
            "User_TotalFlow_Std", "User_BalanceChange_Mean", "User_BalanceChange_Std"
        ]
    },
    "编码": {
        "启用": True,              # 是否启用编码
        "方法": "onehot",          # 编码方法: onehot, label, target
        "特征列表": [              # 需要编码的特征列表
            "Transaction_Activity", "Flow_Type", "Balance_Level"
        ]
    },
    "时间序列特征": {
        "启用": True,              # 是否启用时间序列特征
        "滞后特征": True,          # 滞后特征
        "滞后期数": [1, 7, 30],    # 滞后期数
        "特征列表": [              # 需要计算滞后特征的特征列表
            "Net_Flow", "Total_Flow", "Balance_Change"
        ]
    },
    "特征选择": {
        "启用": True,              # 是否启用特征选择
        "缺失值阈值": 0.5,         # 缺失值比例阈值
        "相关性阈值": 0.95,        # 相关性阈值
        "方差阈值": 0.01           # 方差阈值
    }
} 