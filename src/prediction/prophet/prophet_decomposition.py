# -*- coding: utf-8 -*-
"""
Prophet 分解分析模块
专门处理 y(t) = g(t) + s(t) + h(t) + e 的分解分析

功能：
1. 趋势项 g(t) 分析：拐点检测、趋势稳定性
2. 季节性项 s(t) 分析：季节性强度、周期性分解
3. 节假日项 h(t) 分析：节假日影响、排名
4. 误差项 e 分析：残差分析、异常值检测
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from prophet import Prophet
from prophet.diagnostics import performance_metrics
from scipy import stats
from scipy.stats import normaltest, jarque_bera
import seaborn as sns

from utils.interactive_utils import (
    print_header, print_success, print_error, print_info, print_warning
)
from utils.visualization_utils import setup_matplotlib
from config import OUTPUT_DIR, IMAGES_DIR, PROPHET_TRAINING_CONFIG


class ProphetDecompositionAnalyzer:
    """Prophet分解分析器"""
    
    def __init__(self, model: Prophet, df_ds_y: pd.DataFrame, predictions: pd.DataFrame):
        """
        初始化分解分析器
        
        Args:
            model: 训练好的Prophet模型
            df_ds_y: 训练数据 (ds, y)
            predictions: 预测结果
        """
        self.model = model
        self.df_ds_y = df_ds_y
        self.predictions = predictions
        self.decomposition_config = PROPHET_TRAINING_CONFIG["分解分析配置"]
        self.viz_config = PROPHET_TRAINING_CONFIG["可视化配置"]
        
        # 分解结果存储
        self.decomposition_results = {}
        self.analysis_report = {}
    
    def analyze_trend_component(self) -> Dict:
        """分析趋势项 g(t)"""
        print_info("开始分析趋势项 g(t)...")
        
        trend_config = self.decomposition_config["分析项目"]["趋势分析"]
        if not trend_config["启用"]:
            print_info("趋势分析未启用，跳过")
            return {}
        
        try:
            # 检查预测结果中的列名
            print_info(f"预测结果列名: {list(self.predictions.columns)}")
            
            # 检查是否有趋势列
            if "trend" not in self.predictions.columns:
                print_warning("未找到趋势列，跳过趋势分析")
                return {}
            
            # 获取趋势数据
            trend_data = self.predictions[["ds", "trend"]].copy()
            
            results = {
                "趋势数据": trend_data,
                "趋势统计": {},
                "拐点信息": {},
                "稳定性分析": {}
            }
            
            # 基础统计
            trend_values = trend_data["trend"].values
            results["趋势统计"] = {
                "均值": np.mean(trend_values),
                "标准差": np.std(trend_values),
                "最小值": np.min(trend_values),
                "最大值": np.max(trend_values),
                "趋势斜率": np.polyfit(range(len(trend_values)), trend_values, 1)[0]
            }
            
            # 拐点检测
            if trend_config["拐点检测"]:
                try:
                    changepoints = self.model.changepoints
                    if changepoints is not None and len(changepoints) > 0:
                        results["拐点信息"] = {
                            "拐点数量": len(changepoints),
                            "拐点日期": changepoints.tolist(),
                            "拐点效应": []  # 暂时不获取拐点效应，避免API兼容性问题
                        }
                    else:
                        results["拐点信息"] = {"拐点数量": 0, "拐点日期": [], "拐点效应": []}
                except Exception as e:
                    print_warning(f"拐点检测失败: {e}")
                    results["拐点信息"] = {"拐点数量": 0, "拐点日期": [], "拐点效应": []}
            
            # 趋势稳定性分析
            if trend_config["趋势稳定性"]:
                try:
                    # 计算趋势的变异系数
                    trend_mean = np.mean(trend_values)
                    if abs(trend_mean) > 1e-10:  # 避免除零
                        trend_cv = np.std(trend_values) / abs(trend_mean)
                    else:
                        trend_cv = np.inf
                    
                    # 计算趋势的线性度（R²）
                    x = np.arange(len(trend_values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, trend_values)
                    r_squared = r_value ** 2
                    
                    results["稳定性分析"] = {
                        "变异系数": trend_cv,
                        "线性度_R²": r_squared,
                        "趋势强度": "强" if r_squared > 0.8 else "中等" if r_squared > 0.5 else "弱"
                    }
                except Exception as e:
                    print_warning(f"趋势稳定性分析失败: {e}")
                    results["稳定性分析"] = {
                        "变异系数": np.nan,
                        "线性度_R²": np.nan,
                        "趋势强度": "未知"
                    }
            
            self.decomposition_results["趋势分析"] = results
            print_success("趋势项分析完成")
            return results
            
        except Exception as e:
            print_error(f"趋势项分析失败: {e}")
            return {}
    
    def analyze_seasonality_component(self) -> Dict:
        """分析季节性项 s(t)"""
        print_info("开始分析季节性项 s(t)...")
        
        seasonality_config = self.decomposition_config["分析项目"]["季节性分析"]
        if not seasonality_config["启用"]:
            print_info("季节性分析未启用，跳过")
            return {}
        
        try:
            results = {
                "季节性数据": {},
                "季节性强度": {},
                "季节性分解": {}
            }
            
            # 获取各种季节性数据
            seasonality_columns = [col for col in self.predictions.columns if 'yearly' in col or 'weekly' in col or 'monthly' in col]
            
            for col in seasonality_columns:
                seasonality_name = col.replace('_', ' ').title()
                seasonality_values = self.predictions[col].values
                
                results["季节性数据"][seasonality_name] = {
                    "数据": seasonality_values,
                    "均值": np.mean(seasonality_values),
                    "标准差": np.std(seasonality_values),
                    "振幅": np.max(seasonality_values) - np.min(seasonality_values)
                }
            
            # 季节性强度分析
            if seasonality_config["季节性强度"]:
                try:
                    total_variance = np.var(self.df_ds_y["y"].values)
                    seasonal_variance = 0
                    
                    for col in seasonality_columns:
                        seasonal_variance += np.var(self.predictions[col].values)
                    
                    seasonal_strength = seasonal_variance / total_variance if total_variance > 0 else 0
                    
                    results["季节性强度"] = {
                        "总方差": total_variance,
                        "季节性方差": seasonal_variance,
                        "季节性强度": seasonal_strength,
                        "强度评级": "强" if seasonal_strength > 0.3 else "中等" if seasonal_strength > 0.1 else "弱"
                    }
                except Exception as e:
                    print_warning(f"季节性强度分析失败: {e}")
                    results["季节性强度"] = {
                        "总方差": np.nan,
                        "季节性方差": np.nan,
                        "季节性强度": np.nan,
                        "强度评级": "未知"
                    }
            
            self.decomposition_results["季节性分析"] = results
            print_success("季节性项分析完成")
            return results
            
        except Exception as e:
            print_error(f"季节性项分析失败: {e}")
            return {}
    
    def analyze_holiday_component(self) -> Dict:
        """分析节假日项 h(t)"""
        print_info("开始分析节假日项 h(t)...")
        
        holiday_config = self.decomposition_config["分析项目"]["节假日分析"]
        if not holiday_config["启用"]:
            print_info("节假日分析未启用，跳过")
            return {}
        
        try:
            results = {
                "节假日数据": {},
                "节假日影响": {},
                "节假日排名": {}
            }
            
            # 检查是否有节假日数据
            holiday_columns = [col for col in self.predictions.columns if 'holidays' in col]
            
            if not holiday_columns:
                results["节假日数据"] = {"message": "未检测到节假日数据"}
                self.decomposition_results["节假日分析"] = results
                print_info("未检测到节假日数据")
                return results
            
            # 分析节假日影响
            if holiday_config["节假日影响"]:
                for col in holiday_columns:
                    holiday_values = self.predictions[col].values
                    non_zero_indices = np.where(holiday_values != 0)[0]
                    
                    if len(non_zero_indices) > 0:
                        holiday_effects = holiday_values[non_zero_indices]
                        results["节假日影响"][col] = {
                            "影响天数": len(non_zero_indices),
                            "平均影响": np.mean(holiday_effects),
                            "最大影响": np.max(holiday_effects),
                            "最小影响": np.min(holiday_effects),
                            "影响标准差": np.std(holiday_effects)
                        }
            
            # 节假日排名
            if holiday_config["节假日排名"] and "节假日影响" in results:
                holiday_effects = []
                for col, effect_data in results["节假日影响"].items():
                    holiday_effects.append({
                        "节假日": col,
                        "平均影响": effect_data["平均影响"],
                        "影响天数": effect_data["影响天数"]
                    })
                
                # 按平均影响排序
                holiday_effects.sort(key=lambda x: abs(x["平均影响"]), reverse=True)
                results["节假日排名"] = holiday_effects
            
            self.decomposition_results["节假日分析"] = results
            print_success("节假日项分析完成")
            return results
            
        except Exception as e:
            print_error(f"节假日项分析失败: {e}")
            return {}
    
    def analyze_error_component(self) -> Dict:
        """分析误差项 e"""
        print_info("开始分析误差项 e...")
        
        error_config = self.decomposition_config["分析项目"]["残差分析"]
        if not error_config["启用"]:
            print_info("残差分析未启用，跳过")
            return {}
        
        try:
            # 计算残差
            actual = self.df_ds_y["y"].values
            predicted = self.predictions["yhat"].values[:len(actual)]
            residuals = actual - predicted
            
            results = {
                "残差数据": residuals,
                "残差统计": {},
                "正态性检验": {},
                "异常值检测": {}
            }
            
            # 基础统计
            results["残差统计"] = {
                "均值": np.mean(residuals),
                "标准差": np.std(residuals),
                "偏度": stats.skew(residuals),
                "峰度": stats.kurtosis(residuals),
                "最小值": np.min(residuals),
                "最大值": np.max(residuals)
            }
            
            # 正态性检验
            if error_config["正态性检验"]:
                try:
                    # Shapiro-Wilk检验
                    shapiro_stat, shapiro_p = stats.shapiro(residuals)
                    # Jarque-Bera检验
                    jb_stat, jb_p = jarque_bera(residuals)
                    
                    results["正态性检验"] = {
                        "Shapiro_Wilk": {"统计量": shapiro_stat, "p值": shapiro_p, "正态": shapiro_p > 0.05},
                        "Jarque_Bera": {"统计量": jb_stat, "p值": jb_p, "正态": jb_p > 0.05},
                        "综合判断": "正态" if all([shapiro_p > 0.05, jb_p > 0.05]) else "非正态"
                    }
                except Exception as e:
                    print_warning(f"正态性检验失败: {e}")
                    results["正态性检验"] = {
                        "Shapiro_Wilk": {"统计量": np.nan, "p值": np.nan, "正态": False},
                        "Jarque_Bera": {"统计量": np.nan, "p值": np.nan, "正态": False},
                        "综合判断": "未知"
                    }
            
            # 异常值检测
            if error_config["异常值检测"]:
                try:
                    threshold = PROPHET_TRAINING_CONFIG["模型分解配置"]["误差项"]["噪声处理"]["异常值阈值"]
                    z_scores = np.abs(stats.zscore(residuals))
                    outliers = z_scores > threshold
                    
                    results["异常值检测"] = {
                        "异常值数量": np.sum(outliers),
                        "异常值比例": np.mean(outliers),
                        "异常值索引": np.where(outliers)[0].tolist(),
                        "异常值": residuals[outliers].tolist()
                    }
                except Exception as e:
                    print_warning(f"异常值检测失败: {e}")
                    results["异常值检测"] = {
                        "异常值数量": 0,
                        "异常值比例": 0.0,
                        "异常值索引": [],
                        "异常值": []
                    }
            
            self.decomposition_results["残差分析"] = results
            print_success("误差项分析完成")
            return results
            
        except Exception as e:
            print_error(f"误差项分析失败: {e}")
            return {}
    
    def run_full_decomposition_analysis(self) -> bool:
        """运行完整的分解分析"""
        print_header("Prophet分解分析", "y(t) = g(t) + s(t) + h(t) + e")
        
        if not self.decomposition_config["启用"]:
            print_info("分解分析未启用，跳过")
            return True
        
        # 执行各项分析
        self.analyze_trend_component()
        self.analyze_seasonality_component()
        self.analyze_holiday_component()
        self.analyze_error_component()
        
        print_success("Prophet分解分析完成！")
        return True
