# -*- coding: utf-8 -*-
"""
Prophet 预测主模块（从步骤3开始：数据格式标准化 -> 训练 -> 预测 -> 保存 -> 可视化）
说明：
- 数据来源：严格使用【数据预处理】生成的处理后数据（output/data/*processed*.csv）
- 目录规范：
  - 数据输出: output/data/
  - 图像输出: output/images/prophet/
- 中文注释，UTF-8 编码
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from prophet import Prophet

from utils.interactive_utils import (
    print_header, print_success, print_error, print_info, print_warning
)
from utils.config_utils import get_field_name
from config import OUTPUT_DIR, IMAGES_DIR, PROPHET_TRAINING_CONFIG, PROPHET_DEFAULT_PARAMS
from config.forecast import GLOBAL_FORECAST_CONFIG
from .prophet_decomposition import ProphetDecompositionAnalyzer
from .prophet_fit_evaluator import ProphetFitEvaluator


class ProphetPredictorMain:
    """Prophet 预测器主类（从步骤3开始）"""

    def __init__(self):
        self.module_name = "prophet_predictor"
        self.data: pd.DataFrame | None = None
        self.df_ds_y_purchase: pd.DataFrame | None = None  # 申购数据
        self.df_ds_y_redeem: pd.DataFrame | None = None   # 赎回数据
        self.model_purchase: Prophet | None = None         # 申购模型
        self.model_redeem: Prophet | None = None          # 赎回模型
        self.predictions_purchase: pd.DataFrame | None = None  # 申购预测
        self.predictions_redeem: pd.DataFrame | None = None   # 赎回预测
        self.has_purchase_redeem: bool = False             # 是否有申购赎回数据

    # ----------------------------
    # 第3步：数据格式标准化（仅 ds/y）
    # ----------------------------
    def load_preprocessed_data(self) -> bool:
        """加载预处理后的数据（严格使用 output/data/*processed*.csv）"""
        try:
            processed_files = list((OUTPUT_DIR / "data").glob("*processed*.csv"))
            if not processed_files:
                print_error("未找到预处理后的数据，请先运行【数据预处理】")
                return False
            latest = max(processed_files, key=lambda p: p.stat().st_mtime)
            self.data = pd.read_csv(latest, encoding='utf-8')
            print_success(f"已加载预处理数据: {latest} ({len(self.data):,} 行)")
            return True
        except Exception as e:
            print_error(f"加载预处理数据失败: {e}")
            return False

    def standardize_to_ds_y(self) -> bool:
        """将数据转换为 Prophet 需要的 ds/y 格式（第3步）"""
        if self.data is None:
            print_error("请先加载预处理数据")
            return False
        try:
            # 获取时间字段，如果没有配置文件则尝试自动检测
            time_field = get_field_name("时间字段")
            if not time_field:
                # 尝试自动检测时间字段
                possible_time_fields = ['report_date', 'date', 'time', 'timestamp']
                for field in possible_time_fields:
                    if field in self.data.columns:
                        time_field = field
                        break
                
                if not time_field:
                    print_error("无法自动检测时间字段，请检查数据或运行基础数据分析")
                    return False
                
                print_info(f"自动检测到时间字段: {time_field}")
            
            # 检查是否有申购和赎回字段
            purchase_field = 'total_purchase_amt'
            redeem_field = 'total_redeem_amt'
            
            if purchase_field in self.data.columns and redeem_field in self.data.columns:
                self.has_purchase_redeem = True
                print_info("检测到申购和赎回字段，将分别预测")
                
                # 准备申购数据
                df_purchase = self.data[[time_field, purchase_field]].copy()
                df_purchase[time_field] = pd.to_datetime(df_purchase[time_field])
                df_purchase = df_purchase.groupby(time_field, as_index=False)[purchase_field].sum()
                df_purchase = df_purchase.sort_values(time_field)
                df_purchase = df_purchase.rename(columns={time_field: "ds", purchase_field: "y"})
                
                # 准备赎回数据
                df_redeem = self.data[[time_field, redeem_field]].copy()
                df_redeem[time_field] = pd.to_datetime(df_redeem[time_field])
                df_redeem = df_redeem.groupby(time_field, as_index=False)[redeem_field].sum()
                df_redeem = df_redeem.sort_values(time_field)
                df_redeem = df_redeem.rename(columns={time_field: "ds", redeem_field: "y"})
                
                # 频率补齐
                inferred_freq = pd.infer_freq(df_purchase["ds"]) if len(df_purchase) > 3 else "D"
                full_range = pd.date_range(df_purchase["ds"].min(), df_purchase["ds"].max(), freq=inferred_freq)
                
                # 申购数据补齐
                df_full_purchase = pd.DataFrame({"ds": full_range})
                df_merged_purchase = df_full_purchase.merge(df_purchase, on="ds", how="left")
                df_merged_purchase["y"] = df_merged_purchase["y"].interpolate(method="linear", limit=7)
                
                # 赎回数据补齐
                df_full_redeem = pd.DataFrame({"ds": full_range})
                df_merged_redeem = df_full_redeem.merge(df_redeem, on="ds", how="left")
                df_merged_redeem["y"] = df_merged_redeem["y"].interpolate(method="linear", limit=7)
                
                # 去除极端异常
                for df_merged in [df_merged_purchase, df_merged_redeem]:
                    y_q1, y_q3 = df_merged["y"].quantile([0.25, 0.75])
                    iqr = y_q3 - y_q1
                    lower_bound = y_q1 - 1.5 * iqr
                    upper_bound = y_q3 + 1.5 * iqr
                    df_merged["y"] = df_merged["y"].clip(lower_bound, upper_bound)
                
                self.df_ds_y_purchase = df_merged_purchase
                self.df_ds_y_redeem = df_merged_redeem
                
                print_success(f"申购数据标准化完成: {len(df_merged_purchase):,} 行")
                print_success(f"赎回数据标准化完成: {len(df_merged_redeem):,} 行")
                return True
            else:
                # 回退到原来的Net_Flow预测
                print_warning("未检测到申购赎回字段，回退到Net_Flow预测")
                if "Net_Flow" not in self.data.columns:
                    print_error("预处理数据缺少 Net_Flow 字段")
                    return False

                df = self.data[[time_field, "Net_Flow"]].copy()
                df[time_field] = pd.to_datetime(df[time_field])

                # 按时间聚合（若存在重复时间点），此处使用求和，可按业务改为均值
                df_grouped = df.groupby(time_field, as_index=False)["Net_Flow"].sum()
                df_grouped = df_grouped.sort_values(time_field)

                # 标准化为 ds/y
                df_grouped = df_grouped.rename(columns={time_field: "ds", "Net_Flow": "y"})

                # 频率补齐（按最常见间隔推断频率）
                inferred_freq = pd.infer_freq(df_grouped["ds"]) if len(df_grouped) > 3 else None
                if inferred_freq is None:
                    # 回退为日频（项目常用）
                    inferred_freq = "D"
                    print_warning("无法自动推断频率，默认使用日频 D")
                
                full_range = pd.date_range(df_grouped["ds"].min(), df_grouped["ds"].max(), freq=inferred_freq)
                df_full = pd.DataFrame({"ds": full_range})
                df_merged = df_full.merge(df_grouped, on="ds", how="left")

                # 缺失小段插值，大段保持缺失（避免错误放大）
                df_merged["y"] = df_merged["y"].interpolate(method="linear", limit=7)

                # 去除极端异常（温和裁剪）
                y_q1, y_q3 = df_merged["y"].quantile([0.25, 0.75])
                iqr = y_q3 - y_q1
                lower_bound = y_q1 - 1.5 * iqr
                upper_bound = y_q3 + 1.5 * iqr
                df_merged["y"] = df_merged["y"].clip(lower_bound, upper_bound)

                self.df_ds_y_purchase = df_merged
                self.df_ds_y_redeem = None
                self.has_purchase_redeem = False
                
                print_success(f"Net_Flow 数据标准化完成: {len(df_merged):,} 行")
                return True
                
        except Exception as e:
            print_error(f"数据标准化失败: {e}")
            return False

    # ----------------------------
    # 第5-7步：建模与训练
    # ----------------------------
    def build_model(self, custom_params: dict = None) -> bool:
        """构建 Prophet 模型（第4步）"""
        try:
            # 合并默认参数和自定义参数
            params = PROPHET_DEFAULT_PARAMS.copy()
            if custom_params:
                params.update(custom_params)
            
            if self.has_purchase_redeem:
                # 构建申购模型
                self.model_purchase = Prophet(**params)
                print_info("申购模型构建完成")
                
                # 构建赎回模型
                self.model_redeem = Prophet(**params)
                print_info("赎回模型构建完成")
            else:
                # 回退到原来的单模型
                self.model_purchase = Prophet(**params)
                self.model_redeem = None
                print_info("Net_Flow 模型构建完成")
            
            return True
        except Exception as e:
            print_error(f"模型构建失败: {e}")
            return False

    def train(self) -> bool:
        """训练 Prophet 模型（第5步）"""
        try:
            if self.has_purchase_redeem:
                # 训练申购模型
                print_info("训练申购模型...")
                self.model_purchase.fit(self.df_ds_y_purchase)
                print_success("申购模型训练完成")
                
                # 训练赎回模型
                print_info("训练赎回模型...")
                self.model_redeem.fit(self.df_ds_y_redeem)
                print_success("赎回模型训练完成")
            else:
                # 训练Net_Flow模型
                print_info("训练Net_Flow模型...")
                self.model_purchase.fit(self.df_ds_y_purchase)
                print_success("Net_Flow模型训练完成")
            
            return True
        except Exception as e:
            print_error(f"模型训练失败: {e}")
            return False

    def evaluate_fit(self) -> bool:
        """评估模型拟合度（第6步）"""
        try:
            if self.has_purchase_redeem:
                # 评估申购模型
                print_info("评估申购模型拟合度...")
                evaluator_purchase = ProphetFitEvaluator(self.model_purchase, self.df_ds_y_purchase)
                success_purchase = evaluator_purchase.evaluate_and_report()
                
                # 评估赎回模型
                print_info("评估赎回模型拟合度...")
                evaluator_redeem = ProphetFitEvaluator(self.model_redeem, self.df_ds_y_redeem)
                success_redeem = evaluator_redeem.evaluate_and_report()
                
                return success_purchase and success_redeem
            else:
                # 评估Net_Flow模型
                print_info("评估Net_Flow模型拟合度...")
                evaluator = ProphetFitEvaluator(self.model_purchase, self.df_ds_y_purchase)
                return evaluator.evaluate_and_report()
        except Exception as e:
            print_error(f"拟合度评估失败: {e}")
            return False

    # ----------------------------
    # 第8-9步：构造未来数据与预测输出
    # ----------------------------
    def predict(self, periods: int = None, freq: str = None) -> bool:
        """生成预测（第8步）"""
        try:
            # 确定预测期数
            if periods is None:
                # 从全局配置获取预测结束日期
                try:
                    from config.forecast import GLOBAL_FORECAST_CONFIG
                    cfg = GLOBAL_FORECAST_CONFIG
                    if cfg.get("enabled", False):
                        end_date = pd.to_datetime(cfg.get("end_date_exclusive", "2015-01-01"))
                        if self.has_purchase_redeem:
                            last_date = self.df_ds_y_purchase["ds"].max()
                        else:
                            last_date = self.df_ds_y_purchase["ds"].max()
                        periods = (end_date - last_date).days
                        if periods <= 0:
                            periods = 30
                    else:
                        periods = 30
                except Exception:
                    periods = 30
            
            if freq is None:
                freq = "D"
            
            print_info(f"预测期数: {periods}, 频率: {freq}")
            
            if self.has_purchase_redeem:
                # 申购预测
                print_info("生成申购预测...")
                future_dates = self.model_purchase.make_future_dataframe(periods=periods, freq=freq)
                self.predictions_purchase = self.model_purchase.predict(future_dates)
                print_success("申购预测完成")
                
                # 赎回预测
                print_info("生成赎回预测...")
                future_dates = self.model_redeem.make_future_dataframe(periods=periods, freq=freq)
                self.predictions_redeem = self.model_redeem.predict(future_dates)
                print_success("赎回预测完成")
            else:
                # Net_Flow预测
                print_info("生成Net_Flow预测...")
                future_dates = self.model_purchase.make_future_dataframe(periods=periods, freq=freq)
                self.predictions_purchase = self.model_purchase.predict(future_dates)
                self.predictions_redeem = None
                print_success("Net_Flow预测完成")
            
            return True
        except Exception as e:
            print_error(f"预测失败: {e}")
            return False

    # ----------------------------
    # 第9步：保存结果
    # ----------------------------
    def save_results(self) -> bool:
        """保存预测结果（第9步）"""
        try:
            if self.has_purchase_redeem:
                # 保存申购赎回预测结果
                output_dir = OUTPUT_DIR / "prophet"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 合并申购和赎回预测结果，确保格式一致
                combined_results = pd.DataFrame()
                combined_results["report_date"] = self.predictions_purchase["ds"].dt.strftime("%Y%m%d")
                # 保留2位小数，与用户要求格式一致
                combined_results["purchase"] = self.predictions_purchase["yhat"].round(2)
                combined_results["redeem"] = self.predictions_redeem["yhat"].round(2)
                
                # 保存为CSV，确保格式完全一致
                csv_path = output_dir / "prophet_purchase_redeem_forecast.csv"
                combined_results.to_csv(csv_path, index=False, encoding='utf-8')
                print_success(f"申购赎回预测结果已保存: {csv_path}")
                
                # 保存详细预测结果
                purchase_detail = self.predictions_purchase[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                purchase_detail.columns = ["ds", "purchase", "purchase_lower", "purchase_upper"]
                purchase_detail.to_csv(output_dir / "prophet_purchase_detail.csv", index=False, encoding='utf-8')
                
                redeem_detail = self.predictions_redeem[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                redeem_detail.columns = ["ds", "redeem", "redeem_lower", "redeem_upper"]
                redeem_detail.to_csv(output_dir / "prophet_redeem_detail.csv", index=False, encoding='utf-8')
                
                print_success("详细预测结果已保存")

                # 额外输出一份与 ARIMA 一致的CSV（report_date/purchase/redeem，YYYYMMDD，UTF-8-SIG）
                try:
                    # 仅导出未来区间的预测结果，避免与历史重叠
                    last_hist_date = self.df_ds_y_purchase["ds"].max()
                    future_purchase = self.predictions_purchase[self.predictions_purchase["ds"] > last_hist_date]
                    future_redeem = self.predictions_redeem[self.predictions_redeem["ds"] > last_hist_date]

                    matched_df = pd.DataFrame({
                        'report_date': future_purchase["ds"].dt.strftime('%Y%m%d').values,
                        'purchase': future_purchase["yhat"].values,
                        'redeem': future_redeem["yhat"].values
                    })
                    matched_out_dir = OUTPUT_DIR / "data"
                    matched_out_dir.mkdir(parents=True, exist_ok=True)
                    matched_file = matched_out_dir / "prophet_forecast_201409.csv"
                    matched_df.to_csv(matched_file, index=False, encoding='utf-8-sig')
                    print_success(f"已额外输出与ARIMA格式一致的CSV: {matched_file}")
                except Exception as e:
                    print_warning(f"输出与ARIMA一致格式CSV失败: {e}")
                
            else:
                # 保存Net_Flow预测结果
                output_dir = OUTPUT_DIR / "prophet"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                netflow_results = self.predictions_purchase[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                netflow_results.columns = ["ds", "netflow", "netflow_lower", "netflow_upper"]
                netflow_results.to_csv(output_dir / "prophet_netflow_forecast.csv", index=False, encoding='utf-8')
                
                print_success(f"Net_Flow预测结果已保存: {output_dir}")
            
            return True
        except Exception as e:
            print_error(f"保存结果失败: {e}")
            return False

    # ----------------------------
    # 第10步：可视化
    # ----------------------------
    def plot_predictions(self) -> bool:
        """绘制预测结果图（第10步）"""
        try:
            if self.has_purchase_redeem:
                # 绘制申购和赎回对比图
                import matplotlib.pyplot as plt
                from utils.visualization_utils import setup_matplotlib
                
                setup_matplotlib()
                images_dir = IMAGES_DIR / "prophet"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建对比图
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
                
                # 申购图
                ax1.plot(self.df_ds_y_purchase["ds"], self.df_ds_y_purchase["y"], 
                        label="历史申购", color="blue", alpha=0.7)
                ax1.plot(self.predictions_purchase["ds"], self.predictions_purchase["yhat"], 
                        label="申购预测", color="blue", linestyle="--", linewidth=2)
                ax1.fill_between(self.predictions_purchase["ds"], 
                               self.predictions_purchase["yhat_lower"], 
                               self.predictions_purchase["yhat_upper"], 
                               color="blue", alpha=0.2, label="申购置信区间")
                ax1.set_title("Prophet 申购预测", fontsize=14, fontweight='bold')
                ax1.set_xlabel("时间", fontsize=12)
                ax1.set_ylabel("申购金额", fontsize=12)
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                # 赎回图
                ax2.plot(self.df_ds_y_redeem["ds"], self.df_ds_y_redeem["y"], 
                        label="历史赎回", color="red", alpha=0.7)
                ax2.plot(self.predictions_redeem["ds"], self.predictions_redeem["yhat"], 
                        label="赎回预测", color="red", linestyle="--", linewidth=2)
                ax2.fill_between(self.predictions_redeem["ds"], 
                               self.predictions_redeem["yhat_lower"], 
                               self.predictions_redeem["yhat_upper"], 
                               color="red", alpha=0.2, label="赎回置信区间")
                ax2.set_title("Prophet 赎回预测", fontsize=14, fontweight='bold')
                ax2.set_xlabel("时间", fontsize=12)
                ax2.set_ylabel("赎回金额", fontsize=12)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                fig.savefig(
                    images_dir / "prophet_purchase_redeem_forecast.png",
                    dpi=300, bbox_inches='tight'
                )
                plt.close(fig)
                
                print_success("申购赎回预测图已保存")
                
                # 绘制合并对比图
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # 历史数据
                ax.plot(self.df_ds_y_purchase["ds"], self.df_ds_y_purchase["y"], 
                       label="历史申购", color="blue", alpha=0.7)
                ax.plot(self.df_ds_y_redeem["ds"], self.df_ds_y_redeem["y"], 
                       label="历史赎回", color="red", alpha=0.7)
                
                # 预测数据
                ax.plot(self.predictions_purchase["ds"], self.predictions_purchase["yhat"], 
                       label="申购预测", color="blue", linestyle="--", linewidth=2)
                ax.plot(self.predictions_redeem["ds"], self.predictions_redeem["yhat"], 
                       label="赎回预测", color="red", linestyle="--", linewidth=2)
                
                ax.set_title("Prophet 申购与赎回预测对比", fontsize=16, fontweight='bold')
                ax.set_xlabel("时间", fontsize=12)
                ax.set_ylabel("金额", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                fig.savefig(
                    images_dir / "prophet_purchase_redeem_compare.png",
                    dpi=300, bbox_inches='tight'
                )
                plt.close(fig)
                
                print_success("申购赎回对比图已保存")
                
            else:
                # 原来的Net_Flow图
                import matplotlib.pyplot as plt
                from utils.visualization_utils import setup_matplotlib
                
                setup_matplotlib()
                images_dir = IMAGES_DIR / "prophet"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.plot(self.df_ds_y_purchase["ds"], self.df_ds_y_purchase["y"], 
                       label="历史值", color="blue", alpha=0.7)
                ax.plot(self.predictions_purchase["ds"], self.predictions_purchase["yhat"], 
                       label="预测值", color="red", linestyle="--", linewidth=2)
                ax.fill_between(self.predictions_purchase["ds"], 
                              self.predictions_purchase["yhat_lower"], 
                              self.predictions_purchase["yhat_upper"], 
                              color="red", alpha=0.2, label="置信区间")
                ax.set_title("Prophet Net_Flow 预测", fontsize=14, fontweight='bold')
                ax.set_xlabel("时间", fontsize=12)
                ax.set_ylabel("Net_Flow", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(
                    images_dir / "prophet_netflow_forecast.png",
                    dpi=300, bbox_inches='tight'
                )
                plt.close(fig)
                
                print_success("Net_Flow预测图已保存")
            
            return True
        except Exception as e:
            print_error(f"绘图失败: {e}")
            return False

    def cross_validate(self) -> bool:
        """执行交叉验证（第7步）"""
        try:
            if self.has_purchase_redeem:
                # 申购模型交叉验证
                print_info("执行申购模型交叉验证...")
                success_purchase = self._cross_validate_single_model(
                    self.model_purchase, self.df_ds_y_purchase, "申购"
                )
                
                # 赎回模型交叉验证
                print_info("执行赎回模型交叉验证...")
                success_redeem = self._cross_validate_single_model(
                    self.model_redeem, self.df_ds_y_redeem, "赎回"
                )
                
                return success_purchase and success_redeem
            else:
                # Net_Flow模型交叉验证
                print_info("执行Net_Flow模型交叉验证...")
                return self._cross_validate_single_model(
                    self.model_purchase, self.df_ds_y_purchase, "Net_Flow"
                )
        except Exception as e:
            print_error(f"交叉验证失败: {e}")
            return False
    
    def _cross_validate_single_model(self, model, df_ds_y, model_name: str) -> bool:
        """执行单个模型的交叉验证"""
        try:
            # 获取交叉验证配置
            cv_config = PROPHET_TRAINING_CONFIG.get("交叉验证配置", {})
            if not cv_config.get("启用", True):
                print_info(f"{model_name}模型交叉验证未启用，跳过")
                return True
            
            # 导入正确的交叉验证函数
            from prophet.diagnostics import cross_validation
            
            # 根据数据量自动调整交叉验证参数
            total_days = (df_ds_y["ds"].max() - df_ds_y["ds"].min()).days
            
            if total_days < 90:
                # 数据量较少，使用较小的参数
                initial = "30 days"
                period = "15 days"
                horizon = "7 days"
                print_info(f"数据量较少({total_days}天)，使用较小的交叉验证参数")
            elif total_days < 365:
                # 数据量中等，使用中等参数
                initial = "60 days"
                period = "30 days"
                horizon = "15 days"
                print_info(f"数据量中等({total_days}天)，使用中等交叉验证参数")
            else:
                # 数据量充足，使用默认参数
                initial = cv_config.get("initial", "730 days")
                period = cv_config.get("period", "365 days")
                horizon = cv_config.get("horizon", "90 days")
                print_info(f"数据量充足({total_days}天)，使用默认交叉验证参数")
            
            # 执行交叉验证
            df_cv = cross_validation(
                model, 
                initial=initial,
                period=period,
                horizon=horizon
            )
            
            # 保存交叉验证结果
            output_dir = OUTPUT_DIR / "prophet"
            output_dir.mkdir(parents=True, exist_ok=True)
            cv_path = output_dir / f"prophet_{model_name.lower()}_cv_results.csv"
            df_cv.to_csv(cv_path, index=False, encoding='utf-8')
            print_success(f"{model_name}模型交叉验证结果已保存: {cv_path}")
            
            # 绘制交叉验证图
            viz_config = PROPHET_TRAINING_CONFIG["可视化配置"]
            if viz_config["保存配置"]["保存交叉验证图"]:
                import matplotlib.pyplot as plt
                from utils.visualization_utils import setup_matplotlib
                
                setup_matplotlib()
                images_dir = IMAGES_DIR / "prophet"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                # 绘制交叉验证结果
                fig, ax = plt.subplots(figsize=viz_config["图像质量"]["图像大小"])
                ax.plot(df_cv["ds"], df_cv["y"], label="实际值", color="blue", alpha=0.7)
                ax.plot(df_cv["ds"], df_cv["yhat"], label="预测值", color="red", alpha=0.7)
                ax.fill_between(df_cv["ds"], df_cv["yhat_lower"], df_cv["yhat_upper"], 
                               color="red", alpha=0.2, label="置信区间")
                ax.set_title(f"Prophet {model_name} 交叉验证结果", fontsize=14, fontweight='bold')
                ax.set_xlabel("时间", fontsize=12)
                ax.set_ylabel("数值", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(
                    images_dir / f"prophet_{model_name.lower()}_cv.png", 
                    dpi=viz_config["图像质量"]["dpi"], 
                    bbox_inches='tight'
                )
                plt.close(fig)
            
            print_success(f"{model_name}模型交叉验证完成")
            return True
        except Exception as e:
            print_error(f"{model_name}模型交叉验证失败: {e}")
            return False

    def analyze_decomposition(self) -> bool:
        """执行Prophet分解分析 y(t) = g(t) + s(t) + h(t) + e（第11步）"""
        try:
            if self.has_purchase_redeem:
                # 申购模型分解分析
                print_info("执行申购模型分解分析...")
                analyzer_purchase = ProphetDecompositionAnalyzer(
                    self.model_purchase, self.df_ds_y_purchase, self.predictions_purchase
                )
                success_purchase = analyzer_purchase.run_full_decomposition_analysis()
                
                # 赎回模型分解分析
                print_info("执行赎回模型分解分析...")
                analyzer_redeem = ProphetDecompositionAnalyzer(
                    self.model_redeem, self.df_ds_y_redeem, self.predictions_redeem
                )
                success_redeem = analyzer_redeem.run_full_decomposition_analysis()
                
                if success_purchase and success_redeem:
                    print_success("Prophet分解分析完成")
                    return True
                else:
                    print_error("Prophet分解分析失败")
                    return False
            else:
                # Net_Flow模型分解分析
                print_info("执行Net_Flow模型分解分析...")
                analyzer = ProphetDecompositionAnalyzer(
                    self.model_purchase, self.df_ds_y_purchase, self.predictions_purchase
                )
                success = analyzer.run_full_decomposition_analysis()
                
                if success:
                    print_success("Prophet分解分析完成")
                    return True
                else:
                    print_error("Prophet分解分析失败")
                    return False
                
        except Exception as e:
            print_error(f"分解分析失败: {e}")
            return False

    # ----------------------------
    # 一键流程
    # ----------------------------
    def run_full_prophet(self, periods: int = None, freq: str = None, custom_params: dict = None) -> bool:
        """完整流程：加载预处理数据 -> 标准化 -> 建模 -> 训练 -> 拟合评估 -> 交叉验证 -> 预测 -> 分解分析 -> 保存 -> 可视化"""
        print_header("Prophet 预测流程", "从步骤3开始：数据格式标准化")
        if not self.load_preprocessed_data():
            return False
        if not self.standardize_to_ds_y():
            return False
        if not self.build_model(custom_params=custom_params):
            return False
        if not self.train():
            return False
        # 在预测前进行拟合度评估
        if not self.evaluate_fit():
            return False
        if not self.cross_validate():
            return False
        if not self.predict(periods=periods, freq=freq):
            return False
        if not self.analyze_decomposition():
            return False
        if not self.save_results():
            return False
        if not self.plot_predictions():
            return False
        print_success("Prophet 完整流程执行完成！")
        return True


# 便捷入口函数

def run_prophet_prediction(periods: int = None, freq: str = None, custom_params: dict = None) -> bool:
    """便捷入口函数：运行Prophet预测"""
    predictor = ProphetPredictorMain()
    return predictor.run_full_prophet(periods=periods, freq=freq, custom_params=custom_params)
