# -*- coding: utf-8 -*-
"""
Prophet 拟合度评估模块
在进行预测前，对模型在训练集上的拟合情况进行度量与评级。

功能：
1) 计算拟合指标：MAE, MSE, RMSE, MAPE, R², RMSE/STD(y)
2) 输出中文评级：优秀/良好/一般
3) 保存评估数据与报告到 output/

使用：
from src.prediction.prophet.prophet_fit_evaluator import ProphetFitEvaluator
report = ProphetFitEvaluator(model, df_ds_y).evaluate_and_report()
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from utils.interactive_utils import (
    print_header, print_success, print_error, print_info, print_warning
)
from config import OUTPUT_DIR, IMAGES_DIR, PROPHET_TRAINING_CONFIG


@dataclass
class FitThresholds:
    """拟合度阈值配置对象。"""
    # MAPE 阈值
    mape_good: float = 0.20
    mape_excellent: float = 0.10
    # RMSE 与 y 的标准差之比
    rmse_std_good: float = 0.80
    rmse_std_excellent: float = 0.50

    @staticmethod
    def from_config(cfg: Dict[str, Any] | None) -> "FitThresholds":
        if not cfg:
            return FitThresholds()
        return FitThresholds(
            mape_good=cfg.get("MAPE良好阈值", 0.20),
            mape_excellent=cfg.get("MAPE优秀阈值", 0.10),
            rmse_std_good=cfg.get("RMSE_STD良好阈值", 0.80),
            rmse_std_excellent=cfg.get("RMSE_STD优秀阈值", 0.50),
        )


class ProphetFitEvaluator:
    """Prophet 模型拟合度评估器。"""

    def __init__(self, model, df_ds_y: pd.DataFrame):
        """
        Args:
            model: 已训练的 Prophet 模型
            df_ds_y: 训练数据，包含列 ds, y
        """
        self.model = model
        self.df_ds_y = df_ds_y
        self.cfg = PROPHET_TRAINING_CONFIG.get("拟合评估配置", {})
        self.thresholds = FitThresholds.from_config(self.cfg.get("阈值", None))

        self.images_dir = (IMAGES_DIR / "prophet")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = (OUTPUT_DIR / "data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 核心评估
    # ----------------------------
    def evaluate(self) -> Dict[str, Any]:
        """计算拟合度指标并返回结果。"""
        try:
            # 仅在历史范围内预测，获得拟合值
            future = self.model.make_future_dataframe(periods=0)
            forecast = self.model.predict(future)

            # 对齐训练数据
            df_hist = self.df_ds_y[["ds", "y"]].copy()
            df_pred = forecast[["ds", "yhat"]].copy()
            df_merged = df_hist.merge(df_pred, on="ds", how="inner").dropna()

            y_true = df_merged["y"].values.astype(float)
            y_pred = df_merged["yhat"].values.astype(float)
            residual = y_true - y_pred

            mae = float(np.mean(np.abs(residual)))
            mse = float(np.mean(residual ** 2))
            rmse = float(np.sqrt(mse))

            # 避免除以 0
            eps = 1e-12
            mape = float(np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < eps, eps, np.abs(y_true)))))

            # R² = 1 - SSE/SST
            sse = float(np.sum((y_true - y_pred) ** 2))
            y_mean = float(np.mean(y_true))
            sst = float(np.sum((y_true - y_mean) ** 2))
            r2 = float(1.0 - (sse / sst)) if sst > 0 else float('nan')

            std_y = float(np.std(y_true))
            rmse_std_ratio = float(rmse / std_y) if std_y > eps else float('inf')

            metrics = {
                "样本量": int(len(df_merged)),
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2,
                "RMSE_STD_RATIO": rmse_std_ratio,
            }

            rating = self._rate(metrics)
            return {"metrics": metrics, "rating": rating, "data": df_merged.assign(residual=residual)}
        except Exception as e:
            print_error(f"拟合度评估失败: {e}")
            return {"metrics": {}, "rating": "未知", "data": pd.DataFrame()}

    def _rate(self, metrics: Dict[str, float]) -> str:
        """根据阈值给出中文评级。"""
        mape = metrics.get("MAPE", np.inf)
        rsr = metrics.get("RMSE_STD_RATIO", np.inf)

        # 优先判断“优秀”，再判断“良好”，否则“一般”
        if (mape <= self.thresholds.mape_excellent) and (rsr <= self.thresholds.rmse_std_excellent):
            return "优秀"
        if (mape <= self.thresholds.mape_good) and (rsr <= self.thresholds.rmse_std_good):
            return "良好"
        return "一般"

    # ----------------------------
    # 报告 & 保存
    # ----------------------------
    def evaluate_and_report(self, save: bool = True) -> Dict[str, Any]:
        """计算并打印/保存报告。"""
        print_header("Prophet 模型拟合度评估", "基于历史拟合")
        result = self.evaluate()
        metrics, rating, df_detail = result["metrics"], result["rating"], result["data"]

        if not metrics:
            print_warning("未生成有效的拟合度指标")
            return result

        # 打印摘要
        print_info("拟合度指标：")
        print_info(f"  样本量: {metrics['样本量']}")
        print_info(f"  MAE: {metrics['MAE']:.4f}")
        print_info(f"  MSE: {metrics['MSE']:.4f}")
        print_info(f"  RMSE: {metrics['RMSE']:.4f}")
        print_info(f"  MAPE: {metrics['MAPE']:.4f}")
        print_info(f"  R²: {metrics['R2']:.4f}")
        print_info(f"  RMSE/STD(y): {metrics['RMSE_STD_RATIO']:.4f}")
        print_success(f"模型拟合评级：{rating}")

        if not save:
            return result

        try:
            # 保存明细（覆盖模式）
            if self.cfg.get("保存配置", {}).get("保存拟合明细", True) and not df_detail.empty:
                df_detail.to_csv(self.output_dir / "prophet_fit.csv", index=False, encoding="utf-8")

            # 保存报告
            if self.cfg.get("保存配置", {}).get("保存拟合报告", True):
                lines = [
                    "=" * 60,
                    "Prophet 模型拟合度评估报告",
                    "=" * 60,
                    f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    f"样本量: {metrics['样本量']}",
                    f"MAE: {metrics['MAE']:.6f}",
                    f"MSE: {metrics['MSE']:.6f}",
                    f"RMSE: {metrics['RMSE']:.6f}",
                    f"MAPE: {metrics['MAPE']:.6f}",
                    f"R²: {metrics['R2']:.6f}",
                    f"RMSE/STD(y): {metrics['RMSE_STD_RATIO']:.6f}",
                    f"评级: {rating}",
                    "",
                ]
                report_path = self.output_dir / "prophet_fit_report.txt"
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                print_success(f"拟合评估报告已保存: {report_path}")
        except Exception as e:
            print_warning(f"保存拟合评估结果失败: {e}")

        return result
