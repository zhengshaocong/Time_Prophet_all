# -*- coding: utf-8 -*-
"""
经典分解法核心算法模块
Classical Decomposition Core Algorithm Module

功能：
1. 计算周期因子
2. 计算Base值
3. 计算趋势
4. 预测未来值
5. 时间序列分解
6. 申购和赎回资金流预测
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from utils.interactive_utils import print_info, print_success, print_warning, print_error


class ClassicalDecompositionCore:
    """经典分解法核心算法类"""
    
    def __init__(self, config):
        """
        初始化核心算法
        
        Args:
            config: 经典分解法配置
        """
        self.config = config
        self.seasonality_mode = 'multiplicative'  # 默认乘法模式
        
    def calculate_periodic_factors(self, data, period_type="weekday", date_column='date', value_column='value'):
        """
        计算周期因子
        
        Args:
            data: 数据
            period_type: 周期类型，支持weekday, monthly, yearly
            date_column: 日期列名
            value_column: 值列名
            
        Returns:
            dict: 周期因子
        """
        print_info(f"计算{period_type}周期因子...")
        
        try:
            if period_type == "weekday":
                return self._calculate_weekday_factors(data, date_column, value_column)
            elif period_type == "monthly":
                return self._calculate_monthly_factors(data, date_column, value_column)
            elif period_type == "yearly":
                return self._calculate_yearly_factors(data, date_column, value_column)
            else:
                print_warning(f"不支持的周期类型: {period_type}")
                return None
                
        except Exception as e:
            print_error(f"计算周期因子失败: {e}")
            return None
    
    def _calculate_weekday_factors(self, data, date_column, value_column):
        """计算星期几周期因子"""
        try:
            # 确保日期列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                data[date_column] = pd.to_datetime(data[date_column])
            
            # 添加星期几列
            data = data.copy()
            data['weekday'] = data[date_column].dt.dayofweek
            
            # 处理缺失值
            data[value_column] = data[value_column].fillna(method='ffill').fillna(method='bfill')
            
            # 按星期几分组计算均值
            weekday_means = data.groupby('weekday')[value_column].mean()
            
            # 计算总体均值
            overall_mean = data[value_column].mean()
            
            if overall_mean == 0 or pd.isna(overall_mean):
                print_warning("总体均值为零或NaN，无法计算周期因子")
                return None
            
            # 计算周期因子
            periodic_factors = {}
            for weekday in range(7):
                if weekday in weekday_means and not pd.isna(weekday_means[weekday]):
                    factor = weekday_means[weekday] / overall_mean
                    periodic_factors[weekday] = factor
                else:
                    periodic_factors[weekday] = 1.0
            
            # 检查周期因子的变化程度，如果变化太小则增强对比度
            factor_values = np.array(list(periodic_factors.values()), dtype=float)
            factor_std = np.nanstd(factor_values)
            factor_mean = np.nanmean(factor_values)
            
            # 如果标准差太小（变化不明显），增强对比度
            if factor_std < 0.01:
                print_warning("周期因子变化不明显，增强对比度")
                min_factor = np.nanmin(factor_values)
                max_factor = np.nanmax(factor_values)
                if max_factor > min_factor:
                    normalized_factors = {}
                    for k, v in periodic_factors.items():
                        normalized_value = 0.8 + 0.4 * (v - min_factor) / (max_factor - min_factor)
                        normalized_factors[k] = normalized_value
                else:
                    normalized_factors = periodic_factors
            else:
                # 正常归一化，保持均值接近1
                if abs(factor_mean) < 1e-10:
                    print_warning("周期因子均值接近零，跳过归一化")
                    normalized_factors = periodic_factors
                else:
                    normalized_factors = {k: v / factor_mean for k, v in periodic_factors.items()}
            
            print_success(f"星期几周期因子计算完成")
            for weekday, factor in normalized_factors.items():
                weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
                print_info(f"  {weekday_names[weekday]}: {factor:.4f}")
            
            return {'weekday': normalized_factors}
            
        except Exception as e:
            print_error(f"计算星期几周期因子失败: {e}")
            return None
    
    def _calculate_monthly_factors(self, data, date_column, value_column):
        """计算月度周期因子"""
        try:
            # 确保日期列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                data[date_column] = pd.to_datetime(data[date_column])
            
            # 添加月份列
            data = data.copy()
            data['month'] = data[date_column].dt.month
            
            # 按月份分组计算均值
            monthly_means = data.groupby('month')[value_column].mean()
            
            # 计算总体均值
            overall_mean = data[value_column].mean()
            
            if overall_mean == 0 or pd.isna(overall_mean):
                print_warning("总体均值为零或NaN，无法计算周期因子")
                return None
            
            # 计算周期因子
            periodic_factors = {}
            for month in range(1, 13):
                if month in monthly_means and not pd.isna(monthly_means[month]):
                    factor = monthly_means[month] / overall_mean
                    periodic_factors[month] = factor
                else:
                    periodic_factors[month] = 1.0
            
            # 归一化处理
            factor_values = np.array(list(periodic_factors.values()), dtype=float)
            factor_mean = np.nanmean(factor_values)
            
            if abs(factor_mean) > 1e-10:
                normalized_factors = {k: v / factor_mean for k, v in periodic_factors.items()}
            else:
                normalized_factors = periodic_factors
            
            print_success("月度周期因子计算完成")
            return {'monthly': normalized_factors}
            
        except Exception as e:
            print_error(f"计算月度周期因子失败: {e}")
            return None
    
    def _calculate_yearly_factors(self, data, date_column, value_column):
        """计算年度周期因子"""
        try:
            # 确保日期列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                data[date_column] = pd.to_datetime(data[date_column])
            
            # 添加一年中的第几天列
            data = data.copy()
            data['day_of_year'] = data[date_column].dt.dayofyear
            
            # 按天分组计算均值
            daily_means = data.groupby('day_of_year')[value_column].mean()
            
            # 计算总体均值
            overall_mean = data[value_column].mean()
            
            if overall_mean == 0 or pd.isna(overall_mean):
                print_warning("总体均值为零或NaN，无法计算周期因子")
                return None
            
            # 计算周期因子
            periodic_factors = {}
            for day in range(1, 367):  # 考虑闰年
                if day in daily_means and not pd.isna(daily_means[day]):
                    factor = daily_means[day] / overall_mean
                    periodic_factors[day] = factor
                else:
                    periodic_factors[day] = 1.0
            
            # 归一化处理
            factor_values = np.array(list(periodic_factors.values()), dtype=float)
            factor_mean = np.nanmean(factor_values)
            
            if abs(factor_mean) > 1e-10:
                normalized_factors = {k: v / factor_mean for k, v in periodic_factors.items()}
            else:
                normalized_factors = periodic_factors
            
            print_success("年度周期因子计算完成")
            return {'yearly': normalized_factors}
            
        except Exception as e:
            print_error(f"计算年度周期因子失败: {e}")
            return None
    
    def calculate_base_values(self, data, remove_periodic_effect=True, smooth_window=3, 
                             date_column='date', value_column='value'):
        """
        计算Base值（去除周期影响）
        
        Args:
            data: 数据
            remove_periodic_effect: 是否去除周期影响
            smooth_window: 平滑窗口大小
            date_column: 日期列名
            value_column: 值列名
            
        Returns:
            pd.Series: Base值序列
        """
        print_info("计算Base值...")
        
        try:
            if not remove_periodic_effect:
                print_info("不去除周期影响，直接返回原始值")
                return data[value_column]
            
            # 计算周期因子
            periodic_factors = self.calculate_periodic_factors(data, 'weekday', date_column, value_column)
            
            if periodic_factors is None:
                print_warning("无法计算周期因子，返回原始值")
                return data[value_column]
            
            # 去除周期影响
            base_values = []
            for i, row in data.iterrows():
                value = row[value_column]
                if pd.isna(value):
                    base_values.append(np.nan)
                    continue
                
                # 获取对应的周期因子
                if 'weekday' in periodic_factors:
                    weekday = row[date_column].weekday()
                    factor = periodic_factors['weekday'].get(weekday, 1.0)
                else:
                    factor = 1.0
                
                # 去除周期影响
                if factor != 0:
                    base_value = value / factor
                else:
                    base_value = value
                
                base_values.append(base_value)
            
            base_series = pd.Series(base_values, index=data.index)
            
            # 平滑处理
            if smooth_window > 1:
                base_series = base_series.rolling(window=smooth_window, center=True, min_periods=1).mean()
            
            print_success("Base值计算完成")
            return base_series
            
        except Exception as e:
            print_error(f"计算Base值失败: {e}")
            return None
    
    def calculate_trend(self, base_values, window=7):
        """
        计算趋势
        
        Args:
            base_values: Base值序列
            window: 平滑窗口大小
            
        Returns:
            pd.Series: 趋势序列
        """
        print_info("计算趋势...")
        
        try:
            # 使用指数加权移动平均计算趋势
            trend = base_values.ewm(span=window, adjust=False).mean()
            
            print_success("趋势计算完成")
            return trend
            
        except Exception as e:
            print_error(f"计算趋势失败: {e}")
            return None
    
    def predict_future(self, base_values, periodic_factors, forecast_steps=30, 
                      confidence_level=0.95, last_history_date: pd.Timestamp | None = None,
                      history_dates: pd.Series | None = None,
                      hist_original_values: pd.Series | None = None):
        """
        预测未来数据
        
        Args:
            base_values: 基础值序列
            periodic_factors: 周期因子
            forecast_steps: 预测步数
            confidence_level: 置信水平
            last_history_date: 历史最后日期
            history_dates: 历史日期序列
            hist_original_values: 历史原始值序列
            
        Returns:
            dict: 预测结果
        """
        print_info("开始预测未来数据...")
        
        try:
            # 检查是否有申购和赎回数据
            has_purchase_redeem = False
            purchase_base = None
            redeem_base = None
            
            if hist_original_values is not None and hasattr(hist_original_values, 'columns'):
                # 检查是否有申购和赎回列
                if 'total_purchase_amt' in hist_original_values.columns and 'total_redeem_amt' in hist_original_values.columns:
                    has_purchase_redeem = True
                    print_info("检测到申购和赎回数据，将分别预测")
                    
                    # 计算申购和赎回的base值
                    purchase_base = self._calculate_purchase_redeem_base(
                        hist_original_values, 'total_purchase_amt', periodic_factors
                    )
                    redeem_base = self._calculate_purchase_redeem_base(
                        hist_original_values, 'total_redeem_amt', periodic_factors
                    )
            
            # 获取配置参数
            pred_cfg = self.config.get('预测配置', {}) if isinstance(self.config, dict) else {}
            win_cfg = int(pred_cfg.get('趋势窗口', 7))
            if win_cfg <= 0:
                win_cfg = 7
            
            # 随机性控制
            enable_stochastic = bool(pred_cfg.get('启用随机性', False))
            rng_seed = pred_cfg.get('随机种子', None)
            if enable_stochastic and rng_seed is not None:
                try:
                    np.random.seed(int(rng_seed))
                except Exception:
                    pass
            
            # 计算趋势序列
            trend_series = self.calculate_trend(base_values, win_cfg)
            if trend_series is None:
                print_error("无法计算趋势序列")
                return None
            
            # 用最近窗口拟合线性趋势并外推
            recent_window = min(len(trend_series), max(14, int(win_cfg) * 2))
            xs = np.arange(recent_window)
            ys = trend_series.tail(recent_window).values
            
            # 线性回归拟合
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
                print_info(f"趋势拟合: 斜率={slope:.4f}, 截距={intercept:.4f}, R²={r_value**2:.4f}")
            except Exception as e:
                print_warning(f"线性回归失败，使用简单趋势: {e}")
                slope = (ys[-1] - ys[0]) / (len(ys) - 1) if len(ys) > 1 else 0
                intercept = ys[0]
            
            # 对去趋势后的残差做 AR(1) 以产生上下波动
            residuals_tr = base_values - trend_series
            try:
                r = residuals_tr.dropna().values
                if len(r) >= 3:
                    r1 = r[1:]; r0 = r[:-1]
                    denom = (r0**2).sum()
                    phi = float((r0*r1).sum()/denom) if denom > 1e-12 else 0.0
                    eps = r1 - phi*r0
                else:
                    phi = 0.0; eps = np.array([0.0])
            except Exception:
                phi = 0.0; eps = np.array([0.0])

            def draw_eps():
                if not enable_stochastic:
                    return 0.0
                if eps.size <= 1:
                    return 0.0
                idx = np.random.randint(0, eps.size)
                return float(eps[idx])

            r_prev = float(residuals_tr.iloc[-1]) if len(residuals_tr) > 0 else 0.0

            # 趋势阻尼 + 均值回归，让未来不单边下滑
            damping = float(pred_cfg.get('趋势阻尼系数', 0.85))
            mean_reversion_alpha = float(pred_cfg.get('均值回归强度', 0.10))
            mr_window = int(pred_cfg.get('均值回归窗口', 30))
            if mr_window <= 0:
                mr_window = 30
            
            # 确保数据类型正确
            try:
                target_level = float(base_values.tail(min(len(base_values), mr_window)).mean())
                if np.isnan(target_level):
                    target_level = float(base_values.mean())
                    if np.isnan(target_level):
                        target_level = 0.0
            except Exception as e:
                print_warning(f"计算目标水平时出错: {e}")
                target_level = 0.0

            try:
                base_prev = float(base_values.iloc[-1]) if len(base_values) > 0 else 0.0
                if np.isnan(base_prev):
                    base_prev = 0.0
            except Exception as e:
                print_warning(f"获取基础值前一个值时出错: {e}")
                base_prev = 0.0
                
            slope_step = float(slope)
            if np.isnan(slope_step):
                slope_step = 0.0
                
            future_base = []
            for i in range(forecast_steps):
                # 逐步衰减的趋势分量
                slope_step *= damping
                # AR(1) 残差生成上下波动
                r_prev = phi * r_prev + draw_eps()
                # 均值回归将base拉回到近期平均水平
                base_pred = base_prev + slope_step + mean_reversion_alpha * (target_level - base_prev) + r_prev
                base_prev = float(base_pred)
                if np.isnan(base_prev):
                    base_prev = target_level
                future_base.append(base_prev)
            
            # 应用周期因子
            future_predictions = []
            future_dates = []
            
            # 生成未来日期
            if last_history_date is not None:
                start_date = last_history_date + pd.Timedelta(days=1)
            else:
                if hasattr(base_values, 'index') and len(base_values.index) > 0:
                    if isinstance(base_values.index, pd.DatetimeIndex):
                        start_date = base_values.index[-1] + pd.Timedelta(days=1)
                    else:
                        try:
                            start_date = pd.to_datetime(base_values.index[-1]) + pd.Timedelta(days=1)
                        except:
                            start_date = pd.Timestamp.now().normalize()
                else:
                    start_date = pd.Timestamp.now().normalize()
            
            # 从全局配置获取预测结束日期，并据此确定步数
            local_steps = int(forecast_steps)
            try:
                from config.forecast import GLOBAL_FORECAST_CONFIG
                cfg = GLOBAL_FORECAST_CONFIG if isinstance(GLOBAL_FORECAST_CONFIG, dict) else {}
                enabled = bool(cfg.get('enabled', False))
                date_fmt = cfg.get('date_format', '%Y-%m-%d')
                freq = str(cfg.get('freq', 'D')).upper()
                if enabled:
                    end_str = cfg.get('end_date_exclusive', None)
                    if end_str:
                        try:
                            end_date_exclusive = pd.to_datetime(end_str, format=date_fmt, errors='coerce')
                        except Exception:
                            end_date_exclusive = pd.to_datetime(end_str)
                    else:
                        end_date_exclusive = None
                else:
                    end_date_exclusive = None
            except Exception:
                end_date_exclusive = None
                freq = 'D'
                enabled = False
            
            # 若配置启用，覆盖步数以严格到达 end_date_exclusive（不包含该日）
            if enabled and end_date_exclusive is not None:
                # 根据频率仅支持日频
                if isinstance(start_date, pd.Timestamp):
                    delta_days = max(0, (end_date_exclusive - start_date).days)
                else:
                    delta_days = int(forecast_steps)
                local_steps = delta_days
            
            future_date = start_date
            for i in range(local_steps):
                # 检查是否超过结束日期（若配置启用）
                if enabled and (end_date_exclusive is not None) and (future_date >= end_date_exclusive):
                    break
                    
                future_dates.append(future_date)
                
                # 获取对应的周期因子
                if isinstance(periodic_factors, dict):
                    if 'weekday' in periodic_factors:
                        weekday = future_date.weekday()
                        factor = periodic_factors['weekday'].get(weekday, 1.0)
                    elif 'monthly' in periodic_factors:
                        month = future_date.month
                        factor = periodic_factors['monthly'].get(month, 1.0)
                    elif 'yearly' in periodic_factors:
                        day_of_year = future_date.dayofyear
                        factor = periodic_factors['yearly'].get(day_of_year, 1.0)
                    else:
                        factor = 1.0
                else:
                    factor = 1.0
                
                # 添加因子扰动（避免过于规律）
                factor_perturbation = float(pred_cfg.get('因子扰动比例', 0.02))
                if not enable_stochastic:
                    factor_perturbation = 0.0
                if factor_perturbation > 0:
                    factor *= (1 + np.random.uniform(-factor_perturbation, factor_perturbation))
                
                # 计算预测值
                if i < len(future_base):
                    predicted_value = future_base[i] * factor
                else:
                    predicted_value = future_base[-1] * factor if future_base else 0.0
                
                # 确保预测值是有效的数值
                try:
                    predicted_value = float(predicted_value)
                    if np.isnan(predicted_value) or np.isinf(predicted_value):
                        predicted_value = 0.0
                except (ValueError, TypeError):
                    predicted_value = 0.0
                
                # 添加残差噪声
                noise_scale = float(pred_cfg.get('残差噪声比例', 0.1))
                if not enable_stochastic:
                    noise_scale = 0.0
                if noise_scale > 0 and hist_original_values is not None:
                    try:
                        if hasattr(hist_original_values, 'std'):
                            hist_std = hist_original_values.std()
                            if pd.notna(hist_std) and hist_std > 0:
                                noise = np.random.normal(0, hist_std * noise_scale)
                                predicted_value += noise
                    except:
                        pass
                
                future_predictions.append(predicted_value)
                future_date += pd.Timedelta(days=1)
            
            # 起点对齐（主预测）：将预测首点与历史最后一个原始值对齐，避免与历史脱节
            try:
                if hist_original_values is not None and len(future_predictions) > 0:
                    last_hist_val = float(pd.to_numeric(hist_original_values, errors='coerce').dropna().iloc[-1])
                    if np.isfinite(last_hist_val):
                        shift = last_hist_val - float(future_predictions[0])
                        decay = float(pred_cfg.get('起点对齐衰减', 0.15))  # 0~1，越大回归越快
                        aligned_preds = []
                        for i, v in enumerate(future_predictions):
                            factor = (1 - decay) ** i
                            aligned_preds.append(float(v) + shift * factor)
                        future_predictions = aligned_preds
            except Exception:
                pass
            
            # 应用预测值边界
            min_bound, max_bound = self._calculate_prediction_bounds(hist_original_values, pred_cfg)
            future_predictions = np.clip(future_predictions, min_bound, max_bound).tolist()
            
            # 计算置信区间
            confidence_interval = self._calculate_confidence_intervals(
                future_predictions, base_values, trend_series, min_bound, max_bound
            )
            
            # 如果有申购和赎回数据，分别预测
            if has_purchase_redeem and purchase_base is not None and redeem_base is not None:
                purchase_predictions = self._predict_purchase_redeem(
                    purchase_base, periodic_factors, len(future_predictions), 
                    pred_cfg, hist_original_values
                )
                redeem_predictions = self._predict_purchase_redeem(
                    redeem_base, periodic_factors, len(future_predictions), 
                    pred_cfg, hist_original_values
                )
                
                result = {
                    'dates': future_dates,
                    'predictions': future_predictions,
                    'confidence_intervals': confidence_interval,
                    'base_values': future_base[:len(future_predictions)],
                    'purchase_predictions': purchase_predictions,
                    'redeem_predictions': redeem_predictions,
                    'has_purchase_redeem': True
                }
            else:
                result = {
                    'dates': future_dates,
                    'predictions': future_predictions,
                    'confidence_intervals': confidence_interval,
                    'base_values': future_base[:len(future_predictions)],
                    'has_purchase_redeem': False
                }
            
            print_success(f"预测完成，共{len(future_predictions)}步")
            return result
            
        except Exception as e:
            print_warning(f"预测失败: {e}")
            return None
    
    def _calculate_prediction_bounds(self, hist_data, pred_cfg):
        """计算预测值边界（支持关闭与宽松策略）"""
        # 优先读取新边界配置
        bounds_cfg = self.config.get('边界约束', {}) if isinstance(self.config, dict) else {}
        enable_bounds = bool(bounds_cfg.get('启用', True))
        if not enable_bounds:
            return -np.inf, np.inf

        # 旧配置兼容
        min_bound = pred_cfg.get('最小值下界', None)
        max_bound = pred_cfg.get('最大值上界', None)

        # 历史序列准备
        history_values = hist_data if hist_data is not None else pd.Series(dtype=float)
        try:
            if hasattr(history_values, 'iloc'):
                if getattr(history_values, 'ndim', 1) > 1:
                    history_values = history_values.iloc[:, 0]
                history_values = pd.to_numeric(history_values, errors='coerce')
            history_values = history_values.dropna()
        except Exception:
            history_values = pd.Series(dtype=float)

        if history_values.empty:
            return (-np.inf if min_bound is None else float(min_bound),
                    np.inf if max_bound is None else float(max_bound))

        # 读取宽松策略参数
        win_days = int(bounds_cfg.get('窗口天数', 90))
        hi_q = float(bounds_cfg.get('上界', {}).get('分位数', 0.999))
        hi_k = float(bounds_cfg.get('上界', {}).get('标准差倍数', 6.0))
        hi_max_mul = float(bounds_cfg.get('上界', {}).get('历史最大放宽', 1.20))
        lo_q = float(bounds_cfg.get('下界', {}).get('分位数', 0.001))
        lo_k = float(bounds_cfg.get('下界', {}).get('标准差倍数', 6.0))
        lo_min_div = float(bounds_cfg.get('下界', {}).get('历史最小放宽', 1.20))

        recent = history_values.tail(min(len(history_values), win_days))
        mean = recent.mean(); std = recent.std();
        p_hi = recent.quantile(hi_q); p_lo = recent.quantile(lo_q)
        std_hi = mean + hi_k * std; std_lo = mean - lo_k * std
        hist_max = history_values.max() * hi_max_mul
        hist_min = history_values.min() / lo_min_div if history_values.min() != 0 else 0

        # 取更宽松的上界与更宽松的下界
        calc_max = max(p_hi, std_hi, hist_max)
        calc_min = min(p_lo, std_lo, hist_min)

        # 若旧配置显式给定边界，则与新计算综合取更宽松
        if max_bound is not None:
            try:
                calc_max = max(calc_max, float(max_bound))
            except Exception:
                pass
        if min_bound is not None:
            try:
                calc_min = min(calc_min, float(min_bound))
            except Exception:
                pass

        # 容错
        if not np.isfinite(calc_min):
            calc_min = -np.inf
        if not np.isfinite(calc_max):
            calc_max = np.inf
        if calc_min > calc_max:
            calc_min, calc_max = calc_max - 1e-6, calc_max

        return calc_min, calc_max
    
    def _calculate_confidence_intervals(self, predictions, base_values, trend_series, min_bound, max_bound):
        """计算置信区间"""
        try:
            # 确保数据类型正确
            residuals = base_values - trend_series
            residuals = pd.to_numeric(residuals, errors='coerce').dropna()
            
            if len(residuals) == 0:
                # 如果没有有效的残差数据，返回简单的置信区间
                confidence_interval = []
                for pred in predictions:
                    confidence_interval.append((float(pred) * 0.9, float(pred) * 1.1))
                return confidence_interval
            
            residual_std = float(residuals.std())
            if np.isnan(residual_std) or residual_std == 0:
                residual_std = 1.0  # 默认值
            
            confidence_interval = []
            z_score = 1.96
            for i in range(len(predictions)):
                try:
                    pred_value = float(predictions[i])
                    lower_bound = pred_value - z_score * residual_std
                    upper_bound = pred_value + z_score * residual_std
                    
                    # 应用边界
                    if min_bound is not None:
                        lower_bound = max(float(min_bound), lower_bound)
                    if max_bound is not None:
                        upper_bound = min(float(max_bound), upper_bound)
                    
                    confidence_interval.append((lower_bound, upper_bound))
                except (ValueError, TypeError) as e:
                    print_warning(f"计算第{i}个置信区间时出错: {e}")
                    # 使用默认值
                    pred_value = float(predictions[i]) if i < len(predictions) else 0.0
                    confidence_interval.append((pred_value * 0.9, pred_value * 1.1))
            
            return confidence_interval
            
        except Exception as e:
            print_warning(f"计算置信区间时出错: {e}")
            # 返回简单的默认置信区间
            confidence_interval = []
            for pred in predictions:
                try:
                    pred_value = float(pred)
                    confidence_interval.append((pred_value * 0.9, pred_value * 1.1))
                except:
                    confidence_interval.append((0.0, 1.0))
            return confidence_interval
    
    def _calculate_purchase_redeem_base(self, hist_data, column_name, periodic_factors):
        """计算申购或赎回的base值"""
        try:
            if column_name not in hist_data.columns:
                return None
            
            # 确保数据类型正确
            values = hist_data[column_name].copy()
            values = pd.to_numeric(values, errors='coerce')
            values = values.dropna()
            
            if len(values) == 0:
                return None
            
            # 计算周期因子
            if 'weekday' in periodic_factors:
                weekday_factors = periodic_factors['weekday']
                # 去除周期影响
                base_values = []
                
                for i, value in enumerate(values.index):
                    try:
                        if i < len(hist_data):
                            row_data = hist_data.iloc[i]
                            if hasattr(row_data, 'name') and hasattr(row_data.name, 'weekday'):
                                weekday = row_data.name.weekday()
                            else:
                                weekday = i % 7
                            
                            factor = weekday_factors.get(weekday, 1.0)
                            if factor != 0 and not np.isnan(factor) and not np.isinf(factor):
                                base_value = values.iloc[i] / factor
                            else:
                                base_value = values.iloc[i]
                            
                            # 确保base值是有效数值
                            if np.isnan(base_value) or np.isinf(base_value):
                                base_value = values.iloc[i]
                                
                            base_values.append(base_value)
                        else:
                            base_values.append(values.iloc[i])
                    except Exception as e:
                        print_warning(f"计算第{i}个base值时出错: {e}")
                        base_values.append(values.iloc[i])
                
                if base_values:
                    return pd.Series(base_values, index=values.index[:len(base_values)])
                else:
                    return values
            
            return values
            
        except Exception as e:
            print_warning(f"计算{column_name} base值失败: {e}")
            return None
    
    def _predict_purchase_redeem(self, base_values, periodic_factors, forecast_steps, pred_cfg, hist_data):
        """预测申购或赎回值"""
        try:
            if base_values is None or len(base_values) == 0:
                return [0.0] * forecast_steps
            
            # 确保数据类型正确
            base_values = pd.to_numeric(base_values, errors='coerce').dropna()
            if len(base_values) == 0:
                return [0.0] * forecast_steps
            
            # 使用简单的趋势外推
            if len(base_values) > 1:
                try:
                    trend = (base_values.iloc[-1] - base_values.iloc[0]) / (len(base_values) - 1)
                    if np.isnan(trend) or np.isinf(trend):
                        trend = 0.0
                except:
                    trend = 0.0
                    
                try:
                    last_value = float(base_values.iloc[-1])
                    if np.isnan(last_value) or np.isinf(last_value):
                        last_value = float(base_values.mean())
                        if np.isnan(last_value):
                            last_value = 0.0
                except:
                    last_value = 0.0
            else:
                trend = 0.0
                try:
                    last_value = float(base_values.iloc[0]) if len(base_values) > 0 else 0.0
                    if np.isnan(last_value) or np.isinf(last_value):
                        last_value = 0.0
                except:
                    last_value = 0.0
            
            predictions = []
            for i in range(forecast_steps):
                # 简单线性趋势
                pred_value = last_value + trend * (i + 1)
                
                # 添加随机扰动
                noise_scale = float(pred_cfg.get('残差噪声比例', 0.1))
                if noise_scale > 0:
                    try:
                        hist_std = base_values.std() if hasattr(base_values, 'std') else 0.0
                        if pd.notna(hist_std) and hist_std > 0:
                            noise = np.random.normal(0, hist_std * noise_scale)
                            pred_value += noise
                    except:
                        pass
                
                # 确保非负且为有效数值
                try:
                    pred_value = float(pred_value)
                    if np.isnan(pred_value) or np.isinf(pred_value):
                        pred_value = 0.0
                    pred_value = max(0.0, pred_value)
                except (ValueError, TypeError):
                    pred_value = 0.0
                    
                predictions.append(pred_value)
            
            # 起点对齐：将预测首点与历史最后一个原始值对齐，整体平移
            try:
                last_original = None
                if hist_data is not None:
                    last_original = float(pd.to_numeric(hist_data, errors='coerce').dropna().iloc[-1])
                if last_original is not None and len(predictions) > 0:
                    shift = last_original - predictions[0]
                    # 采用平移+轻微衰减（防止长期偏移），逐步回归到模型趋势
                    decay = float(pred_cfg.get('起点对齐衰减', 0.15))  # 0~1，越大越快回归
                    for i in range(len(predictions)):
                        factor = (1 - decay) ** i
                        predictions[i] = max(0.0, predictions[i] + shift * factor)
            except Exception:
                pass
            
            return predictions
            
        except Exception as e:
            print_warning(f"预测申购/赎回失败: {e}")
            return [0.0] * forecast_steps
    
    def decompose_series(self, data, period_type="weekday", date_column='date', value_column='value'):
        """
        时间序列分解
        
        Args:
            data: 数据
            period_type: 周期类型
            date_column: 日期列名
            value_column: 值列名
            
        Returns:
            dict: 分解结果
        """
        print_info("进行时间序列分解...")
        
        try:
            # 自动检测季节性模式
            values = data[value_column].dropna()
            if len(values) == 0:
                print_warning("数据为空，无法进行分解")
                return None
            
            # 检查是否包含负值或跨越零
            has_negative = (values < 0).any()
            crosses_zero = (values > 0).any() and (values < 0).any()
            
            if has_negative or crosses_zero:
                self.seasonality_mode = 'additive'
                print_info("检测到负值或跨越零，使用加法季节性模式")
            else:
                self.seasonality_mode = 'multiplicative'
                print_info("检测到正值，使用乘法季节性模式")
            
            # 计算周期因子
            periodic_factors = self.calculate_periodic_factors(data, period_type, date_column, value_column)
            
            if periodic_factors is None:
                print_warning("无法计算周期因子，返回原始数据")
                return {
                    'original': data[value_column],
                    'trend': data[value_column],
                    'seasonal': pd.Series([1.0] * len(data), index=data.index),
                    'residual': pd.Series([0.0] * len(data), index=data.index)
                }
            
            # 计算Base值
            base_values = self.calculate_base_values(data, True, 3, date_column, value_column)
            
            if base_values is None:
                print_warning("无法计算Base值，返回原始数据")
                return {
                    'original': data[value_column],
                    'trend': data[value_column],
                    'seasonal': pd.Series([1.0] * len(data), index=data.index),
                    'residual': pd.Series([0.0] * len(data), index=data.index)
                }
            
            # 计算趋势
            trend = self.calculate_trend(base_values, 7)
            
            if trend is None:
                print_warning("无法计算趋势，使用Base值作为趋势")
                trend = base_values
            
            # 计算季节性成分
            if self.seasonality_mode == 'additive':
                # 加法模式：使用移动平均去除趋势，得到季节性
                detrended = data[value_column] - trend
                period_length = 7 if period_type == 'weekday' else 30
                seasonal = detrended.rolling(window=period_length, center=True, min_periods=1).mean()
                seasonal = seasonal - seasonal.mean()
            else:
                # 乘法模式：季节性 = 原始值 / 趋势
                seasonal = data[value_column] / trend
                seasonal = seasonal / seasonal.mean()
            
            # 计算残差
            if self.seasonality_mode == 'additive':
                residual = data[value_column] - trend - seasonal
            else:
                residual = data[value_column] - (trend * seasonal)
            
            result = {
                'original': data[value_column],
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'base': base_values,
                'mode': self.seasonality_mode
            }
            
            print_success("时间序列分解完成")
            return result
            
        except Exception as e:
            print_error(f"时间序列分解失败: {e}")
            return None
