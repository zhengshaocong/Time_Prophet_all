# -*- coding: utf-8 -*-
"""
ARIMA预测模块
支持多变量预测：净资金流、申购金额、赎回金额
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

from utils.data_processor import DataProcessor
from utils.interactive_utils import print_header, print_success, print_error, print_info, print_warning
from config import DATA_DIR, OUTPUT_DIR, IMAGES_DIR, ARIMA_TRAINING_CONFIG
from utils.config_utils import get_field_name

# 导入ARIMA相关模块
from .arima_model_trainer import ARIMAModelTrainer
from .arima_predictor_module import ARIMAPredictor
from .arima_visualization import ARIMAVisualizer
from .arima_visualization_enhanced import ARIMAVisualizationEnhanced


class ARIMAPredictorMain(DataProcessor):
    """ARIMA预测器主类"""

    def __init__(self):
        """初始化ARIMA预测器"""
        super().__init__()
        self.module_name = "arima_predictor"
        self.time_series = None
        self.purchase_series = None
        self.redemption_series = None
        self.original_net_flow_series = None  # 保存原始净资金流序列
        self.original_purchase_series = None  # 保存原始申购序列
        self.original_redemption_series = None  # 保存原始赎回序列
        self.model = None
        self.purchase_model = None
        self.redemption_model = None
        # 自相关分析建议参数（用于收缩搜索范围）
        self.suggested_p = None
        self.suggested_q = None
        self.predictions = None
        self.purchase_predictions = None
        self.redemption_predictions = None
        self.visualizer = ARIMAVisualizer()
        self.enhanced_visualizer = ARIMAVisualizationEnhanced()

    
    def load_data(self, file_path=None, use_data_processing=False, module_name=None):
        """重写load_data方法，优先加载预处理后的数据"""
        try:
            # 优先尝试加载预处理后的数据
            processed_data_files = list(OUTPUT_DIR.glob("data/*processed*.csv"))

            if processed_data_files:
                # 按修改时间排序，选择最新的预处理数据
                latest_file = max(processed_data_files, key=lambda x: x.stat().st_mtime)
                print_info(f"找到预处理数据文件: {latest_file}")

                # 加载预处理后的数据
                self.data = pd.read_csv(latest_file, encoding='utf-8')
                print_success(f"预处理数据加载成功: {len(self.data):,} 条记录")
                print_info(f"数据包含 {len(self.data.columns)} 个特征")

                # 显示前几个特征列名
                feature_columns = [col for col in self.data.columns if col not in ['user_id', 'report_date']]
                print_info(f"特征列示例: {feature_columns[:5]}")

                return True
            else:
                # 如果没有预处理数据，尝试加载原始数据
                data_file = DATA_DIR / "user_balance_table.csv"
                if not data_file.exists():
                    print_error(f"未找到预处理数据或原始数据文件")
                    print_info("请先运行数据预处理功能")
                    return False

                    
                print_info(f"加载原始数据文件: {data_file}")
                self.data = pd.read_csv(data_file, encoding='utf-8')
                print_success(f"原始数据加载成功: {len(self.data):,} 条记录")
                return True

        except Exception as e:
            print_error(f"数据加载失败: {e}")
            return False

    
    def _get_time_field_from_specialized_config(self):
        """从特化配置文件读取时间字段映射"""
        try:
            # 查找特化配置文件
            config_files = list(DATA_DIR.glob("*/config.json"))

            if not config_files:
                return None

            # 使用第一个找到的配置文件
            config_file = config_files[0]
            print_info(f"读取特化配置文件: {config_file}")

            # 读取配置文件
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 获取时间字段映射
            if 'field_mapping' in config and '时间字段' in config['field_mapping']:
                time_field = config['field_mapping']['时间字段']
                print_success(f"从特化配置获取时间字段: {time_field}")
                return time_field

            return None

        except Exception as e:
            print_warning(f"读取特化配置文件失败: {e}")
            return None

    
    def prepare_data_for_arima(self, data_source=None):
        """为ARIMA模型准备数据（支持多变量预测）"""
        if self.data is None:
            print_error("请先加载数据")
            return False

        print_header("ARIMA数据准备", "多变量时间序列处理")

        try:
            # 尝试获取字段名
            time_field = get_field_name("时间字段", data_source)
            purchase_field = get_field_name("申购金额字段", data_source)
            redemption_field = get_field_name("赎回金额字段", data_source)

            if not time_field:
                # 尝试从特化配置文件读取字段映射
                time_field = self._get_time_field_from_specialized_config()

                if not time_field:
                    print_error("缺少时间字段映射，请先运行基础数据分析")
                    return False

            # 确保数据已预处理
            if 'Net_Flow' not in self.data.columns:
                print("数据未预处理，正在自动预处理...")
                if not self.preprocess_data():
                    print_error("数据预处理失败")
                    return False

            # 转换时间字段
            self.data[time_field] = pd.to_datetime(self.data[time_field])

            # 按时间排序
            self.data = self.data.sort_values(time_field)

            # 检查必要的字段是否存在
            required_fields = ['Net_Flow', 'total_purchase_amt', 'total_redeem_amt']
            missing_fields = [field for field in required_fields if field not in self.data.columns]

            if missing_fields:
                print_error(f"缺少必要字段: {missing_fields}")
                print_info("请确保数据预处理包含以下字段:")
                print_info("  - Net_Flow: 净资金流")
                print_info("  - total_purchase_amt: 申购金额")
                print_info("  - total_redeem_amt: 赎回金额")
                return False

            # 创建时间序列数据
            print("=" * 50)
            print("🚀 开始准备多变量时间序列数据...")
            print("=" * 50)
            print_info("准备时间序列数据...")
            print(f"  使用时间字段: {time_field}")
            print(f"  使用字段: Net_Flow, total_purchase_amt, total_redeem_amt")

            # 设置时间索引
            data_temp = self.data.set_index(time_field)

            # 创建时间序列
            net_flow_series = data_temp['Net_Flow'].dropna()
            purchase_series = data_temp['total_purchase_amt'].dropna()
            redemption_series = data_temp['total_redeem_amt'].dropna()

            # 输出数据信息
            print(f"  数据量: {len(self.data):,} 条")
            print(f"  时间序列长度: {len(net_flow_series):,} 条")
            print(f"  时间范围: {net_flow_series.index.min()} ~ {net_flow_series.index.max()}")
            print(f"  字段: Net_Flow, total_purchase_amt, total_redeem_amt")
            print_info("多变量时间序列数据准备完成")

            # 检查时间序列的平稳性
            print("检查时间序列平稳性...")

            # 检查净资金流平稳性
            adf_result_net = adfuller(net_flow_series)
            print(f"净资金流ADF统计量: {adf_result_net[0]:.4f}, p值: {adf_result_net[1]:.4f}")

            # 进行自相关分析
            print("进行自相关分析...")
            autocorr_result = self.visualizer.comprehensive_autocorrelation_analysis(net_flow_series)
            if autocorr_result:
                print(f"  建议的ARIMA参数: p={autocorr_result['suggested_p']}, q={autocorr_result['suggested_q']}")
                print(f"  序列特征: {autocorr_result['analysis_summary']['suggested_model']}")
                # 记录建议参数，后续用于收缩网格搜索范围
                try:
                    self.suggested_p = int(autocorr_result.get('suggested_p'))
                except Exception:
                    self.suggested_p = None
                try:
                    self.suggested_q = int(autocorr_result.get('suggested_q'))
                except Exception:
                    self.suggested_q = None

            # 检查申购金额平稳性
            adf_result_purchase = adfuller(purchase_series)
            print(f"申购金额ADF统计量: {adf_result_purchase[0]:.4f}, p值: {adf_result_purchase[1]:.4f}")

            # 检查赎回金额平稳性
            adf_result_redemption = adfuller(redemption_series)
            print(f"赎回金额ADF统计量: {adf_result_redemption[0]:.4f}, p值: {adf_result_redemption[1]:.4f}")

            # 保存原始序列
            self.original_net_flow_series = net_flow_series
            self.original_purchase_series = purchase_series
            self.original_redemption_series = redemption_series

            # 处理平稳性
            if adf_result_net[1] <= 0.05:
                print_success("净资金流时间序列平稳")
                self.time_series = net_flow_series
            else:
                print_info("净资金流时间序列非平稳，使用一阶差分")
                self.time_series = net_flow_series.diff().dropna()

            if adf_result_purchase[1] <= 0.05:
                print_success("申购金额时间序列平稳")
                self.purchase_series = purchase_series
            else:
                print_info("申购金额时间序列非平稳，使用一阶差分")
                self.purchase_series = purchase_series.diff().dropna()

            if adf_result_redemption[1] <= 0.05:
                print_success("赎回金额时间序列平稳")
                self.redemption_series = redemption_series
            else:
                print_info("赎回金额时间序列非平稳，使用一阶差分")
                self.redemption_series = redemption_series.diff().dropna()

            print(f"净资金流时间序列长度: {len(self.time_series)}")
            print(f"申购金额时间序列长度: {len(self.purchase_series)}")
            print(f"赎回金额时间序列长度: {len(self.redemption_series)}")

            return True

        except Exception as e:
            print_error(f"ARIMA数据准备失败: {e}")
            return False

    
    def train_arima_model(self):
        """训练多变量ARIMA模型"""
        if self.time_series is None or self.purchase_series is None or self.redemption_series is None:
            print_error("请先准备数据")
            return False

        print_header("多变量ARIMA模型训练", "参数选择与模型训练")

        try:
            # 训练净资金流模型
            print("训练净资金流ARIMA模型...")
            # 尝试不同的参数组合，选择AIC最小的
            best_aic = float('inf')
            best_model = None
            best_params = None

            # 从配置文件获取参数范围
            model_config = ARIMA_TRAINING_CONFIG["模型参数"]["ARIMA参数"]
            p_values = model_config["p_range"]
            d_values = model_config["d_range"]
            q_values = model_config["q_range"]

            print(f"使用配置的参数范围: p={p_values}, d={d_values}, q={q_values}")

            # 若有自相关分析的建议值，则以建议值为中心收缩p/q范围
            try:
                if self.suggested_p is not None:
                    candidate_p = sorted(set([max(0, self.suggested_p - 1), self.suggested_p, self.suggested_p + 1]))
                    narrowed_p = [v for v in candidate_p if v in p_values]
                    if narrowed_p:
                        p_values = narrowed_p
                if self.suggested_q is not None:
                    candidate_q = sorted(set([max(0, self.suggested_q - 1), self.suggested_q, self.suggested_q + 1]))
                    narrowed_q = [v for v in candidate_q if v in q_values]
                    if narrowed_q:
                        q_values = narrowed_q
                if (self.suggested_p is not None) or (self.suggested_q is not None):
                    print(f"结合自相关分析后的参数范围: p={p_values}, q={q_values}")
            except Exception as _:
                # 遇到异常则保持原范围，避免影响已有流程
                pass

            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(self.time_series, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                                best_params = (p, d, q)
                        except:
                            continue

            if best_model is None:
                # 如果网格搜索失败，使用默认参数
                print("网格搜索失败，使用默认参数")
                self.model = ARIMA(self.time_series, order=(1, 1, 1)).fit()
            else:
                self.model = best_model
                print(f"净资金流最佳参数: {best_params}, AIC: {best_aic:.4f}")

            # 训练申购金额模型
            print("训练申购金额ARIMA模型...")
            best_aic_purchase = float('inf')
            best_model_purchase = None
            best_params_purchase = None

            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(self.purchase_series, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic_purchase:
                                best_aic_purchase = fitted_model.aic
                                best_model_purchase = fitted_model
                                best_params_purchase = (p, d, q)
                        except:
                            continue

            if best_model_purchase is None:
                self.purchase_model = ARIMA(self.purchase_series, order=(1, 1, 1)).fit()
            else:
                self.purchase_model = best_model_purchase
                print(f"申购金额最佳参数: {best_params_purchase}, AIC: {best_aic_purchase:.4f}")

            # 训练赎回金额模型
            print("训练赎回金额ARIMA模型...")
            best_aic_redemption = float('inf')
            best_model_redemption = None
            best_params_redemption = None

            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(self.redemption_series, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic_redemption:
                                best_aic_redemption = fitted_model.aic
                                best_model_redemption = fitted_model
                                best_params_redemption = (p, d, q)
                        except:
                            continue

            if best_model_redemption is None:
                self.redemption_model = ARIMA(self.redemption_series, order=(1, 1, 1)).fit()
            else:
                self.redemption_model = best_model_redemption
                print(f"赎回金额最佳参数: {best_params_redemption}, AIC: {best_aic_redemption:.4f}")

            print_success("多变量ARIMA模型训练完成")

            # 进行残差自相关检验
            print("进行残差自相关检验...")
            if self.model:
                residuals = self.model.resid
                residual_test = self.visualizer.residual_autocorrelation_test(residuals)
                if residual_test:
                    print(f"  残差质量: {'良好' if residual_test['is_white_noise'] else '需要改进'}")

            return True

        except Exception as e:
            print_error(f"ARIMA模型训练失败: {e}")
            return False

    
    def make_predictions(self, steps=None):
        """进行多变量预测"""
        if self.model is None or self.purchase_model is None or self.redemption_model is None:
            print_error("请先训练模型")
            return False

        if steps is None:
            steps = ARIMA_TRAINING_CONFIG["预测配置"]["预测步数"]  # 从配置文件读取预测步数

        print_header("多变量ARIMA预测", f"预测未来{steps}天")

        try:
            # 预测净资金流
            print("预测净资金流...")
            net_flow_forecast = self.model.forecast(steps=steps)

            # 添加一些随机波动，避免过于平稳
            np.random.seed(42)  # 设置随机种子确保可重复性
            noise_ratio = ARIMA_TRAINING_CONFIG.get("预测配置", {}).get("噪声比例", 0.1)  # 从配置文件读取噪声比例
            noise_scale = net_flow_forecast.std() * noise_ratio
            noise = np.random.normal(0, noise_scale, len(net_flow_forecast))
            net_flow_forecast = net_flow_forecast + noise

            # 如果净资金流使用了差分，需要还原
            if len(self.time_series) != len(self.original_net_flow_series):
                print("净资金流使用了差分，正在还原...")
                last_original_value = self.original_net_flow_series.iloc[-1]
                net_flow_forecast = net_flow_forecast.cumsum() + last_original_value
                print(f"还原后净资金流预测值范围: {net_flow_forecast.min():.2f} 到 {net_flow_forecast.max():.2f}")

            # 预测申购金额
            print("预测申购金额...")
            print(f"申购金额时间序列长度: {len(self.purchase_series)}")
            print(f"申购金额时间序列范围: {self.purchase_series.min():.2f} 到 {self.purchase_series.max():.2f}")
            print(f"申购金额时间序列均值: {self.purchase_series.mean():.2f}")
            purchase_forecast = self.purchase_model.forecast(steps=steps)

            # 添加一些随机波动，避免过于平稳
            noise_ratio_purchase = ARIMA_TRAINING_CONFIG.get("预测配置", {}).get("申购噪声比例", 0.15)  # 从配置文件读取申购噪声比例
            noise_scale_purchase = purchase_forecast.std() * noise_ratio_purchase
            noise_purchase = np.random.normal(0, noise_scale_purchase, len(purchase_forecast))
            purchase_forecast = purchase_forecast + noise_purchase

            print(f"申购金额预测值范围: {purchase_forecast.min():.2f} 到 {purchase_forecast.max():.2f}")
            print(f"申购金额预测值均值: {purchase_forecast.mean():.2f}")

            # 如果申购金额使用了差分，需要还原
            if len(self.purchase_series) != len(self.original_purchase_series):
                print("申购金额使用了差分，正在还原...")
                last_original_value = self.original_purchase_series.iloc[-1]
                purchase_forecast = purchase_forecast.cumsum() + last_original_value
                print(f"还原后申购金额预测值范围: {purchase_forecast.min():.2f} 到 {purchase_forecast.max():.2f}")

            # 预测赎回金额
            print("预测赎回金额...")
            print(f"赎回金额时间序列长度: {len(self.redemption_series)}")
            print(f"赎回金额时间序列范围: {self.redemption_series.min():.2f} 到 {self.redemption_series.max():.2f}")
            print(f"赎回金额时间序列均值: {self.redemption_series.mean():.2f}")
            redemption_forecast = self.redemption_model.forecast(steps=steps)

            # 添加更多的随机波动，避免过于平稳
            noise_ratio_redemption = ARIMA_TRAINING_CONFIG.get("预测配置", {}).get("赎回噪声比例", 0.25)  # 增加噪声比例
            noise_scale_redemption = redemption_forecast.std() * noise_ratio_redemption
            if noise_scale_redemption == 0 or np.isnan(noise_scale_redemption):
                # 如果标准差为0或NaN，使用原始数据的标准差
                noise_scale_redemption = self.original_redemption_series.std() * 0.1

            noise_redemption = np.random.normal(0, noise_scale_redemption, len(redemption_forecast))
            redemption_forecast = redemption_forecast + noise_redemption

            print(f"赎回金额预测值范围: {redemption_forecast.min():.2f} 到 {redemption_forecast.max():.2f}")
            print(f"赎回金额预测值均值: {redemption_forecast.mean():.2f}")

            # 如果赎回金额使用了差分，需要还原
            if len(self.redemption_series) != len(self.original_redemption_series):
                print("赎回金额使用了差分，正在还原...")
                last_original_value = self.original_redemption_series.iloc[-1]

                # 改进的差分还原逻辑
                redemption_forecast_cumsum = redemption_forecast.cumsum()
                redemption_forecast = redemption_forecast_cumsum + last_original_value

                # 确保还原后的值在合理范围内
                min_redemption = self.original_redemption_series.min() * 0.5
                max_redemption = self.original_redemption_series.max() * 1.5
                redemption_forecast = np.clip(redemption_forecast, min_redemption, max_redemption)

                print(f"还原后赎回金额预测值范围: {redemption_forecast.min():.2f} 到 {redemption_forecast.max():.2f}")
            else:
                # 如果没有使用差分，也添加一些随机波动
                print("赎回金额未使用差分，添加随机波动...")
                # 添加基于原始数据标准差的随机波动
                original_std = self.original_redemption_series.std()
                additional_noise = np.random.normal(0, original_std * 0.1, len(redemption_forecast))
                redemption_forecast = redemption_forecast + additional_noise

                # 确保预测值在合理范围内
                min_redemption = self.original_redemption_series.min() * 0.5
                max_redemption = self.original_redemption_series.max() * 1.5
                redemption_forecast = np.clip(redemption_forecast, min_redemption, max_redemption)

                print(f"添加波动后赎回金额预测值范围: {redemption_forecast.min():.2f} 到 {redemption_forecast.max():.2f}")

            # 创建预测结果数据框
            last_date = self.time_series.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')

            self.predictions = pd.Series(net_flow_forecast, index=future_dates)
            self.purchase_predictions = pd.Series(purchase_forecast, index=future_dates)
            self.redemption_predictions = pd.Series(redemption_forecast, index=future_dates)

            print_success("多变量预测完成")
            print(f"净资金流预测范围: {self.predictions.index[0].date()} 到 {self.predictions.index[-1].date()}")
            print(f"申购金额预测范围: {self.purchase_predictions.index[0].date()} 到 {self.purchase_predictions.index[-1].date()}")
            print(f"赎回金额预测范围: {self.redemption_predictions.index[0].date()} 到 {self.redemption_predictions.index[-1].date()}")

            return True

        except Exception as e:
            print_error(f"预测失败: {e}")
            return False

    
    def save_results(self):
        """保存多变量预测结果"""
        if (self.predictions is None or self.purchase_predictions is None or 
            self.redemption_predictions is None):
            print_error("没有预测结果可保存")
            return False

        try:
            # 创建输出目录
            output_dir = OUTPUT_DIR / "data"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存预测结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            predictions_file = output_dir / f"multi_arima_predictions_{timestamp}.csv"

            # 创建结果数据框
            results_df = pd.DataFrame({
                '预测日期': self.predictions.index,
                '净资金流预测值': self.predictions.values,
                '申购金额预测值': self.purchase_predictions.values,
                '赎回金额预测值': self.redemption_predictions.values
            })

            # 保存到文件
            results_df.to_csv(predictions_file, index=False, encoding='utf-8')
            print_success(f"多变量预测结果已保存: {predictions_file}")

            return True

        except Exception as e:
            print_error(f"保存结果失败: {e}")
            return False

    
    def generate_enhanced_visualizations(self):
        """生成增强的多变量可视化图表"""
        if (self.predictions is None or self.purchase_predictions is None or 
            self.redemption_predictions is None or self.data is None):
            print_error("没有预测结果或数据可可视化")
            return False

        print_header("生成增强多变量可视化图表", "申购赎回净资金流预测结果")

        try:
            # 创建输出目录
            output_dir = IMAGES_DIR / "arima"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. 生成多变量综合预测图
            print_info("生成多变量综合预测图...")
            comprehensive_file = output_dir / f"multi_comprehensive_predictions_{timestamp}.png"
            self._plot_multi_variable_predictions(comprehensive_file)

            # 2. 生成多变量预测摘要图
            print_info("生成多变量预测摘要图...")
            summary_file = output_dir / f"multi_prediction_summary_{timestamp}.png"
            self._plot_multi_variable_summary(summary_file)

            print_success("增强多变量可视化图表生成完成")
            return True

        except Exception as e:
            print_error(f"生成增强多变量可视化图表失败: {e}")
            return False

    
    def _plot_multi_variable_predictions(self, save_path):
        """绘制多变量预测图"""
        try:
            import matplotlib.pyplot as plt
            from utils.visualization_utils import setup_matplotlib

            setup_matplotlib()

            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            fig.suptitle('多变量ARIMA预测结果', fontsize=16, fontweight='bold')

            # 1. 净资金流预测
            time_field = get_field_name("时间字段")
            original_net_flow = self.data.groupby(time_field)['Net_Flow'].sum()

            axes[0].plot(original_net_flow.index, original_net_flow.values, 
                        label='实际净资金流', color='blue', linewidth=2, alpha=0.8)
            axes[0].plot(self.predictions.index, self.predictions.values, 
                        label='预测净资金流', color='red', linewidth=2, linestyle='--')
            axes[0].set_title('净资金流预测', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('时间')
            axes[0].set_ylabel('净资金流')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 2. 申购金额预测
            original_purchase = self.data.groupby(time_field)['total_purchase_amt'].sum()

            axes[1].plot(original_purchase.index, original_purchase.values, 
                        label='实际申购金额', color='green', linewidth=2, alpha=0.8)
            axes[1].plot(self.purchase_predictions.index, self.purchase_predictions.values, 
                        label='预测申购金额', color='darkgreen', linewidth=2, linestyle='--')
            axes[1].set_title('申购金额预测', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('时间')
            axes[1].set_ylabel('申购金额')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # 3. 赎回金额预测
            original_redemption = self.data.groupby(time_field)['total_redeem_amt'].sum()

            axes[2].plot(original_redemption.index, original_redemption.values, 
                        label='实际赎回金额', color='orange', linewidth=2, alpha=0.8)
            axes[2].plot(self.redemption_predictions.index, self.redemption_predictions.values, 
                        label='预测赎回金额', color='darkorange', linewidth=2, linestyle='--')
            axes[2].set_title('赎回金额预测', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('时间')
            axes[2].set_ylabel('赎回金额')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print_success(f"多变量预测图已保存: {save_path}")

        except Exception as e:
            print_error(f"绘制多变量预测图失败: {e}")

    
    def _plot_multi_variable_summary(self, save_path):
        """绘制多变量预测摘要图"""
        try:
            import matplotlib.pyplot as plt
            from utils.visualization_utils import setup_matplotlib

            setup_matplotlib()

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('多变量ARIMA预测摘要', fontsize=16, fontweight='bold')

            # 1. 预测值对比
            axes[0, 0].plot(self.predictions.index, self.predictions.values, 
                           label='净资金流', color='blue', linewidth=2)
            axes[0, 0].plot(self.purchase_predictions.index, self.purchase_predictions.values, 
                           label='申购金额', color='green', linewidth=2)
            axes[0, 0].plot(self.redemption_predictions.index, self.redemption_predictions.values, 
                           label='赎回金额', color='orange', linewidth=2)
            axes[0, 0].set_title('预测值对比', fontsize=12)
            axes[0, 0].set_xlabel('时间')
            axes[0, 0].set_ylabel('金额')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 预测统计信息
            stats_text = f"净资金流预测统计:\n"
            stats_text += f"  均值: {self.predictions.mean():.2f}\n"
            stats_text += f"  标准差: {self.predictions.std():.2f}\n"
            stats_text += f"  最大值: {self.predictions.max():.2f}\n"
            stats_text += f"  最小值: {self.predictions.min():.2f}\n\n"
            stats_text += f"申购金额预测统计:\n"
            stats_text += f"  均值: {self.purchase_predictions.mean():.2f}\n"
            stats_text += f"  标准差: {self.purchase_predictions.std():.2f}\n"
            stats_text += f"  最大值: {self.purchase_predictions.max():.2f}\n"
            stats_text += f"  最小值: {self.purchase_predictions.min():.2f}"

            axes[0, 1].text(0.05, 0.95, stats_text, transform=axes[0, 1].transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[0, 1].set_title('预测统计信息', fontsize=12)
            axes[0, 1].axis('off')

            # 3. 净资金流趋势分析
            axes[1, 0].plot(self.predictions.index, self.predictions.values, 
                           color='blue', linewidth=2, label='净资金流趋势')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='零线')
            axes[1, 0].set_title('净资金流趋势分析', fontsize=12)
            axes[1, 0].set_xlabel('时间')
            axes[1, 0].set_ylabel('净资金流')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 申购赎回比例
            ratio = self.purchase_predictions / (self.purchase_predictions + self.redemption_predictions)
            axes[1, 1].plot(self.predictions.index, ratio, 
                           color='purple', linewidth=2, label='申购比例')
            axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50%线')
            axes[1, 1].set_title('申购赎回比例趋势', fontsize=12)
            axes[1, 1].set_xlabel('时间')
            axes[1, 1].set_ylabel('申购比例')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print_success(f"多变量预测摘要图已保存: {save_path}")

        except Exception as e:
            print_error(f"绘制多变量预测摘要图失败: {e}")

    
    def run_full_arima_analysis(self):
        """运行完整的多变量ARIMA分析流程"""
        print_header("完整多变量ARIMA分析", "数据准备 -> 模型训练 -> 预测 -> 可视化")

        try:
            # 1. 准备数据
            if not self.prepare_data_for_arima():
                return False

            # 2. 训练模型
            if not self.train_arima_model():
                return False

            # 3. 进行预测
            if not self.make_predictions():
                return False

            # 4. 保存结果
            if not self.save_results():
                return False

            # 5. 生成增强可视化
            if not self.generate_enhanced_visualizations():
                return False

            print_success("完整多变量ARIMA分析流程执行完成！")
            return True

        except Exception as e:
            print_error(f"多变量ARIMA分析失败: {e}")
            return False



def run_arima_prediction():
    """运行多变量ARIMA预测分析"""
    print_header("多变量ARIMA预测", "净资金流、申购金额、赎回金额预测")

    # 创建ARIMA预测器实例
    predictor = ARIMAPredictorMain()

    # 加载数据
    if not predictor.load_data():
        print_error("数据加载失败")
        return False

    # 运行完整的多变量ARIMA分析
    success = predictor.run_full_arima_analysis()

    if success:
        print_success("多变量ARIMA预测分析完成！")
        print_info("预测结果已保存到输出目录")
    else:
        print_error("多变量ARIMA预测分析失败")

    return success
