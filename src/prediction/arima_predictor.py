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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.var_model import VAR

from utils.data_processor import DataProcessor
from utils.interactive_utils import print_header, print_success, print_error, print_info, print_warning
from config import DATA_DIR, OUTPUT_DIR, IMAGES_DIR
from config.forecast import GLOBAL_FORECAST_CONFIG
from config.arima_config import (
    ARIMA_TRAINING_CONFIG, 
    ARIMA_OUTPUT_CONFIG, 
    ARIMA_SEASONALITY_CONFIG,
    ARIMA_OPTIMIZATION_CONFIG,
    ARIMA_VISUALIZATION_CONFIG,
    ARIMA_DATA_PREPROCESSING_CONFIG
)
import json
from pathlib import Path
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
        
        # 周期性检测结果
        self.seasonal_periods = {}
        self.seasonal_strength = {}
        self.seasonal_patterns = {}
        self.predictions = None
        self.purchase_predictions = None
        self.redemption_predictions = None
        self.visualizer = ARIMAVisualizer()
        self.enhanced_visualizer = ARIMAVisualizationEnhanced()

    
    def load_data(self, file_path=None, use_data_processing=False, module_name=None):
        """重写load_data方法，优先加载预处理后的数据"""
        try:
            # 从配置文件获取预处理文件配置
            from config.data_processing import GLOBAL_DATA_PROCESSING_CONFIG
            preprocess_config = GLOBAL_DATA_PROCESSING_CONFIG.get("预处理文件配置", {})
            
            # 获取配置的预处理文件名和路径
            primary_filename = preprocess_config.get("预处理文件名", "user_balance_table_processed.csv")
            backup_filenames = preprocess_config.get("备用文件名", [])
            file_path = preprocess_config.get("文件路径", "output/data")
            force_use_preprocessed = preprocess_config.get("强制使用预处理", True)
            file_format = preprocess_config.get("文件格式", "csv")
            
            print_info(f"查找预处理数据文件，配置路径: {file_path}")
            print_info(f"主要文件名: {primary_filename}")
            print_info(f"备用文件名: {backup_filenames}")
            
            # 构建完整的文件路径
            if file_path.startswith("output"):
                # 相对路径，从项目根目录开始
                from config import ROOT_DIR
                search_dir = ROOT_DIR / file_path
            else:
                search_dir = OUTPUT_DIR / file_path
            
            # 首先尝试主要文件名
            primary_file = search_dir / primary_filename
            if primary_file.exists():
                print_info(f"找到主要预处理文件: {primary_file}")
                self.data = pd.read_csv(primary_file, encoding='utf-8')
                print_success(f"预处理数据加载成功: {len(self.data):,} 条记录")
                print_info(f"数据包含 {len(self.data.columns)} 个特征")
                
                # 显示前几个特征列名
                feature_columns = [col for col in self.data.columns if col not in ['user_id', 'report_date']]
                print_info(f"特征列示例: {feature_columns[:5]}")
                return True
            
            # 尝试备用文件名
            for backup_filename in backup_filenames:
                backup_file = search_dir / backup_filename
                if backup_file.exists():
                    print_info(f"找到备用预处理文件: {backup_file}")
                    self.data = pd.read_csv(backup_file, encoding='utf-8')
                    print_success(f"预处理数据加载成功: {len(self.data):,} 条记录")
                    print_info(f"数据包含 {len(self.data.columns)} 个特征")
                    
                    # 显示前几个特征列名
                    feature_columns = [col for col in self.data.columns if col not in ['user_id', 'report_date']]
                    print_info(f"特征列示例: {feature_columns[:5]}")
                    return True
            
            # 如果强制使用预处理数据但找不到，则报错
            if force_use_preprocessed:
                print_error(f"未找到预处理数据文件")
                print_error(f"已尝试以下文件:")
                print_error(f"  主要文件: {primary_file}")
                for backup_filename in backup_filenames:
                    print_error(f"  备用文件: {search_dir / backup_filename}")
                print_error("请先运行数据预处理功能生成预处理数据")
                return False
            else:
                # 如果不强制使用预处理数据，尝试加载原始数据
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

            # 检测周期性模式
            print("检测周期性模式...")
            self._detect_seasonality(net_flow_series, 'Net_Flow')
            self._detect_seasonality(purchase_series, 'Purchase_Amount')
            self._detect_seasonality(redemption_series, 'Redemption_Amount')

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
            print("训练净资金流模型...")
            
            # 检查是否有强季节性
            seasonal_strength = self.seasonal_strength.get('Net_Flow', 0)
            seasonal_periods = self.seasonal_periods.get('Net_Flow', [7])
            main_period = seasonal_periods[0] if seasonal_periods else 7
            
            if seasonal_strength > 0.3 and len(self.time_series) >= main_period * 4:
                # 使用SARIMA模型
                print(f"  检测到强季节性（强度: {seasonal_strength:.4f}），使用SARIMA模型")
                self.model = self._train_sarima_model(self.time_series, 'Net_Flow', main_period)
            else:
                # 使用ARIMA模型
                print(f"  使用ARIMA模型")
                self.model = self._train_arima_model(self.time_series, 'Net_Flow')

            # 训练申购金额模型
            print("训练申购金额模型...")
            
            # 检查是否有强季节性
            seasonal_strength = self.seasonal_strength.get('Purchase_Amount', 0)
            seasonal_periods = self.seasonal_periods.get('Purchase_Amount', [7])
            main_period = seasonal_periods[0] if seasonal_periods else 7
            
            if seasonal_strength > 0.3 and len(self.purchase_series) >= main_period * 4:
                # 使用SARIMA模型
                print(f"  检测到强季节性（强度: {seasonal_strength:.4f}），使用SARIMA模型")
                self.purchase_model = self._train_sarima_model(self.purchase_series, 'Purchase_Amount', main_period)
            else:
                # 使用ARIMA模型
                print(f"  使用ARIMA模型")
                self.purchase_model = self._train_arima_model(self.purchase_series, 'Purchase_Amount')

            # 训练赎回金额模型
            print("训练赎回金额模型...")
            
            # 检查是否有强季节性
            seasonal_strength = self.seasonal_strength.get('Redemption_Amount', 0)
            seasonal_periods = self.seasonal_periods.get('Redemption_Amount', [7])
            main_period = seasonal_periods[0] if seasonal_periods else 7
            
            if seasonal_strength > 0.3 and len(self.redemption_series) >= main_period * 4:
                # 使用SARIMA模型
                print(f"  检测到强季节性（强度: {seasonal_strength:.4f}），使用SARIMA模型")
                self.redemption_model = self._train_sarima_model(self.redemption_series, 'Redemption_Amount', main_period)
            else:
                # 使用ARIMA模型
                print(f"  使用ARIMA模型")
                self.redemption_model = self._train_arima_model(self.redemption_series, 'Redemption_Amount')

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
            steps = ARIMA_TRAINING_CONFIG.get("预测配置", {}).get("预测步数", 30)  # 默认30天

        print_header("多变量ARIMA预测", f"预测未来{steps}天")

        try:
            # 在预测前，根据全局配置统一调整步数，确保三条序列与日期索引长度一致
            last_date = self.time_series.index[-1]
            try:
                if GLOBAL_FORECAST_CONFIG.get("enabled", False):
                    end_date_str = GLOBAL_FORECAST_CONFIG.get("end_date_exclusive")
                    if end_date_str:
                        end_date_exclusive = pd.to_datetime(end_date_str)
                        calc_steps = (end_date_exclusive - (last_date + pd.Timedelta(days=1))).days
                        steps = max(0, calc_steps)
                        print_info(f"全局预测结束日期生效，预测步数重设为: {steps}")
            except Exception as e:
                print_warning(f"全局预测时间配置处理失败，继续使用默认步数: {e}")

            # 预测净资金流
            print("预测净资金流...")
            net_flow_forecast = self.model.forecast(steps=steps)

            # 添加季节性调整
            if 'Net_Flow' in self.seasonal_patterns:
                seasonal_pattern = self.seasonal_patterns['Net_Flow']
                if len(seasonal_pattern) > 0:
                    print("  应用净资金流季节性调整...")
                    net_flow_forecast = self._apply_seasonal_adjustment(net_flow_forecast, seasonal_pattern)

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

            # 添加季节性调整
            if 'Purchase_Amount' in self.seasonal_patterns:
                seasonal_pattern = self.seasonal_patterns['Purchase_Amount']
                if len(seasonal_pattern) > 0:
                    print("  应用申购金额季节性调整...")
                    purchase_forecast = self._apply_seasonal_adjustment(purchase_forecast, seasonal_pattern)

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

            # 添加季节性调整
            if 'Redemption_Amount' in self.seasonal_patterns:
                seasonal_pattern = self.seasonal_patterns['Redemption_Amount']
                if len(seasonal_pattern) > 0:
                    print("  应用赎回金额季节性调整...")
                    redemption_forecast = self._apply_seasonal_adjustment(redemption_forecast, seasonal_pattern)

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
            # 从配置文件获取输出路径
            data_output_path = ARIMA_OUTPUT_CONFIG.get("数据输出路径", "output/arima")
            data_format = ARIMA_OUTPUT_CONFIG.get("数据格式", "csv")
            
            # 创建输出目录
            import os
            os.makedirs(data_output_path, exist_ok=True)

            # 保存预测结果
            predictions_file = os.path.join(data_output_path, f"multi_arima_predictions.{data_format}")

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

            # 额外输出一份与 predict.py 匹配的CSV格式
            # 列: report_date(YYYYMMDD字符串), purchase, redeem；编码: UTF-8-SIG
            matched_df = pd.DataFrame({
                'report_date': self.predictions.index.strftime('%Y%m%d'),
                'purchase': self.purchase_predictions.values,
                'redeem': self.redemption_predictions.values
            })
            matched_file = os.path.join(data_output_path, f"arima_forecast_201409.{data_format}")
            matched_df.to_csv(matched_file, index=False, encoding='utf-8-sig')
            print_success(f"已额外输出与predict.py匹配的CSV: {matched_file}")

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
            # 从配置文件获取输出路径
            image_output_path = ARIMA_OUTPUT_CONFIG.get("图片保存路径", "output/arima")
            
            # 创建输出目录
            import os
            os.makedirs(image_output_path, exist_ok=True)

            # 1. 生成多变量综合预测图
            print_info("生成多变量综合预测图...")
            comprehensive_file = os.path.join(image_output_path, "multi_comprehensive_predictions.png")
            self._plot_multi_variable_predictions(comprehensive_file)

            # 2. 生成多变量预测摘要图
            print_info("生成多变量预测摘要图...")
            summary_file = os.path.join(image_output_path, "multi_prediction_summary.png")
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

    # ==================== 模型训练方法 ====================

    def _train_sarima_model(self, series, series_name, seasonal_period):
        """训练SARIMA模型（会遍历候选周期）"""
        try:
            # 参数缓存：键 = (series_name, seasonal_period)
            cache_conf = ARIMA_OPTIMIZATION_CONFIG.get("参数缓存", {"启用": False})
            cache_enabled = cache_conf.get("启用", False)
            cache_path = Path(cache_conf.get("路径", "cache/arima_param_cache.json"))
            cache = {}
            if cache_enabled and cache_path.exists():
                try:
                    cache = json.load(open(cache_path, 'r'))
                except Exception:
                    cache = {}
            cache_key = f"SARIMA::{series_name}::s={seasonal_period}"
            if cache_enabled and cache_key in cache:
                params = cache[cache_key]
                try:
                    model = SARIMAX(
                        series,
                        order=tuple(params["order"]),
                        seasonal_order=tuple(params["seasonal_order"]),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted_model = model.fit(disp=False)
                    print_success(
                        f"  使用缓存参数 → SARIMA order={params['order']}, seasonal={params['seasonal_order']}"
                    )
                    try:
                        s_cache = params["seasonal_order"][3]
                        print_info(f"  最终选择季节周期 s={s_cache}（来源: 缓存）")
                    except Exception:
                        pass
                    return fitted_model
                except Exception:
                    print_warning("  缓存参数无效，退回搜索")

            # SARIMA参数网格搜索
            best_aic = float('inf')
            best_model = None
            best_params = None
            
            # 参数范围
            mconf = ARIMA_TRAINING_CONFIG["模型参数"]["ARIMA参数"]["季节性参数"]
            p_range = [0, 1, 2, 3]
            d_range = [0, 1]
            q_range = [0, 1, 2, 3]
            P_range = mconf.get("P_range", [0,1,2])
            D_range = mconf.get("D_range", [0,1])
            Q_range = mconf.get("Q_range", [0,1,2])
            seasonal_candidates = mconf.get("候选周期", [seasonal_period])
            
            print(f"  SARIMA参数搜索范围: p={p_range}, d={d_range}, q={q_range}")
            print(f"  季节性参数: P={P_range}, D={D_range}, Q={Q_range}, s候选={seasonal_candidates}")
            
            # 限制搜索次数
            max_combinations = 30
            combinations_tried = 0
            
            for s in seasonal_candidates:
                for p in p_range:
                    for d in d_range:
                        for q in q_range:
                            for P in P_range:
                                for D in D_range:
                                    for Q in Q_range:
                                        if combinations_tried >= max_combinations:
                                            break
                                        try:
                                            model = SARIMAX(
                                                series,
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, s),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False
                                            )
                                            fitted_model = model.fit(disp=False)
                                            aic = fitted_model.aic
                                            if aic < best_aic:
                                                best_aic = aic
                                                best_model = fitted_model
                                                best_params = (p, d, q, P, D, Q, s)
                                            combinations_tried += 1
                                        except Exception:
                                            continue
                                
                                if combinations_tried >= max_combinations:
                                    break
                            if combinations_tried >= max_combinations:
                                break
                        if combinations_tried >= max_combinations:
                            break
                    if combinations_tried >= max_combinations:
                        break
                if combinations_tried >= max_combinations:
                    break
            
            if best_model is not None:
                print(f"  SARIMA最佳参数: {best_params}, AIC: {best_aic:.2f}")
                # 日志：输出最终选择的季节周期
                try:
                    print_info(f"  最终选择季节周期 s={best_params[6]}（来源: 搜索）")
                except Exception:
                    pass
                if cache_enabled:
                    cache[cache_key] = {
                        "order": list(best_params[:3]),
                        "seasonal_order": [best_params[3], best_params[4], best_params[5], best_params[6]]
                    }
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    json.dump(cache, open(cache_path, 'w'), ensure_ascii=False, indent=2)
                return best_model
            else:
                print_warning(f"  {series_name} SARIMA模型训练失败，回退到ARIMA")
                return self._train_arima_model(series, series_name)
                
        except Exception as e:
            print_warning(f"  {series_name} SARIMA训练失败: {e}")
            return self._train_arima_model(series, series_name)

    def _train_arima_model(self, series, series_name):
        """训练ARIMA模型"""
        try:
            # 参数缓存
            cache_conf = ARIMA_OPTIMIZATION_CONFIG.get("参数缓存", {"启用": False})
            cache_enabled = cache_conf.get("启用", False)
            cache_path = Path(cache_conf.get("路径", "cache/arima_param_cache.json"))
            cache = {}
            if cache_enabled and cache_path.exists():
                try:
                    cache = json.load(open(cache_path, 'r'))
                except Exception:
                    cache = {}
            cache_key = f"ARIMA::{series_name}"
            if cache_enabled and cache_key in cache:
                params = cache[cache_key]
                try:
                    model = ARIMA(series, order=tuple(params))
                    fitted_model = model.fit()
                    print_success(f"  使用缓存参数 → ARIMA order={tuple(params)}")
                    return fitted_model
                except Exception:
                    print_warning("  缓存参数无效，退回搜索")

            best_aic = float('inf')
            best_model = None
            best_params = None
            
            # 从配置文件获取参数范围
            model_config = ARIMA_TRAINING_CONFIG["模型参数"]["ARIMA参数"]
            p_values = model_config["p_range"]
            d_values = model_config["d_range"]
            q_values = model_config["q_range"]
            
            print(f"  ARIMA参数搜索范围: p={p_values}, d={d_values}, q={q_values}")
            
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
                    print(f"  结合自相关分析后的参数范围: p={p_values}, q={q_values}")
            except Exception as _:
                pass
            
            # 限制搜索次数
            max_combinations = ARIMA_OPTIMIZATION_CONFIG.get("搜索策略", {}).get("最大组合数", 20)
            combinations_tried = 0
            
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        if combinations_tried >= max_combinations:
                            break
                            
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted_model = model.fit()
                            aic = fitted_model.aic
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_model = fitted_model
                                best_params = (p, d, q)
                                
                            combinations_tried += 1
                            
                        except Exception:
                            continue
                    
                    if combinations_tried >= max_combinations:
                        break
                if combinations_tried >= max_combinations:
                    break
            
            if best_model is not None:
                print(f"  ARIMA最佳参数: {best_params}, AIC: {best_aic:.2f}")
                print_info("  参数来源: 搜索")
                if cache_enabled:
                    cache[cache_key] = list(best_params)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    json.dump(cache, open(cache_path, 'w'), ensure_ascii=False, indent=2)
                return best_model
            else:
                print_warning(f"  {series_name} ARIMA模型训练失败，使用默认参数")
                return ARIMA(series, order=(1, 1, 1)).fit()
                
        except Exception as e:
            print_error(f"  {series_name} ARIMA训练失败: {e}")
            return ARIMA(series, order=(1, 1, 1)).fit()

    # ==================== 周期性检测方法 ====================

    def _detect_seasonality(self, series, series_name):
        """检测季节性"""
        try:
            print(f"检测 {series_name} 的季节性...")
            
            # 1. 自相关分析检测周期性
            lags = min(50, len(series) // 4)
            acf_values = pd.Series(series).autocorr(lag=1)
            
            # 检测主要周期
            periods = []
            for period in [7, 14, 30, 90, 365]:  # 周、双周、月、季度、年
                if len(series) >= period * 2:
                    seasonal_score = self._calculate_seasonal_score(series, period)
                    if seasonal_score > 0.3:  # 阈值可调整
                        periods.append(period)
                        print_info(f"  检测到 {period} 天周期，强度: {seasonal_score:.4f}")
            
            # 2. 季节性分解
            if len(series) >= 30:
                try:
                    decomposition = seasonal_decompose(series, period=7, extrapolate_trend='freq')
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(series)
                    
                    self.seasonal_periods[series_name] = periods
                    self.seasonal_strength[series_name] = seasonal_strength
                    self.seasonal_patterns[series_name] = decomposition.seasonal
                    
                    print_info(f"  {series_name} 季节性强度: {seasonal_strength:.4f}")
                    
                except Exception as e:
                    print_warning(f"季节性分解失败: {e}")
                    self.seasonal_periods[series_name] = [7]  # 默认周周期
                    self.seasonal_strength[series_name] = 0.1
            else:
                self.seasonal_periods[series_name] = [7]
                self.seasonal_strength[series_name] = 0.1
                
        except Exception as e:
            print_warning(f"季节性检测失败: {e}")
            self.seasonal_periods[series_name] = [7]
            self.seasonal_strength[series_name] = 0.1

    def _calculate_seasonal_score(self, series, period):
        """计算季节性得分"""
        try:
            if len(series) < period * 2:
                return 0
            
            # 计算不同周期段之间的相关性
            segments = []
            for i in range(0, len(series) - period, period):
                segment = series.iloc[i:i+period]
                if len(segment) == period:
                    segments.append(segment.values)
            
            if len(segments) < 2:
                return 0
            
            # 计算段间相关性
            correlations = []
            for i in range(len(segments)):
                for j in range(i+1, len(segments)):
                    corr = np.corrcoef(segments[i], segments[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                return np.mean(correlations)
            return 0
            
        except Exception:
            return 0

    def _apply_seasonal_adjustment(self, forecast, seasonal_pattern):
        """应用季节性调整"""
        try:
            pattern_length = len(seasonal_pattern)
            if pattern_length == 0:
                return forecast
            
            # 获取季节性模式的均值
            seasonal_mean = seasonal_pattern.mean()
            
            # 应用季节性调整
            adjusted_forecast = forecast.copy()
            for i in range(len(forecast)):
                pattern_index = i % pattern_length
                seasonal_factor = seasonal_pattern.iloc[pattern_index] - seasonal_mean
                adjusted_forecast.iloc[i] += seasonal_factor * 0.3  # 调整强度
            
            return adjusted_forecast
            
        except Exception as e:
            print_warning(f"季节性调整失败: {e}")
            return forecast

    def generate_purchase_redemption_visualization(self):
            """生成申购赎回历史数据与预测数据的对比可视化图"""
            try:
                print_header("生成申购赎回可视化图")
                
                if (self.original_purchase_series is None or 
                    self.original_redemption_series is None or
                    self.purchase_predictions is None or
                    self.redemption_predictions is None):
                    print_error("缺少必要的数据，无法生成可视化图")
                    return False
                
                import matplotlib.pyplot as plt
                import matplotlib.dates as mdates
                from datetime import datetime
                
                # 设置中文字体
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                
                # 创建图形
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
                
                # 获取历史数据的时间范围
                historical_dates = self.original_purchase_series.index
                forecast_dates = self.purchase_predictions.index
                
                # 1. 申购金额对比图
                # 确保数据类型正确
                historical_purchase_values = pd.to_numeric(self.original_purchase_series.values, errors='coerce')
                forecast_purchase_values = pd.to_numeric(self.purchase_predictions.values, errors='coerce')
                
                ax1.plot(historical_dates, historical_purchase_values, 
                        color='blue', linewidth=2, label='历史申购金额', alpha=0.8)
                ax1.plot(forecast_dates, forecast_purchase_values, 
                        color='blue', linewidth=2, linestyle='--', label='预测申购金额', alpha=0.8)
                
                # 添加分隔线
                if len(historical_dates) > 0:
                    last_historical_date = historical_dates[-1]
                    ax1.axvline(x=last_historical_date, color='red', linestyle=':', alpha=0.7, 
                               label='历史/预测分界线')
                
                ax1.set_title('申购金额历史数据与预测数据对比', fontsize=16, fontweight='bold')
                ax1.set_xlabel('日期', fontsize=12)
                ax1.set_ylabel('申购金额', fontsize=12)
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                # 格式化x轴日期
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                
                # 2. 赎回金额对比图
                # 确保数据类型正确
                historical_redemption_values = pd.to_numeric(self.original_redemption_series.values, errors='coerce')
                forecast_redemption_values = pd.to_numeric(self.redemption_predictions.values, errors='coerce')
                
                ax2.plot(historical_dates, historical_redemption_values, 
                        color='red', linewidth=2, label='历史赎回金额', alpha=0.8)
                ax2.plot(forecast_dates, forecast_redemption_values, 
                        color='red', linewidth=2, linestyle='--', label='预测赎回金额', alpha=0.8)
                
                # 添加分隔线
                if len(historical_dates) > 0:
                    ax2.axvline(x=last_historical_date, color='red', linestyle=':', alpha=0.7, 
                               label='历史/预测分界线')
                
                ax2.set_title('赎回金额历史数据与预测数据对比', fontsize=16, fontweight='bold')
                ax2.set_xlabel('日期', fontsize=12)
                ax2.set_ylabel('赎回金额', fontsize=12)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                # 格式化x轴日期
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax2.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                
                # 调整布局
                plt.tight_layout()
                
                # 保存图片
                import os
                
                # 从配置文件获取保存路径
                image_path = ARIMA_OUTPUT_CONFIG.get("图片保存路径", "output/arima")
                image_format = ARIMA_OUTPUT_CONFIG.get("图片格式", "png")
                image_dpi = ARIMA_OUTPUT_CONFIG.get("图片DPI", 300)
                
                # 确保目录存在
                os.makedirs(image_path, exist_ok=True)
                
                filename = f"purchase_redemption_forecast_comparison.{image_format}"
                full_path = os.path.join(image_path, filename)
                
                # 直接保存图片
                plt.savefig(full_path, dpi=image_dpi, bbox_inches='tight')
                print_success(f"申购赎回可视化图已保存: {full_path}")
                
                print_success(f"申购赎回可视化图已保存: {filename}")
                
                # 显示图片
                plt.show()
                
                return True
                
            except Exception as e:
                print_error(f"生成申购赎回可视化图失败: {e}")
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


def _train_arima_model(self, series, series_name):
    """训练ARIMA模型"""
    try:
        best_aic = float('inf')
        best_model = None
        best_params = None
        
        # 从配置文件获取参数范围
        model_config = ARIMA_TRAINING_CONFIG["模型参数"]["ARIMA参数"]
        p_values = model_config["p_range"]
        d_values = model_config["d_range"]
        q_values = model_config["q_range"]
        
        print(f"  ARIMA参数搜索范围: p={p_values}, d={d_values}, q={q_values}")
        
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
                print(f"  结合自相关分析后的参数范围: p={p_values}, q={q_values}")
        except Exception as _:
            pass
        
        # 限制搜索次数
        max_combinations = 20
        combinations_tried = 0
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    if combinations_tried >= max_combinations:
                        break
                        
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_model = fitted_model
                            best_params = (p, d, q)
                            
                        combinations_tried += 1
                        
                    except Exception:
                        continue
                
                if combinations_tried >= max_combinations:
                    break
            if combinations_tried >= max_combinations:
                break
        
        if best_model is not None:
            print(f"  ARIMA最佳参数: {best_params}, AIC: {best_aic:.2f}")
            return best_model
        else:
            print_warning(f"  {series_name} ARIMA模型训练失败，使用默认参数")
            return ARIMA(series, order=(1, 1, 1)).fit()
            
    except Exception as e:
        print_error(f"  {series_name} ARIMA训练失败: {e}")
        return ARIMA(series, order=(1, 1, 1)).fit()

    # ==================== 周期性检测方法 ====================

    def _detect_seasonality(self, series, series_name):
        """检测季节性"""
        try:
            print(f"检测 {series_name} 的季节性...")
            
            # 1. 自相关分析检测周期性
            lags = min(50, len(series) // 4)
            acf_values = pd.Series(series).autocorr(lag=1)
            
            # 检测主要周期
            periods = []
            for period in [7, 14, 30, 90, 365]:  # 周、双周、月、季度、年
                if len(series) >= period * 2:
                    seasonal_score = self._calculate_seasonal_score(series, period)
                    if seasonal_score > 0.3:  # 阈值可调整
                        periods.append(period)
                        print_info(f"  检测到 {period} 天周期，强度: {seasonal_score:.4f}")
            
            # 2. 季节性分解
            if len(series) >= 30:
                try:
                    decomposition = seasonal_decompose(series, period=7, extrapolate_trend='freq')
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(series)
                    
                    self.seasonal_periods[series_name] = periods
                    self.seasonal_strength[series_name] = seasonal_strength
                    self.seasonal_patterns[series_name] = decomposition.seasonal
                    
                    print_info(f"  {series_name} 季节性强度: {seasonal_strength:.4f}")
                    
                except Exception as e:
                    print_warning(f"季节性分解失败: {e}")
                    self.seasonal_periods[series_name] = [7]  # 默认周周期
                    self.seasonal_strength[series_name] = 0.1
            else:
                self.seasonal_periods[series_name] = [7]
                self.seasonal_strength[series_name] = 0.1
                
        except Exception as e:
            print_warning(f"季节性检测失败: {e}")
            self.seasonal_periods[series_name] = [7]
            self.seasonal_strength[series_name] = 0.1

    def _calculate_seasonal_score(self, series, period):
        """计算季节性得分"""
        try:
            if len(series) < period * 2:
                return 0
            
            # 计算不同周期段之间的相关性
            segments = []
            for i in range(0, len(series) - period, period):
                segment = series.iloc[i:i+period]
                if len(segment) == period:
                    segments.append(segment.values)
            
            if len(segments) < 2:
                return 0
            
            # 计算段间相关性
            correlations = []
            for i in range(len(segments)):
                for j in range(i+1, len(segments)):
                    corr = np.corrcoef(segments[i], segments[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                return np.mean(correlations)
            return 0
            
        except Exception:
            return 0

    def _apply_seasonal_adjustment(self, forecast, seasonal_pattern):
        """应用季节性调整"""
        try:
            pattern_length = len(seasonal_pattern)
            if pattern_length == 0:
                return forecast
            
            # 获取季节性模式的均值
            seasonal_mean = seasonal_pattern.mean()
            
            # 应用季节性调整
            adjusted_forecast = forecast.copy()
            for i in range(len(forecast)):
                pattern_index = i % pattern_length
                seasonal_factor = seasonal_pattern.iloc[pattern_index] - seasonal_mean
                adjusted_forecast.iloc[i] += seasonal_factor * 0.3  # 调整强度
            
            return adjusted_forecast
            
        except Exception as e:
            print_warning(f"季节性调整失败: {e}")
            return forecast
