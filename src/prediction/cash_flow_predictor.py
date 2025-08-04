# -*- coding: utf-8 -*-
"""
资金流预测模块
基于时间序列数据进行资金流预测
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.data_processor import DataProcessor
from utils.interactive_utils import print_header, print_success, print_error, print_info
from utils.config_utils import get_field_name

class CashFlowPredictor(DataProcessor):
    """资金流预测类"""
    
    def __init__(self):
        """初始化资金流预测器"""
        super().__init__()
        self.model = None
        self.feature_columns = []
        self.target_column = 'Net_Flow'
        self.scaler = None
        
    def prepare_features(self):
        """
        准备预测特征
        
        Returns:
            bool: 是否准备成功
        """
        if self.data is None:
            print_error("请先加载数据")
            return False
            
        # 确保数据已预处理
        if 'Net_Flow' not in self.data.columns:
            if not self.preprocess_data():
                return False
        
        print_info("准备预测特征...")
        
        try:
            # 基础特征
            self.feature_columns = [
                'Purchase_Amount', 'Redemption_Amount', 'Balance_Change',
                'Year', 'Month', 'Day', 'Weekday', 'Quarter'
            ]
            
            # 添加滞后特征
            for lag in [1, 3, 7]:
                self.data[f'Net_Flow_lag_{lag}'] = self.data['Net_Flow'].shift(lag)
                self.data[f'Purchase_Amount_lag_{lag}'] = self.data['Purchase_Amount'].shift(lag)
                self.data[f'Redemption_Amount_lag_{lag}'] = self.data['Redemption_Amount'].shift(lag)
                self.feature_columns.extend([f'Net_Flow_lag_{lag}', f'Purchase_Amount_lag_{lag}', f'Redemption_Amount_lag_{lag}'])
            
            # 添加移动平均特征
            for window in [3, 7, 14]:
                self.data[f'Net_Flow_ma_{window}'] = self.data['Net_Flow'].rolling(window=window).mean()
                self.data[f'Purchase_Amount_ma_{window}'] = self.data['Purchase_Amount'].rolling(window=window).mean()
                self.data[f'Redemption_Amount_ma_{window}'] = self.data['Redemption_Amount'].rolling(window=window).mean()
                self.feature_columns.extend([f'Net_Flow_ma_{window}', f'Purchase_Amount_ma_{window}', f'Redemption_Amount_ma_{window}'])
            
            # 移除包含NaN的行
            self.data = self.data.dropna()
            
            print_success(f"特征准备完成，共{len(self.feature_columns)}个特征")
            return True
            
        except Exception as e:
            print_error(f"特征准备失败: {e}")
            return False
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        训练预测模型
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            bool: 是否训练成功
        """
        if not self.prepare_features():
            return False
        
        print_info("开始训练预测模型...")
        
        try:
            # 准备数据
            X = self.data[self.feature_columns]
            y = self.data[self.target_column]
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # 这里可以添加具体的模型训练代码
            # 目前只是占位符
            print_info("模型训练功能正在开发中...")
            
            # 保存训练结果
            self.training_results = {
                "训练集大小": len(X_train),
                "测试集大小": len(X_test),
                "特征数量": len(self.feature_columns),
                "特征列表": self.feature_columns
            }
            
            print_success("模型训练完成")
            return True
            
        except Exception as e:
            print_error(f"模型训练失败: {e}")
            return False
    
    def predict(self, days_ahead=7):
        """
        进行预测
        
        Args:
            days_ahead: 预测未来天数
            
        Returns:
            pd.DataFrame: 预测结果
        """
        if self.model is None:
            print_error("请先训练模型")
            return None
        
        print_info(f"预测未来{days_ahead}天的资金流...")
        
        try:
            # 这里可以添加具体的预测代码
            # 目前只是占位符
            print_info("预测功能正在开发中...")
            
            # 返回空的预测结果
            future_dates = pd.date_range(
                start=self.data[get_field_name("时间字段")].max() + timedelta(days=1),
                periods=days_ahead,
                freq='D'
            )
            
            predictions = pd.DataFrame({
                'date': future_dates,
                'predicted_net_flow': [0] * days_ahead,
                'predicted_purchase': [0] * days_ahead,
                'predicted_redemption': [0] * days_ahead
            })
            
            print_success("预测完成")
            return predictions
            
        except Exception as e:
            print_error(f"预测失败: {e}")
            return None
    
    def evaluate_model(self):
        """
        评估模型性能
        
        Returns:
            dict: 评估结果
        """
        if self.model is None:
            print_error("请先训练模型")
            return None
        
        print_info("评估模型性能...")
        
        try:
            # 这里可以添加具体的模型评估代码
            # 目前只是占位符
            evaluation_results = {
                "MSE": 0.0,
                "MAE": 0.0,
                "R2": 0.0,
                "评估时间": datetime.now().isoformat()
            }
            
            print_success("模型评估完成")
            return evaluation_results
            
        except Exception as e:
            print_error(f"模型评估失败: {e}")
            return None

def run_prediction_analysis():
    """
    运行资金流预测分析
    """
    print_header("资金流预测", "模型训练和预测")
    
    # 创建预测器实例
    predictor = CashFlowPredictor()
    
    if predictor.load_data():
        if predictor.preprocess_data():
            if predictor.train_model():
                # 进行预测
                predictions = predictor.predict(days_ahead=7)
                if predictions is not None:
                    print("\n=== 预测结果 ===")
                    print(predictions)
                
                # 评估模型
                evaluation = predictor.evaluate_model()
                if evaluation:
                    print("\n=== 模型评估 ===")
                    for metric, value in evaluation.items():
                        print(f"{metric}: {value}")
                
                print_success("预测分析完成")
            else:
                print_error("模型训练失败")
        else:
            print_error("数据预处理失败")
    else:
        print_error("数据加载失败")

if __name__ == "__main__":
    run_prediction_analysis() 