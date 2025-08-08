# -*- coding: utf-8 -*-
"""
ARIMA配置管理器
提供ARIMA模型参数的查看和修改功能
"""

import json
from pathlib import Path
from utils.interactive_utils import print_header, print_success, print_error, print_info, print_warning, show_menu, wait_for_key
from config import ARIMA_TRAINING_CONFIG

class ARIMAConfigManager:
    """ARIMA配置管理器"""
    
    def __init__(self):
        """初始化配置管理器"""
        self.config = ARIMA_TRAINING_CONFIG
    
    def show_current_config(self):
        """显示当前ARIMA配置"""
        print_header("当前ARIMA配置")
        
        # 显示模型参数
        model_params = self.config["模型参数"]["ARIMA参数"]
        print("📊 ARIMA模型参数:")
        print(f"  p范围: {model_params['p_range']}")
        print(f"  d范围: {model_params['d_range']}")
        print(f"  q范围: {model_params['q_range']}")
        
        # 显示预测配置
        pred_config = self.config["预测配置"]
        print("\n🎯 预测配置:")
        print(f"  预测步数: {pred_config['预测步数']} 天")
        print(f"  置信区间: {pred_config['置信区间']}")
        print(f"  预测频率: {pred_config['预测频率']}")
        print(f"  净资金流噪声比例: {pred_config['噪声比例']}")
        print(f"  申购金额噪声比例: {pred_config['申购噪声比例']}")
        print(f"  赎回金额噪声比例: {pred_config['赎回噪声比例']}")
        
        # 显示数据配置
        data_config = self.config["数据极限"]
        print("\n📈 数据配置:")
        print(f"  最小数据量: {data_config['最小数据量']}")
        print(f"  最大数据量: {data_config['最大数据量']}")
        print(f"  数据采样: {'启用' if data_config['数据采样']['启用采样'] else '禁用'}")
        if data_config['数据采样']['启用采样']:
            print(f"  采样比例: {data_config['数据采样']['采样比例']}")
    
    def modify_model_params(self):
        """修改模型参数"""
        print_header("修改ARIMA模型参数")
        
        model_params = self.config["模型参数"]["ARIMA参数"]
        
        print("当前参数范围:")
        print(f"  p范围: {model_params['p_range']}")
        print(f"  d范围: {model_params['d_range']}")
        print(f"  q范围: {model_params['q_range']}")
        
        try:
            # 修改p范围
            print("\n请输入新的p范围（用逗号分隔，如: 0,1,2,3）:")
            p_input = input("p范围: ").strip()
            if p_input:
                p_range = [int(x.strip()) for x in p_input.split(',')]
                model_params['p_range'] = p_range
                print_success(f"p范围已更新为: {p_range}")
            
            # 修改d范围
            print("\n请输入新的d范围（用逗号分隔，如: 0,1,2）:")
            d_input = input("d范围: ").strip()
            if d_input:
                d_range = [int(x.strip()) for x in d_input.split(',')]
                model_params['d_range'] = d_range
                print_success(f"d范围已更新为: {d_range}")
            
            # 修改q范围
            print("\n请输入新的q范围（用逗号分隔，如: 0,1,2,3）:")
            q_input = input("q范围: ").strip()
            if q_input:
                q_range = [int(x.strip()) for x in q_input.split(',')]
                model_params['q_range'] = q_range
                print_success(f"q范围已更新为: {q_range}")
            
            return True
            
        except Exception as e:
            print_error(f"参数修改失败: {e}")
            return False
    
    def modify_prediction_config(self):
        """修改预测配置"""
        print_header("修改预测配置")
        
        pred_config = self.config["预测配置"]
        
        print("当前预测配置:")
        print(f"  预测步数: {pred_config['预测步数']} 天")
        print(f"  净资金流噪声比例: {pred_config['噪声比例']}")
        print(f"  申购金额噪声比例: {pred_config['申购噪声比例']}")
        print(f"  赎回金额噪声比例: {pred_config['赎回噪声比例']}")
        
        try:
            # 修改预测步数
            print("\n请输入新的预测步数:")
            steps_input = input("预测步数: ").strip()
            if steps_input:
                steps = int(steps_input)
                pred_config['预测步数'] = steps
                print_success(f"预测步数已更新为: {steps}")
            
            # 修改噪声比例
            print("\n请输入新的噪声比例（0-1之间的小数）:")
            print("净资金流噪声比例:")
            noise_input = input("噪声比例: ").strip()
            if noise_input:
                noise_ratio = float(noise_input)
                pred_config['噪声比例'] = noise_ratio
                print_success(f"净资金流噪声比例已更新为: {noise_ratio}")
            
            print("申购金额噪声比例:")
            purchase_noise_input = input("噪声比例: ").strip()
            if purchase_noise_input:
                purchase_noise_ratio = float(purchase_noise_input)
                pred_config['申购噪声比例'] = purchase_noise_ratio
                print_success(f"申购金额噪声比例已更新为: {purchase_noise_ratio}")
            
            print("赎回金额噪声比例:")
            redemption_noise_input = input("噪声比例: ").strip()
            if redemption_noise_input:
                redemption_noise_ratio = float(redemption_noise_input)
                pred_config['赎回噪声比例'] = redemption_noise_ratio
                print_success(f"赎回金额噪声比例已更新为: {redemption_noise_ratio}")
            
            return True
            
        except Exception as e:
            print_error(f"配置修改失败: {e}")
            return False
    
    def save_config(self):
        """保存配置到文件"""
        try:
            config_file = Path("config/arima_config_backup.json")
            config_file.parent.mkdir(exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            print_success(f"配置已保存到: {config_file}")
            return True
            
        except Exception as e:
            print_error(f"配置保存失败: {e}")
            return False
    
    def load_config(self):
        """从文件加载配置"""
        try:
            config_file = Path("config/arima_config_backup.json")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print_success(f"配置已从文件加载: {config_file}")
                return True
            else:
                print_warning("未找到配置文件，使用默认配置")
                return False
                
        except Exception as e:
            print_error(f"配置加载失败: {e}")
            return False
    
    def reset_config(self):
        """重置为默认配置"""
        from config import ARIMA_TRAINING_CONFIG
        self.config = ARIMA_TRAINING_CONFIG
        print_success("配置已重置为默认值")
        return True
    
    def show_menu(self):
        """显示配置管理菜单"""
        menu_items = [
            {
                "name": "查看当前配置",
                "description": "显示当前ARIMA模型配置",
                "action": self.show_current_config
            },
            {
                "name": "修改模型参数",
                "description": "修改ARIMA模型的p、d、q参数范围",
                "action": self.modify_model_params
            },
            {
                "name": "修改预测配置",
                "description": "修改预测步数和噪声比例",
                "action": self.modify_prediction_config
            },
            {
                "name": "保存配置",
                "description": "保存当前配置到文件",
                "action": self.save_config
            },
            {
                "name": "加载配置",
                "description": "从文件加载配置",
                "action": self.load_config
            },
            {
                "name": "重置配置",
                "description": "重置为默认配置",
                "action": self.reset_config
            },
            {
                "name": "返回主菜单",
                "description": "返回主菜单",
                "action": None
            }
        ]
        
        while True:
            selected_action = show_menu(menu_items, "ARIMA配置管理")
            
            if selected_action is None:
                break
            else:
                try:
                    selected_action()
                    wait_for_key("按回车键继续...")
                except Exception as e:
                    print_error(f"操作失败: {e}")
                    wait_for_key("按回车键继续...")


def run_arima_config_manager():
    """运行ARIMA配置管理器"""
    print_header("ARIMA配置管理器", "管理ARIMA模型参数和预测配置")
    
    manager = ARIMAConfigManager()
    manager.show_menu()


if __name__ == "__main__":
    run_arima_config_manager() 