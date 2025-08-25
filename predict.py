import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
import platform

# 解决中文乱码
if platform.system() == 'Darwin':  # macOS系统
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取数据（使用脚本所在目录为基准，避免相对路径导致的找不到文件问题）
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / 'user_balance_table.csv'
df = pd.read_csv(data_path, encoding='utf-8')

# 转换 report_date 为日期格式
df['report_date'] = pd.to_datetime(df['report_date'], format='%Y%m%d')

# 截取 2014-03 到 2014-08 的数据
mask = (df['report_date'] >= '2014-03-01') & (df['report_date'] <= '2014-08-31')
df_period = df.loc[mask]

# 按 report_date 聚合申购和赎回金额
result = df_period.groupby('report_date')[['total_purchase_amt', 'total_redeem_amt']].sum().reset_index()

# 可选：打印结果
print(result)

# 可选：绘制趋势图
plt.figure(figsize=(12, 6))
plt.plot(result['report_date'], result['total_purchase_amt'], label='申购金额')
plt.plot(result['report_date'], result['total_redeem_amt'], label='赎回金额')
plt.xlabel('日期')
plt.ylabel('金额')
plt.title('2014-03到2014-08每日申购与赎回金额趋势')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ADF检验
adf_result = adfuller(result['total_purchase_amt'])
print('ADF检验结果：')
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'   {key}: {value}')

# 对赎回金额进行ADF检验
redeem_series = df_period.groupby('report_date')['total_redeem_amt'].sum()
adf_result = adfuller(redeem_series)
print('ADF检验结果：')
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'   {key}: {value}')

# 申购金额ARIMA建模（平稳序列，d=0）
purchase_series = result['total_purchase_amt']
model_purchase = ARIMA(purchase_series, order=(7,0,7))  # 可根据实际情况调整p,q
fit_purchase = model_purchase.fit()
forecast_purchase = fit_purchase.forecast(steps=30)

# 赎回金额ARIMA建模（非平稳序列，d=1）
redeem_series = result['total_redeem_amt']
model_redeem = ARIMA(redeem_series, order=(7,1,7))  # 可根据实际情况调整p,q
fit_redeem = model_redeem.fit()
forecast_redeem = fit_redeem.forecast(steps=30)

# 生成未来30天日期（2014-09-01 ~ 2014-09-30）
future_dates = pd.date_range('2014-09-01', periods=30, freq='D')

# 可视化
plt.figure(figsize=(16, 7))
# 历史数据（使用日期作为横轴，确保与预测日期对齐）
plt.plot(result['report_date'], result['total_purchase_amt'], label='申购金额-历史', color='blue')
plt.plot(result['report_date'], result['total_redeem_amt'], label='赎回金额-历史', color='orange')
# 预测数据
plt.plot(future_dates, forecast_purchase, label='申购金额-预测', color='blue', linestyle='--')
plt.plot(future_dates, forecast_redeem, label='赎回金额-预测', color='orange', linestyle='--')

plt.xlabel('日期')
plt.ylabel('金额')
plt.title('2014-03到2014-09每日申购与赎回金额趋势（含预测）')
plt.legend()
plt.xticks(rotation=45)
plt.xlim([pd.to_datetime('2014-03-01'), pd.to_datetime('2014-09-30')])  # 限定横坐标范围
plt.tight_layout()
plt.show()

# 输出到csv
output = pd.DataFrame({
    'report_date': future_dates.strftime('%Y%m%d'),
    'purchase': forecast_purchase.values,
    'redeem': forecast_redeem.values
})

# 输出到csv（保存到脚本所在目录）
output_path = BASE_DIR / 'arima_forecast_201409.csv'
output.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f'预测结果已保存到 {output_path.name}')