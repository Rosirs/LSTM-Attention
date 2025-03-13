import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Reading the dataset
data_excel = ".\data\daily_prices.xlsx"
df = pd.read_excel(data_excel)

df = df.iloc[:, :2].copy()
df.columns = ['Date', 'USD']

df.dropna(inplace=True)

# Converting the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sorting the data by date
df.sort_values(by='Date', inplace=True)

# 假设 df 是你的数据框，train_size 是训练集的大小

train_size = int(len(df) * 0.75)  # 假设训练集占总数据的 75%

# 设置图像大小和分辨率
plt.figure(figsize=(15, 6), dpi=150)

# 设置绘图风格
plt.rcParams['axes.facecolor'] = 'white'
plt.rc('axes', edgecolor='white')

# 绘制训练集和测试集
plt.plot(df.Date[:train_size], df.USD[:train_size], color='black', lw=1, label='Training set')  # 黑线表示训练集
plt.plot(df.Date[train_size:], df.USD[train_size:], color='blue', lw=1, label='Test set')  # 蓝线表示测试集

# 添加标题和标签
plt.title('Gold Price Training and Test Sets', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)

# 添加图例
plt.legend()

# 添加网格
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.savefig('train_test_split.png')



# # Splitting the dataset into training and testing sets
# train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

# train_size = train_df.shape[0]

# plt.figure(figsize=(15, 6), dpi=150)
# plt.rcParams['axes.facecolor'] = 'white'
# plt.rc('axes',edgecolor='white')
# plt.plot(df.Date[:train_size], df.USD[:train_size], color='black', lw=1)
# plt.plot(df.Date[train_size:], df.USD[train_size:], color='blue', lw=1)
# plt.title('Gold Price Training and Test Sets', fontsize=15)
# plt.xlabel('Date', fontsize=12)
# plt.ylabel('Price', fontsize=12)
# plt.legend(['Training set', 'Test set'], loc='upper center',bbox_to_anchor=(0.5, -0.15), ncol=2, prop={'size': 15})
# plt.grid(color='gray', linestyle='--', linewidth=0.5)

# plt.show()
