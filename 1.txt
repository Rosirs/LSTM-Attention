import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 定义模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.attention = nn.Linear(self.hidden_size, self.hidden_size)

        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        #--------------------vanilla LSTM----------------------
        batch_size, seq_len = input_seq.size(0), input_seq.size(1)
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))  # output(5, 30, 64)
        pred = self.linear(output)
        pred = pred[:, -1, :]
        #-------------------------------------------------------------

        # #--------------------attention mechanism----------------------
        # batch_size, seq_len = input_seq.size(0), input_seq.size(1)
        # h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # # output(batch_size, seq_len, num_directions * hidden_size)
        # output, _ = self.lstm(input_seq, (h_0, c_0))  # output(5, 30, 64)
        
        # # Attention mechanism
        # # output shape: (batch_size, seq_len, hidden_size)
        # # attention shape: (batch_size, seq_len, hidden_size)
        # attention = torch.tanh(self.attention(output))
        # # attention_weights shape: (batch_size, seq_len, 1)
        # attention_weights = torch.softmax(attention, dim=1)
        # # context_vector shape: (batch_size, hidden_size)
        # context_vector = torch.sum(attention_weights * output, dim=1)
        
        # pred = self.linear(context_vector)
        # #-------------------------------------------------------------

        return pred
    
# 定义数据集类
class PriceDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    

# 读取数据
data_excel = ".\\data\\daily_prices.xlsx"
df = pd.read_excel(data_excel)
df = df.iloc[:, :2].copy()
df.columns = ['Date', 'USD']
df.dropna(inplace=True)

# 将日期列转换为 datetime 类型
df['Date'] = pd.to_datetime(df['Date'])

# 提取价格列
prices = df['USD'].values

# 数据标准化
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# 计算分割点
split_point = int(len(prices_scaled) * 0.75)

# 分割数据
train_prices = prices_scaled[:split_point]
test_prices = prices_scaled[split_point:]

