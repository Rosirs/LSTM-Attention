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

# 定义超参数
seq_lengths = [ 3, 6 ]  # 使用前三天的价格预测后一天
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
batch_size = 64
num_epochs = 50
learning_rate = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = []
criterions = []
optimizers = []
train_dataloaders = []
test_dataloaders = []

for seq_length in seq_lengths:
    # 创建训练集和测试集
    train_dataset = PriceDataset(train_prices, seq_length)
    test_dataset = PriceDataset(test_prices, seq_length)
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 将数据加载器添加到列表中
    train_dataloaders.append(train_dataloader)
    test_dataloaders.append(test_dataloader)

for train_dataloader, test_dataloader in zip(train_dataloaders, test_dataloaders):
    # 初始化模型、损失函数和优化器
    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    models.append(model)
    criterions.append(criterion)
    optimizers.append(optimizer)

losses = [[] for _ in range(len(seq_lengths))]
# 训练模型
for idx, (model, criterion, optimizer, train_dataloader) in enumerate(zip(models, criterions, optimizers, train_dataloaders)):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0  # 初始化当前 epoch 的总损失
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # 累加当前批次的损失值
        epoch_loss /= len(train_dataloader)  # 计算当前 epoch 的平均损失
        losses[idx].append(epoch_loss)  # 记录损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
for seq_length, loss in zip(seq_lengths, losses):
    plt.plot(range(1, num_epochs + 1), loss, marker='o', linestyle='-', label=f'Sequence Length: {seq_length}')
plt.title('Training Loss Over Epochs for Different Sequence Lengths')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 评估模型并记录测试损失
test_losses = []
for model, criterion, test_dataloader, seq_length in zip(models, criterions, test_dataloaders, seq_lengths):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    test_loss /= len(test_dataloader)
    test_losses.append(test_loss)
    print(f'Seq Length: {seq_length}, Test Loss: {test_loss:.4f}')

# 绘制测试损失曲线
plt.figure(figsize=(10, 6))
plt.plot(seq_lengths, test_losses, marker='o', linestyle='-', color='b')
plt.xlabel('Sequence Length')
plt.ylabel('Test Loss')
plt.title('Test Loss for Different Sequence Lengths')
plt.grid(True)
plt.show()