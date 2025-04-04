import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, type):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size
        self.type = type
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        if self.type in [3, 4]:  # 如果模型类型为3或4，添加VLSTM的多尺度LSTM
            self.levels = [1, 5, 20, 100]  # 不同的频率
            self.lstm_layers = nn.ModuleList([nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True) for _ in self.levels])

        if self.type in [2, 4]:  # 如果模型类型为2或4，添加注意力机制
            # input_size = self.hidden_size if self.type == 2 else self.hidden_size * len(self.levels)
            input_size = self.hidden_size if self.type == 2 else self.hidden_size * len(self.levels)
            self.attention = nn.Linear(input_size, self.hidden_size)

        # 根据模型类型调整全连接层的输入大小
        if self.type in [1, 2]:
            self.linear = nn.Linear(self.hidden_size, self.output_size)
        elif self.type in [3, 4]:
            self.linear = nn.Linear(self.hidden_size * len(self.levels), self.output_size)

            

    def forward(self, input_seq):
        if self.type == 1:  # Vanilla LSTM
            batch_size, seq_len = input_seq.size(0), input_seq.size(1)
            h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
            output, _ = self.lstm(input_seq, (h_0, c_0))
            pred = self.linear(output[:, -1, :])
        
        elif self.type == 2:  # LSTM with Attention
            batch_size, seq_len = input_seq.size(0), input_seq.size(1)
            h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
            output, _ = self.lstm(input_seq, (h_0, c_0))
            attention = torch.tanh(self.attention(output))
            attention_weights = torch.softmax(attention, dim=1)
            context_vector = torch.sum(attention_weights * output, dim=1)
            pred = self.linear(context_vector)
        
        elif self.type == 3:  # VLSTM
            outputs = []
            for i, level in enumerate(self.levels):
                sampled_input = input_seq[:, ::level, :]
                lstm_out, _ = self.lstm_layers[i](sampled_input)
                # 确保输出的时间步长与原始输入一致
                lstm_out = lstm_out.repeat_interleave(level, dim=1)
                # 如果时间步长不匹配，进行填充
                if lstm_out.size(1) < input_seq.size(1):
                    pad_size = input_seq.size(1) - lstm_out.size(1)
                    pad = torch.zeros(batch_size, pad_size, self.hidden_size).to(device)
                    lstm_out = torch.cat((lstm_out, pad), dim=1)
                outputs.append(lstm_out)
            # 确保所有输出的时间步长一致
            max_seq_len = input_seq.size(1)
            outputs = [output[:, :max_seq_len, :] for output in outputs]
            concatenated_output = torch.cat(outputs, dim=-1)
            pred = self.linear(concatenated_output[:, -1, :])

        elif self.type == 4:  # VLSTM with Attention (优化后的实现)
            outputs = []
            for i, level in enumerate(self.levels):
                sampled_input = input_seq[:, ::level, :]
                lstm_out, _ = self.lstm_layers[i](sampled_input)
                lstm_out = lstm_out.repeat_interleave(level, dim=1)
                if lstm_out.size(1) < input_seq.size(1):
                    pad_size = input_seq.size(1) - lstm_out.size(1)
                    pad = torch.zeros(batch_size, pad_size, self.hidden_size).to(input_seq.device)
                    lstm_out = torch.cat((lstm_out, pad), dim=1)
                outputs.append(lstm_out)
            
            max_seq_len = input_seq.size(1)
            outputs = [output[:, :max_seq_len, :] for output in outputs]
            concatenated_output = torch.cat(outputs, dim=-1)

            # 优化后的注意力机制
            attention_scores = self.attention(concatenated_output)  # [batch_size, seq_len, hidden_size]
            attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_len, hidden_size]
            attention_weights = attention_weights.unsqueeze(-1).expand(-1, -1, -1, 4)
            attention_weights = attention_weights.reshape(attention_weights.shape[0], attention_weights.shape[1], -1)
            context_vector = torch.sum(attention_weights * concatenated_output, dim=1)  # [batch_size, hidden_size * len(levels)]
            pred = self.linear(context_vector)

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
# scaler = MinMaxScaler()
# prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
scaler = StandardScaler()
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# 计算分割点
split_point = int(len(prices_scaled) * 0.75)

# 分割数据
train_prices = prices_scaled[:split_point]
test_prices = prices_scaled[split_point:]

# 定义超参数
seq_lengths = [2, 3]  # 不同的序列长度
# seq_lengths = [3, 6, 15, 30, 60]  # 不同的序列长度
input_size = 1  # 输入特征的维度
hidden_size = 128
num_layers = 3
output_size = 1
batch_size = 128
num_epochs = 50
learning_rate = 1e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义模型类型
model_types = [2, 4]  # 1: Vanilla LSTM, 2: LSTM with Attention, 3: VLSTM
# model_types = [1, 2, 3, 4]  # 1: Vanilla LSTM, 2: LSTM with Attention, 3: VLSTM

# 训练和评估每个模型类型
for model_type in model_types:
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
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size, model_type).to(device)
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
                print(f'Model Type: {model_type}, Seq Length: {seq_lengths[idx]}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

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
        print(f'Model Type: {model_type}, Seq Length: {seq_length}, Test Loss: {test_loss:.4f}')
    
    
    model_name = 'LSTM' if model_type == 1 else 'LSTM+Attention' if model_type == 2 else'VLSTM' if model_type == 3 else 'VLSTM+Attention'

# ------------------------------------ 以模型名称画图 -----------------------------
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    for seq_length, loss in zip(seq_lengths, losses):
        plt.plot(range(1, num_epochs + 1), loss, marker='o', linestyle='-', label=f'Sequence Length: {seq_length}')
    plt.title(f'{model_name}-Training Loss Over Epochs for Different Sequence Lengths')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'./img/{model_name}_training_loss.png')
    plt.show()

    # 绘制测试损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, test_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Sequence Length')
    plt.ylabel('Test Loss')
    plt.title(f'{model_name}-Test Loss for Different Sequence Lengths')
    plt.grid(True)

    plt.savefig(f'./img/{model_name}_test_loss.png')
    plt.show()
