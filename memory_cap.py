import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义LSTM模型
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
            h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
            c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
            output, _ = self.lstm(input_seq, (h_0, c_0))
            pred = self.linear(output[:, -1, :])
        
        elif self.type == 2:  # LSTM with Attention
            batch_size, seq_len = input_seq.size(0), input_seq.size(1)
            h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
            c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
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
                lstm_out = lstm_out.repeat_interleave(level, dim=1)
                if lstm_out.size(1) < input_seq.size(1):
                    pad_size = input_seq.size(1) - lstm_out.size(1)
                    pad = torch.zeros(batch_size, pad_size, self.hidden_size).to(input_seq.device)
                    lstm_out = torch.cat((lstm_out, pad), dim=1)
                outputs.append(lstm_out)
            max_seq_len = input_seq.size(1)
            outputs = [output[:, :max_seq_len, :] for output in outputs]
            concatenated_output = torch.cat(outputs, dim=-1)
            pred = self.linear(concatenated_output[:, -1, :])

        elif self.type == 4:  # VLSTM with Attention
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

            attention_scores = self.attention(concatenated_output)
            attention_weights = torch.softmax(attention_scores, dim=1)
            context_vector = torch.sum(attention_weights * concatenated_output, dim=1)
            pred = self.linear(context_vector)

        return pred

# 生成包含重复模式的长序列数据
def generate_repeating_pattern_data(batch_size, seq_len, input_size, pattern_length):
    x = torch.randn(batch_size, pattern_length, input_size)
    x = x.repeat(1, seq_len // pattern_length + 1, 1)[:, :seq_len, :]
    y = (x[:, -1, :] > 0).float()
    return x, y

# 训练模型
def train_model(model, criterion, optimizer, num_epochs, batch_size, seq_len, input_size, pattern_length):
    start_time = time.time()
    for epoch in range(num_epochs):
        x, y = generate_repeating_pattern_data(batch_size, seq_len, input_size, pattern_length)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    training_time = end_time - start_time
    return training_time

# 测试模型
def test_model(model, batch_size, seq_len, input_size, pattern_length):
    x, y = generate_repeating_pattern_data(batch_size, seq_len, input_size, pattern_length)
    outputs = model(x)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y).sum().item() / batch_size
    return accuracy

# 参数设置
input_size = 10
hidden_size = 20
output_size = 2
batch_size = 32
seq_len = 1000
num_epochs = 10
learning_rate = 0.001
pattern_length = 100  # 重复模式的长度

# 创建模型、损失函数和优化器
models = []
for model_type in [1, 2, 3, 4]:
    model = LSTM(input_size, hidden_size, 2, output_size, batch_size, model_type)
    models.append(model)
criterion = nn.CrossEntropyLoss()

# 训练模型并记录时间
training_times = []
accuracies = []
model_types = [1, 2, 3, 4]
for i, model in enumerate(models):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_time = train_model(model, criterion, optimizer, num_epochs, batch_size, seq_len, input_size, pattern_length)
    training_times.append(training_time)
    accuracy = test_model(model, batch_size, seq_len, input_size, pattern_length)
    accuracies.append(accuracy)
    print(f"Model Type {model_types[i]} Training Time: {training_time:.4f} seconds, Accuracy: {accuracy:.4f}")

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar([f"Type {t}" for t in model_types], training_times)
plt.title("Training Time Comparison")
plt.xlabel("Model Type")
plt.ylabel("Training Time (seconds)")

plt.subplot(1, 2, 2)
plt.bar([f"Type {t}" for t in model_types], accuracies)
plt.title("Accuracy Comparison")
plt.xlabel("Model Type")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()