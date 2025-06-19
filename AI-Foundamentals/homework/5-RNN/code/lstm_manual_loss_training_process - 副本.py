# 用Pytorch手写一个LSTM网络，在IMDB数据集上进行训练
# 跑5个epoch，hidden size保持64不变
import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from utils import load_imdb_dataset, Accuracy
import sys

from utils import load_imdb_dataset, Accuracy
from tqdm import tqdm
import math

# 超参数
n_epoch = 10
batch_size = 32
print_freq = 2


use_mlu = False
# try:
#     import torch_mlu
#     import torch_mlu.core.mlu_model as ct
#     global ct
#     use_mlu = torch.mlu.is_available()
# except:
#     use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

X_train, y_train, X_test, y_test = load_imdb_dataset('data', nb_words=20000, test_split=0.2)

seq_Len = 200
vocab_size = len(X_train) + 1


class ImdbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        data = self.X[index]
        data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype('int32')  # set
        label = self.y[index]
        return data, label

    def __len__(self):

        return len(self.y)


# 你需要实现的手写LSTM内容，包括LSTM类所属的__init__函数和forward函数
class LSTM(nn.Module):
    '''
    手写lstm，可以用全连接层nn.Linear，不能直接用nn.LSTM
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.imput_size = input_size
        
        # 初始化所有门控的参数：权重W和偏置b

        # 输入门
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        
        # 遗忘门
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        
        # 细胞状态
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ic = nn.Parameter(torch.Tensor(hidden_size))
        
        # 输出门
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        # 使用均匀分布初始化权重。常用策略
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, x, init_states=None):
        """
        x: 输入序列, shape: (batch_size, seq_len, input_size)
        返回: output, (h_n, c_n)
        """
        batch_size, seq_len, _ = x.size()
        
        # 初始化隐藏状态和细胞状态
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states
        
        # 存储所有时间步的输出
        outputs = []
        
        for t in range(seq_len):
            # 获取当前时间步的输入
            x_t = x[:, t, :]  # (batch_size, input_size)
            
            # 输入门计算
            i_t = torch.sigmoid(
                torch.matmul(x_t, self.W_ii.t()) + torch.matmul(h_t, self.W_hi.t()) + self.b_ii
            )
            
            # 遗忘门计算
            f_t = torch.sigmoid(
                torch.matmul(x_t, self.W_if.t()) + torch.matmul(h_t, self.W_hf.t()) + self.b_if
            )
            
            # 细胞状态计算
            c_tilde = torch.tanh(
                torch.matmul(x_t, self.W_ic.t()) + torch.matmul(h_t, self.W_hc.t()) + self.b_ic
            )
            
            # 更新细胞状态
            c_t = f_t * c_t + i_t * c_tilde
            
            # 输出门计算
            o_t = torch.sigmoid(
                torch.matmul(x_t, self.W_io.t()) + torch.matmul(h_t, self.W_ho.t()) + self.b_io
            )
            
            # 更新隐藏状态
            h_t = o_t * torch.tanh(c_t) 
            
            # 保存当前时间步的输出
            outputs.append(h_t.unsqueeze(1))
        
        # 拼接所有时间步的输出
        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        
        return outputs, (h_t, c_t)


class Net(nn.Module):
    '''
    一层LSTM的文本分类模型
    '''
    def __init__(self, embedding_size=64, hidden_size=64, num_classes=2):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = LSTM(input_size=embedding_size, hidden_size=hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        '''
        x: 输入, shape: (batch_size, seq_len)
        '''
        # 词嵌入 (batch_size, seq_len) -> (batch_size, seq_len, embedding_size)
        x = self.embedding(x)
        
        # LSTM层 (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出 (batch_size, hidden_size)
        last_out = lstm_out[:, -1, :]
        
        # 全连接层
        x = torch.relu(self.fc1(last_out))
        x = self.fc2(x)
        
        return x


train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = ImdbDataset(X=X_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net = Net()
metric = Accuracy()
print(net)


def train(model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_acc = 0
    train_loss = 0
    n_iter = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        target = target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        metric.update(output, target)
        train_acc += metric.result()
        train_loss += loss.item()
        metric.reset()
        n_iter += 1
    print('Train Epoch: {} Loss: {:.6f} \t Acc: {:.6f}'.format(epoch, train_loss / n_iter, train_acc / n_iter))


def test(model, device, test_loader):
    model = model.to(device)
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    test_loss = 0
    test_acc = 0
    n_iter = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += loss_func(output, target).item()
            metric.update(output, target)
            test_acc += metric.result()
            metric.reset()
            n_iter += 1
    test_loss /= n_iter
    test_acc /= n_iter
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(net, device, train_loader, optimizer, epoch)
    test(net, device, test_loader)
