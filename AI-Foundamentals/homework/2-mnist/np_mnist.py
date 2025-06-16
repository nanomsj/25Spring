import numpy as np
import torch
from tqdm import tqdm

#import os
#print(os.getcwd())

# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy')  # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy')  # (60000, )
y_train = np.eye(10)[y_train]  # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy')  # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy')  # (10000,)
y_val = np.eye(10)[y_val]  # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy')  # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy')  # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码

# 定义激活函数
def relu(x):
    '''
    relu函数，i.e.f(x)=max(0,x)
    '''
    return np.maximum(0, x)

def relu_prime(x):
    '''
    relu函数的导数
    '''
    # return 1 if x>0 else 0
    # 使用np包如下：
    return np.where(x > 0, 1, 0)

# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid函数的导数
def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

# tanh函数
def tanh(x):
    return np.tanh(x)

# tanh函数的导数
def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

#输出层激活函数
def softmax(x):
    '''
    softmax函数, 防止除0
    '''
    x_stab = np.exp(x - np.max(x, axis=1, keepdims=True)) # 减去最大值，避免数据溢出
    return x_stab / np.sum(x_stab, axis=1, keepdims=True)

def softmax_prime(x):
    '''
    softmax函数的导数
    '''
    return softmax(x) * (1.0-softmax(x))

# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出，防0
    '''
    return -np.sum(y_true * np.log(y_pred + 1e-8), axis=-1)

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return y_pred - y_true


# 定义权重初始化函数
def init_weights(shape):
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0 / shape[0]), size=shape)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        '''
        初始化网络结构
        两层全连接神经网络
        input_size=784, hidden_size=256, output_size=10, lr=0.01
        '''
        self.W1 = init_weights((input_size, hidden_size)) # 输入层到隐藏层的权重矩阵
        self.b1 = np.zeros(hidden_size)  # 输入层到隐藏层的偏置
        self.W2 = init_weights((hidden_size, output_size)) # 隐藏层到输出层的权重矩阵
        self.b2 = np.zeros(output_size)  # 隐藏层到输出层的偏置
        self.lr = lr  # 学习率

    def forward(self, x):
        '''
        前向传播
        '''
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''

        # 前向传播
        y_pred = self.forward(x_batch)

        # 计算损失和准确率
        loss = np.mean(loss_fn(y_batch, y_pred))
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

        # 反向传播
        dz2 = loss_fn_prime(y_batch, y_pred)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        dz1 = np.dot(dz2, self.W2.T) * relu_prime(self.a1)
        dW1 = np.dot(x_batch.T, dz1)
        db1 = np.sum(dz1, axis=0)

        # 更新权重
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return loss, accuracy

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.01)
    for epoch in range(10):
        losses = []
        accuracies = []

        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            x_batch = X_train[i:i + 64]
            y_batch = y_train[i:i + 64]
            loss, accuracy = net.step(x_batch, y_batch)

            losses.append(loss)
            accuracies.append(accuracy)

        # 验证网络
        y_pred=net.forward(X_val)
        val_loss=np.mean(loss_fn(y_val, y_pred))
        val_accurary=np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y_val, axis=-1))
        print(f"epoch: {epoch + 1}, val_loss: {val_loss:.4f}, val_accuracy: {val_accurary:.4f}")