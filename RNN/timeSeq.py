import torch
import torch.nn as nn
from d2l import torch as d2l

torch.cuda.is_available()

T = 1000
time = torch.arange(1, T+1, dtype=torch.float32)
# 正弦波基础上加噪声
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# 图像横轴为时间，纵轴为x，显示网格
# d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(10, 5))
# 另存图像
# d2l.plt.savefig('time-series.png')

# 设定时间步长为4
tau = 4
# 每个输入特征相应的也为4
# 共有T个数据样本，因此一共可以构造T-tau个样本
# 对于不足tau的时间步长，直接丢弃(因为序列足够长)
features = torch.zeros((T - tau, tau))
for i in range(tau):
    # 第一列：从x[0] 到 x[T-tau]共T-tau个
    # 第二列：从x[1] 到 x[T-tau+1]共T-tau个 以此类推
    features[:, i] = x[i: T - tau + i] 
# 标签就设置为该行相应的未来时间点数据
# 例如对于第一行(x[0], x[1], x[2], x[3])，标签为x[4]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), 
                            batch_size, is_train=True)

# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机MLP
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss(reduction='none')

def train(net, train_iter, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss : {d2l.evaluate_loss(net, train_iter, loss):f}')
        
net = get_net()
train(net, train_iter, 5, 0.01)
