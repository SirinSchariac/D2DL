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