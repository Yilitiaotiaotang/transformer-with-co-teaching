import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Loss functions  #TODO:核心代码：核心代码为梯度互相更新的部分, 相当于先对每个网络各求了loss，再对loss进行排序，选取最低的一定比例交给另一个网络进行反向传播。
def loss_coteaching(outputs_1, outputs_2, loss_1, loss_2, t, forget_rate):
    ind_1_sorted = np.argsort(loss_1.data.cpu()).cuda() # TODO .cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    ind_2_sorted = np.argsort(loss_2.data.cpu()).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate # TODO remember = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    criteria = torch.nn.NLLLoss(reduction='none')
    loss_1_update = criteria(outputs_1[ind_2_update], t[ind_2_update])
    loss_2_update = criteria(outputs_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember
