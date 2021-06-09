import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, num_actions, input_size, hidden_size):
        super(DQN, self).__init__()  # 子类DQN继承了父类nn.Module的所有属性和方法，并用父类方法进行初始化
        self.num_actions = num_actions
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 在CNN中卷积或者池化之后需要连接全连接层，所以需要把多维度的tensor展平成一维
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 最后输出动作为二维tensor
        return x