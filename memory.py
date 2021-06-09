from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))    # 具名元组


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # save a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)    # 每次调用只加一项，故只需保证有一个位置/空位留出即可
        self.memory[self.position] = Transition(*args)  # *将list/tuple中的元素拆分
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):  # 要让类中的len()函数工作正常，类必须提供一个特殊方法__len__()，它返回元素的个数。
        return len(self.memory)
