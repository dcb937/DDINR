import torch
from einops import rearrange
from typing import Tuple
import numpy as np
import copy
import math
import sys


def create_optim(name, parameters, lr):
    if name == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif name == 'Adamax':
        optimizer = torch.optim.Adamax(parameters, lr=lr)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=lr)
    else:
        raise NotImplemented
    return optimizer


def create_lr_scheduler(optimizer, lr_scheduler_opt):
    lr_scheduler_opt = copy.deepcopy(lr_scheduler_opt)
    lr_scheduler_name = lr_scheduler_opt.pop('name')
    if lr_scheduler_name == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **lr_scheduler_opt)
    elif lr_scheduler_name == 'CyclicLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **lr_scheduler_opt)
    elif lr_scheduler_name == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_opt)
    elif lr_scheduler_name == 'none':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000000000])
    else:
        raise NotImplementedError
    return lr_scheduler

# 扁平化过程：在扁平化过程中，这些三维坐标被转换为一维序列。这通常通过按照某种顺序（例如，先 X 轴、再 Y 轴、
# 最后 Z 轴，或者任何其他顺序）遍历每个点来完成。结果是一个长列表或数组，其中包含了连续的坐标值。
# 这里的shape就是数据的长宽高，eg: 512*512*512


def create_flattened_coords(coords_shape: Tuple) -> torch.Tensor:    # coords的大小是最后叶节点小块的大小
    minimum = -1
    maximum = 1
    coords = torch.stack(torch.meshgrid(
        torch.linspace(minimum, maximum, coords_shape[0]),
        torch.linspace(minimum, maximum, coords_shape[1]),
        torch.linspace(minimum, maximum, coords_shape[2]), indexing='ij'),
        axis=-1)
    flattened_coords = rearrange(coords, 'd h w c -> (d h w) c')
    return flattened_coords

class PointSampler:
    def __init__(self, data: torch.Tensor, max_level: int, batch_size: int, epochs: int, leaf_nodes_num: int, device: str = 'cpu') -> None:
        self.batch_size = int(batch_size / 8**max_level)   # max_level 从0开始，2层就是1
        assert self.batch_size > 512 and self.batch_size < 2097152, "Batch size error"
        self.epochs = epochs
        self.device = device
        self.data = data

        self.pop_size = data.shape[0] / 8**max_level    # 所谓pop_size指的是划分到最后的叶节点的大小
        self.evaled_epochs = []

    def judge_eval(self, eval_epoch):
        if self.epochs_count % eval_epoch == 0 and self.epochs_count != 0 and not (self.epochs_count in self.evaled_epochs):
            self.evaled_epochs.append(self.epochs_count)
            return True
        elif self.index >= self.pop_size and self.epochs_count >= self.epochs-1:
            self.epochs_count = self.epochs
            return True
        else:
            return False


    def __len__(self):
        return self.epochs*math.ceil(self.data.shape[0]/self.batch_size)

    def __iter__(self):
        self.index = 0
        self.epochs_count = 0
        return self

    def __next__(self):              # 每次抽样提供的是划分到最后块的坐标
        if self.index < self.pop_size:
            # sampled_idxs = self.index
            self.index += self.batch_size
            return self.batch_size
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration


if __name__ == "__main__":
    a = torch.randint(0, 100, (10,))
    print(a)
    loss = 1
    loss = loss + a
    print(loss)

    points_array = torch.randint(0, 100, (20,3))
    print(points_array)
    idx = torch.randint(0, 20, (6,))
    print(points_array[idx])
    a = np.random.ranf(0,10)
    print(a.mean())