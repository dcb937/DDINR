import torch
from torch import nn
import numpy as np
import os
from einops import rearrange
import sys
import math
from tqdm import tqdm
import torch.nn.functional as F
from utils.Sampler import create_optim, create_flattened_coords, PointSampler, create_lr_scheduler
from utils.Network import MLP
from utils.VTK import get_vtk_size_bytes, sort_in_3D_axies

def normalize_data(data:np.ndarray, scale_min, scale_max):
    dtype = data.dtype      # 获取到的是整个data的数据类型（在numpy中是一致的），故在后续可能遇到的不同维度不同类型的情况这里没有考虑，不过int和float都是4个byte，倒也无所谓
    data = data.astype(np.float32)
    data_min, data_max = data.min(axis=0), data.max(axis=0)
    data = (data - data_min)/(data_max - data_min)
    data = data*(scale_max - scale_min) + scale_min
    data = torch.tensor(data, dtype=torch.float)
    side_info = {'scale_min':scale_min, 'scale_max':scale_max, 'data_min':data_min, 'data_max':data_max, 'dtype':dtype}
    return data, side_info

def invnormalize_data(data:np.ndarray, scale_min, scale_max, data_min, data_max, dtype):
    # 确保 data_min 和 data_max 是一维数组
    data_min = np.array(data_min)
    data_max = np.array(data_max)
    # 标准化数据
    data = (data - scale_min) / (scale_max - scale_min)
    # 反标准化数据
    data = data * (data_max - data_min) + data_min
    # 转换数据类型
    data = data.astype(dtype=dtype)
    return data

def binary_search(tensor, dim, value, find_first):
    """二分查找"""
    low, high = 0, tensor.size(0) - 1
    while low <= high:
        mid = (low + high) // 2
        if (find_first and tensor[mid, dim] >= value) or (not find_first and tensor[mid, dim] > value):
            high = mid - 1
        else:
            low = mid + 1
    return low

class Node():
    def __init__(self, parent, level, points_array, points_value_array, di, hi, wi, device):
        self.level = level
        self.parent = parent
        self.origin_points_array, self.origin_points_value_array = points_array, points_value_array
        self.points_array, self.points_value_array = [], []
        self.device = device

        # 计算每列的最大值
        max_values, _ = torch.max(points_array, dim=0)
        max_values = max_values + 0.00001
        # 计算每列的最小值
        min_values, _ = torch.min(points_array, dim=0)
        min_values = min_values

        self.di, self.hi, self.wi = di, hi, wi
        # 步长step，每个最终划分的块的大小
        self.ds, self.hs, self.ws = (max_values[0]-min_values[0])/(2**level), (max_values[1]-min_values[1])/(2**level), (max_values[2]-min_values[2])/(2**level)
        # 区间的左边界和右边界  1和2
        self.d1, self.d2 = min_values[0] + self.di*self.ds, min_values[0] + (self.di+1)*self.ds
        self.h1, self.h2 = min_values[1] + self.hi*self.hs, min_values[1] + (self.hi+1)*self.hs
        self.w1, self.w2 = min_values[2] + self.wi*self.ws, min_values[2] + (self.wi+1)*self.ws
        # 属于自己的那一块
        # TODO 可以考虑ghost cell，如果后续出现缝隙的话
        assert points_array.shape[0] == points_value_array.shape[0]

        print(f'At level {level}, node begins to init...', flush=True)
        if level != 0:
            # 对每个维度使用二分查找确定边界
            start_idx = binary_search(self.parent.points_array, 0, self.d1, True)
            end_idx = binary_search(self.parent.points_array, 0, self.d2, False)

            # 使用切片操作提取感兴趣的部分
            filtered_points = self.parent.points_array[start_idx:end_idx]
            filtered_values = self.parent.points_value_array[start_idx:end_idx]

            # 使用向量化操作进一步过滤
            mask = (filtered_points[:, 1] >= self.h1) & (filtered_points[:, 1] < self.h2) & \
                   (filtered_points[:, 2] >= self.w1) & (filtered_points[:, 2] < self.w2)
            self.points_array = filtered_points[mask]
            self.points_value_array = filtered_values[mask]
        else:      # 第一层节点就一个，直接给
            self.points_array = points_array
            self.points_value_array = points_value_array

            self.points_array = torch.Tensor(self.points_array)
            self.points_value_array = torch.Tensor(self.points_value_array)

        print('Node finished init', flush=True)

        self.points_array.to(device)
        self.points_value_array.to(device)
        self.children = []
        self.param = 0     # 新加的
        self.actual_param = 0
        # self.aoi = float((self.data > 0).sum())
        self.var = float(((self.points_value_array-self.points_value_array.mean())**2).mean())
        self.num = self.points_array.shape[0]
        self.num_var = self.var*self.num

    def get_children(self):
        if self.num == 0:
            print('this section has no point, do not continue to dfs')
            return []
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    child = Node(parent=self, level=self.level+1, points_array=self.origin_points_array, points_value_array=self.origin_points_value_array, di=2*self.di+d, hi=2*self.hi+h, wi=2*self.wi+w, device=self.device)
                    self.children.append(child)
        return self.children

    def init_network(self, input, output, hidden, layer, act, output_act, w0=30):
        self.net = MLP(input, output, hidden, layer, act, output_act, w0)


def cal_hidden_output(param, layer, input, output: int = None):
    if output != None:
        if layer >= 2:  # i*h+h+(l-2)*(h^2+h)+h*o+o=p -> (l-2)*h^2+(i+l-1+o)*h+(o-p)=0
            a, b, c = layer-2, input+layer-1+output, output-param
        else:           # i*o+o=p -> wrong
            raise Exception("There is only one layer, and hidden layers cannot be calculated!")
    else:               # i*h+h+(l-1)*(h^2+h)=p -> (l-1)*h^2+(i+l)*h-p=0
        a, b, c = layer-1, input+layer, -param
    if a != 0:
        hidden = int((-b+math.sqrt(b**2-4*a*c))/(2*a))
    else:
        hidden = int(-c/b)
    if hidden < 1:
        hidden = 1
    if output == None:
        output = hidden
    return hidden, output


class OctTreeMLP(nn.Module):
    def __init__(self, opt, points_array, points_value_array) -> None:
        super().__init__()
        self.opt = opt
        self.max_level = opt.Network.max_level - 1
        self.act = opt.Network.act
        self.layer = opt.Network.layer
        self.allocation_scheme = opt.Network.allocation_scheme

        self.data_path = opt.Path
        self.device = opt.Train.device
        self.points_array = points_array
        # self.points_array, self.side_info_1 = normalize_data(points_array, opt.Preprocess.normal_min, opt.Preprocess.normal_max)
        self.points_value_array, self.side_info = normalize_data(points_value_array, opt.Preprocess.normal_min, opt.Preprocess.normal_max)
        self.points_array, self.points_value_array = torch.Tensor(self.points_array), torch.Tensor(self.points_value_array)
        self.loss_weight = opt.Train.weight
        self.leaf_nodes_num = 0

        self.init_tree()
        self.init_network()
        self.init_node_list()
        self.cal_params_total()
        self.move2device(self.device)
        self.sampler = self.init_sampler()
        self.optimizer = self.init_optimizer()
        self.lr_scheduler = self.init_lr_scheduler()

    """init tree structure"""

    def init_tree(self):
        print('init tree...')
        self.base_node = Node(parent=None, level=0, points_array=self.points_array, points_value_array=self.points_value_array, di=0, hi=0, wi=0, device=self.device)
        self.init_tree_dfs(self.base_node)
        print('init tree finished')

    def init_tree_dfs(self, node):  # 用于深度优先搜索（DFS）方式创建八叉树的所有节点。每个节点代表数据的一个子区域。
        if node.level < self.max_level:
            children = node.get_children()
            for child in children:
                self.init_tree_dfs(child)

    """init tree mlps"""                                                        # 注意： level 和 layer  ，前者是树的层，后者是每个节点的MLP的层数

    def get_hyper(self):
        # Parameter allocation scheme: (1) ratio between levels (2) parameter allocation in the same level
        ratio = self.opt.Ratio
        origin_bytes = get_vtk_size_bytes()
        ideal_bytes = int(origin_bytes/ratio)
        ideal_params = int(ideal_bytes/4)
        self.ideal_params = ideal_params   # 新加的
        # level_info = self.opt.Network.level_info
        # node_ratios = [info[0] for info in level_info]                          # 树每层level的 节点的参数的ratio
        # level_ratios = [node_ratios[i] * 8 ** i for i in range(len(node_ratios))]   # 树的每个层level的参数的ratio，是节点ratio*(8**层数)，相当于是这一层节点的总数乘上单个节点的ratio
        # self.level_param = [ideal_params/sum(level_ratios)*ratio for ratio in level_ratios]  # 每一个层的参数的个数
        # self.level_layer = [info[1] for info in level_info]                     # 每个节点由一个MLP组成，每个节点的MLP的层数layer
        # self.level_act = [info[2] for info in level_info]                       # 激活函数  都采用的是sine
        # self.level_allocate = [info[3] for info in level_info]                  # 层内allocate方式有equal均分等方式


    def init_network(self):
        self.get_hyper()
        self.net_structure = {}
        self.init_network_dfs(self.base_node)                                   # 从 base node 往下开始dfs遍历初始化其他节点  层数从0开始
        # sys.exit()
        for key in self.net_structure.keys():
            print('*'*12+key+'*'*12)
            print(self.net_structure[key])

    def init_network_dfs(self, node):
        # layer = self.level_layer[node.level]                                                    # layer表示这一层的MLP有几层
        # act = self.level_act[node.level]

        if self.max_level == 0:                                                                 # 如果只有一层，那么这一层的输入输出的个数就是配置文件里面设置的
            node.param = self.ideal_params
        elif node.level == 0:                                                                   # output设为none 意味着输出层的维度并未预先固定，而是根据某些条件或计算来动态确定。
            node.param = self.ideal_params
        else:
            if self.allocation_scheme == 'equal':
                node.param = node.parent.param / 8
            elif self.allocation_scheme == 'aoi':
                node.param = node.parent.param * node.aoi / node.parent.aoi
            elif self.allocation_scheme == 'var':
                node.param = node.parent.param * node.var / node.parent.var
            elif self.allocation_scheme == 'num':
                node.param = node.parent.param * node.num / node.parent.num
            elif self.allocation_scheme == 'num_var':
                node.param = node.parent.param * node.num_var / node.parent.num_var
            else:
                sys.exit("未设置该层分配策略")
                # node.param = node.parent.param / 8
        # 根据新的设计，只有最后一层设置MLP
        if node.children == []:
            # TODO 设计好 level数、参数个数、MLP的层数的权衡
            input, output = self.opt.Network.input, self.points_value_array.shape[1]
            output_act = False
            # 根据input和output的大小计算这个节点的MLP的hiden（隐藏层）的层数和output的大小
            hidden, output = cal_hidden_output(param=node.param, layer=self.layer, input=input, output=output)
            node.init_network(input=input, output=output, hidden=hidden, layer=self.layer, act=self.act, output_act=output_act, w0=self.opt.Network.w0)  # 构建这一节点的MLP
            node.actual_param = sum([p.data.nelement() for p in node.net.net.parameters()])
            if not f'Level{node.level}' in self.net_structure.keys():
                self.net_structure[f'Level{node.level}'] = {}
            # self.net_structure[f'Level{node.level}'][f'{node.di}-{node.hi}-{node.wi}'] = node.net.hyper
            hyper = node.net.hyper
            self.net_structure[f'Level{node.level}'][f'{node.di}-{node.hi}-{node.wi}'] = '{}->{}->{}({}&{}&{})'.format(
                hyper['input'], hyper['hidden'], hyper['output'], hyper['layer'], hyper['act'], hyper['output_act'])

        print(f'At level {node.level}, number of param of this node is {node.actual_param}. num: {node.num}, var: {node.var}')

        children = node.children
        for child in children:
            self.init_network_dfs(child)

    """init node list"""
    # TODO 结点得用哈希还是按下标找对应的MLP？

    def init_node_list(self):
        self.node_list = []
        self.leaf_node_list = []
        self.tree2list_dfs(self.base_node)

    def tree2list_dfs(self, node):
        self.node_list.append(node)
        children = node.children
        if len(children) != 0:
            for child in children:
                self.tree2list_dfs(child)
        else:
            self.leaf_node_list.append(node)
            self.leaf_nodes_num = self.leaf_nodes_num + 1


    def move2device(self, device: str = 'cpu'):
        for node in self.node_list:
            children = node.children
            if len(children) == 0:
                node.net = node.net.to(device)           # 新设计中只有叶节点有net

    def init_sampler(self):
        batch_size = self.opt.Train.batch_size
        epochs = self.opt.Train.epochs
        self.sampler = PointSampler(data=self.points_array, max_level=self.max_level, batch_size=batch_size, epochs=epochs, leaf_nodes_num=self.leaf_nodes_num, device=self.device)
        return self.sampler

    def init_optimizer(self):
        name = self.opt.Train.optimizer.type
        lr = self.opt.Train.optimizer.lr
        parameters = [{'params': node.net.net.parameters()} for node in self.leaf_node_list]   # 将原先的node_list改为了leaf_node_list
        self.optimizer = create_optim(name, parameters, lr)
        return self.optimizer

    def init_lr_scheduler(self):
        self.lr_scheduler = create_lr_scheduler(self.optimizer, self.opt.Train.lr_scheduler)
        return self.lr_scheduler

    def cal_params_total(self):
        self.params_total = 0
        for node in self.leaf_node_list:   # 修改，只算叶子结点的params，因为在新的设计中，只有叶子节点有MLP
            self.params_total += sum([p.data.nelement() for p in node.net.net.parameters()])
        bytes = self.params_total*4
        origin_bytes = get_vtk_size_bytes()
        self.ratio = origin_bytes/bytes
        print(f'Number of network parameters: {self.params_total}')
        print('Network bytes: {:.2f}KB({:.2f}MB); Origin bytes: {:.2f}KB({:.2f}MB)'.format(bytes/1024, bytes/1024**2, origin_bytes/1024, origin_bytes/1024**2))
        print('Compression ratio: {:.2f}'.format(self.ratio))
        return self.params_total

    """predict in batches"""
    # 每次轮到eval的时候会调用，比较指标，比较的是全体数据，而不是取样！！！

    def predict(self, device: str = 'cpu', batch_size: int = 128):                      # main调用的时候指定了坐标
        self.move2device(device=self.device)
        self.predict_points = np.zeros_like(self.points_array)
        self.predict_points_value = np.zeros_like(self.points_value_array)
        cnt = 0
        for node in tqdm(self.leaf_node_list, desc='Decompressing', leave=False, file=sys.stdout):
            for index in range(0, node.num, batch_size):
                input = node.points_array[index:index+batch_size].to(self.device)  # batch_size超过没有关系
                actual_batch_size = input.shape[0]
                self.predict_points[cnt + index:cnt + index+actual_batch_size] = node.points_array[index:index+actual_batch_size]
                self.predict_points_value[cnt + index:cnt + index+actual_batch_size] = node.net(input).detach().cpu().numpy()
            self.print_loss(node, self.l2loss(node.points_value_array[:], torch.Tensor(self.predict_points_value[cnt:cnt + node.num])))
            cnt = cnt + node.num
        self.predict_points, self.predict_points_value = sort_in_3D_axies(self.predict_points, self.predict_points_value)
        self.move2device(device=self.device)
        return self.predict_points, invnormalize_data(self.predict_points_value, **self.side_info)
        # return invnormalize_data(self.predict_points, **self.side_info_1), invnormalize_data(self.predict_points_value, **self.side_info)

    """cal loss during training"""

    def l2loss(self, data_gt, data_hat):
        loss = F.mse_loss(data_gt, data_hat, reduction='none')
        loss = loss.mean()
        return loss

    def cal_loss(self, batch_size):    # 因为输入的只是一个叶子块大小范围内坐标，故输入一次坐标但是是算了所有叶子节点的坐标的loss
        self.loss = 0
        for node in self.leaf_node_list:
            rand = torch.randint(0, node.points_array.shape[0], (batch_size,))
            input = node.points_array[rand].to(self.device)  # batch_size超过没有关系
            label = node.points_value_array[rand].to(self.device)
            predict = node.net(input)
            self.loss = self.loss + self.l2loss(label, predict)
        # self.loss = self.loss.mean()       # 取平均值
        return self.loss

    def print_loss(self, node, loss):
        print(f'\nnode.param: {node.actual_param}, node.num: {node.num}, node.var: {node.var} -> loss: {loss}')
