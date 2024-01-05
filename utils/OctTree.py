import torch
from torch import nn
import numpy as np
import os
from einops import rearrange
import sys
import math
from tqdm import tqdm
import torch.nn.functional as F
from utils.tool import read_img, save_img
from utils.Sampler import create_optim, create_flattened_coords, PointSampler, create_lr_scheduler
from utils.Network import MLP


class Node():
    def __init__(self, parent, level, origin_data, di, hi, wi):
        self.level = level
        self.parent = parent
        self.origin_data = origin_data
        self.di, self.hi, self.wi = di, hi, wi
        # 步长step，每个最终划分的块的大小
        self.ds, self.hs, self.ws = origin_data.shape[0]//(2**level), origin_data.shape[1]//(2**level), origin_data.shape[2]//(2**level)
        # 区间的左边界和右边界  1和2
        self.d1, self.d2 = self.di*self.ds, (self.di+1)*self.ds
        self.h1, self.h2 = self.hi*self.hs, (self.hi+1)*self.hs
        self.w1, self.w2 = self.wi*self.ws, (self.wi+1)*self.ws
        # 属于自己的那一块
        # TODO 可以考虑ghost cell，如果后续出现缝隙的话
        self.data = origin_data[self.d1:self.d2, self.h1:self.h2, self.w1:self.w2]
        self.data = rearrange(self.data, 'd h w n-> (d h w) n') # 扁平化
        self.children = []
        self.predict_data = np.zeros_like(self.data)
        # TODO 后续可以考虑使用，虽然不再采用原作者的下层继承上层，但可以延续原作者的同一层参数分配的思路
        self.aoi = float((self.data > 0).sum())
        self.var = float(((self.data-self.data.mean())**2).mean())

    def get_children(self):
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    child = Node(parent=self, level=self.level+1, origin_data=self.origin_data, di=2*self.di+d, hi=2*self.hi+h, wi=2*self.wi+w)
                    self.children.append(child)
        return self.children

    def init_network(self, input, output, hidden, layer, act, output_act, w0=30):
        self.net = MLP(input, output, hidden, layer, act, output_act, w0)


def normalize_data(data: np.ndarray, scale_min, scale_max):
    dtype = data.dtype  # 存储了原始数据的数据类型
    data = data.astype(np.float32)
    data_min, data_max = data.min(), data.max()
    data = (data - data_min)/(data_max - data_min)
    data = data*(scale_max - scale_min) + scale_min
    data = torch.tensor(data, dtype=torch.float)
    side_info = {'scale_min': scale_min, 'scale_max': scale_max, 'data_min': data_min, 'data_max': data_max, 'dtype': dtype}
    return data, side_info


def invnormalize_data(data: np.ndarray, scale_min, scale_max, data_min, data_max, dtype):
    data = (data - scale_min)/(scale_max - scale_min)
    data = data*(data_max - data_min) + data_min
    data = data.astype(dtype=dtype)
    return data


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
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.max_level = len(opt.Network.level_info)-1
        self.data_path = opt.Path
        self.device = opt.Train.device
        self.data, self.side_info = normalize_data(read_img(self.data_path), opt.Preprocess.normal_min, opt.Preprocess.normal_max)
        self.loss_weight = opt.Train.weight

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
        self.base_node = Node(parent=None, level=0, origin_data=self.data, di=0, hi=0, wi=0)
        self.init_tree_dfs(self.base_node)

    def init_tree_dfs(self, node):  # 用于深度优先搜索（DFS）方式创建八叉树的所有节点。每个节点代表数据的一个子区域。
        if node.level < self.max_level:
            children = node.get_children()
            for child in children:
                self.init_tree_dfs(child)

    """init tree mlps"""                                                        # 注意： level 和 layer  ，前者是树的层，后者是每个节点的MLP的层数

    def get_hyper(self):
        # Parameter allocation scheme: (1) ratio between levels (2) parameter allocation in the same level
        ratio = self.opt.Ratio
        origin_bytes = os.path.getsize(self.data_path)
        ideal_bytes = int(origin_bytes/ratio)
        ideal_params = int(ideal_bytes/4)
        level_info = self.opt.Network.level_info
        # TODO  后续要改
        node_ratios = [info[0] for info in level_info]                          # 树每层level的 节点的参数的ratio
        level_ratios = [node_ratios[i] * 8 ** i for i in range(len(node_ratios))]   # 树的每个层level的参数的ratio，是节点ratio*(8**层数)，相当于是这一层节点的总数乘上单个节点的ratio
        self.level_param = [ideal_params/sum(level_ratios)*ratio for ratio in level_ratios]  # 每一个层的参数的个数
        self.level_layer = [info[1] for info in level_info]                     # 每个节点由一个MLP组成，每个节点的MLP的层数layer
        self.level_act = [info[2] for info in level_info]                       # 激活函数  都采用的是sine
        self.level_allocate = [info[3] for info in level_info]                  # 层内allocate方式有equal均分等方式

    """
        以配置文件中的三层举例：[[1.0, 2, Sine, euqal], [1.0, 2, Sine, equal], [1.5, 3, Sine, equal]]
        树每层level的每个节点的参数的比例为：    1 : 1 : 1.5
        树每层level的参数的比例为：            8 ： 64 ：768
        以自带的数据举例：0.25MB的数据压缩比为64时的参数个数为928
        这就意味着第一层仅仅能分配到10个左右的参数
        这也就解释了为什么在三层level的树高的解压缩的图片会出现非常非常明显的`分块`的效果
        因为根据树结构的设计，第一层的10个参数会作用于后面的两层level
        因为第一层的10个参数对后面两层施加了较大的影响，后面两层对这种影响无法`扭转`
        // 疑问：MLP的前面的层的影响会比较大吗？

        的确，作者文章中提到的层间和层内参数分配的思路可以解决这个问题，但如何或者是通过何种指标来实现这种分配是一个很大的困难
        作者只是在文章中体现了这种思想，在代码层面没有任何体现，
    """

    def init_network(self):
        self.get_hyper()
        self.net_structure = {}
        self.init_network_dfs(self.base_node)                                   # 从 base node 往下开始dfs遍历初始化其他节点  层数从0开始
        for key in self.net_structure.keys():
            print('*'*12+key+'*'*12)
            print(self.net_structure[key])

    def init_network_dfs(self, node):
        layer = self.level_layer[node.level]                                                    # layer表示这一层的MLP有几层
        act = self.level_act[node.level]
        # if self.max_level == 0:                                                                 # 如果只有一层，那么这一层的输入输出的个数就是配置文件里面设置的
        #     input, output, output_act = self.opt.Network.input, self.opt.Network.output, False  # self.opt.Network.input, self.opt.Network.output 在yaml配置文件有，默认分别为3 1
        #     # output_act的true, false是指这一层的输出是否应用激活函数，显然，如果这一层的output是最后的输出则不需要，否则需要
        #     param = self.level_param[node.level]
        # elif node.level == 0:                                                                   # output设为none 意味着输出层的维度并未预先固定，而是根据某些条件或计算来动态确定。
        #     input, output, output_act = self.opt.Network.input, None, True
        #     param = self.level_param[node.level]
        # elif node.level < self.max_level:
        #     input, output, output_act = node.parent.net.hyper['output'], None, True
        #     if self.level_allocate[node.level] == 'equal':
        #         param = self.level_param[node.level]/8**node.level
        #     elif self.level_allocate[node.level] == 'aoi':
        #         param = self.level_param[node.level]*node.aoi/self.base_node.aoi
        #     elif self.level_allocate[node.level] == 'var':
        #         param = self.level_param[node.level]*node.var/sum([child.var for child in node.parent.children])
        #     else:
        #         param = self.level_param[node.level]/8**node.level
        # else:
        #     input, output, output_act = node.parent.net.hyper['output'], self.opt.Network.output, False
        #     if self.level_allocate[node.level] == 'equal':
        #         param = self.level_param[node.level]/8**node.level
        #     elif self.level_allocate[node.level] == 'aoi':
        #         param = self.level_param[node.level]*node.aoi/self.base_node.aoi
        #     elif self.level_allocate[node.level] == 'var':
        #         param = self.level_param[node.level]*node.var/sum([child.var for child in node.parent.children])
        #     else:
        #         param = self.level_param[node.level]/8**node.level

        # 根据新的设计，只有最后一层设置MLP
        if self.max_level != self.max_level:
            return
        
        # TODO 设计好 level数、参数个数、MLP的层数的权衡
        # TODO 这样的话，解压缩得是并行的才快？
        input, output = self.opt.Network.input, self.opt.Network.output
        # 根据input和output的大小计算这个节点的MLP的hiden（隐藏层）的层数和output的大小
        hidden, output = cal_hidden_output(param=param, layer=layer, input=input, output=output)
        node.init_network(input=input, output=output, hidden=hidden, layer=layer, act=act, output_act=output_act, w0=self.opt.Network.w0)  # 构建这一节点的MLP
        if not f'Level{node.level}' in self.net_structure.keys():
            self.net_structure[f'Level{node.level}'] = {}
        # self.net_structure[f'Level{node.level}'][f'{node.di}-{node.hi}-{node.wi}'] = node.net.hyper
        hyper = node.net.hyper
        self.net_structure[f'Level{node.level}'][f'{node.di}-{node.hi}-{node.wi}'] = '{}->{}->{}({}&{}&{})'.format(
            hyper['input'], hyper['hidden'], hyper['output'], hyper['layer'], hyper['act'], hyper['output_act'])
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

    def move2device(self, device: str = 'cpu'):
        for node in self.node_list:
            node.net = node.net.to(device)

    def init_sampler(self):
        batch_size = self.opt.Train.batch_size
        epochs = self.opt.Train.epochs
        self.sampler = PointSampler(data=self.data, max_level=self.max_level, batch_size=batch_size, epochs=epochs, device=self.device)
        return self.sampler

    def init_optimizer(self):
        name = self.opt.Train.optimizer.type
        lr = self.opt.Train.optimizer.lr
        parameters = [{'params': node.net.net.parameters()} for node in self.node_list]
        self.optimizer = create_optim(name, parameters, lr)
        return self.optimizer

    def init_lr_scheduler(self):
        self.lr_scheduler = create_lr_scheduler(self.optimizer, self.opt.Train.lr_scheduler)
        return self.lr_scheduler

    def cal_params_total(self):
        self.params_total = 0
        for node in self.node_list:
            self.params_total += sum([p.data.nelement() for p in node.net.net.parameters()])
        bytes = self.params_total*4
        origin_bytes = os.path.getsize(self.data_path)
        self.ratio = origin_bytes/bytes
        print(f'Number of network parameters: {self.params_total}')
        print('Network bytes: {:.2f}KB({:.2f}MB); Origin bytes: {:.2f}KB({:.2f}MB)'.format(bytes/1024, bytes/1024**2, origin_bytes/1024, origin_bytes/1024**2))
        print('Compression ratio: {:.2f}'.format(self.ratio))
        return self.params_total

    """predict in batches"""
    # 每次轮到eval的时候会调用，比较指标，比较的是全体数据，而不是取样！！！
    def predict(self, device: str = 'cpu', batch_size: int = 128):                      # TODO 为什么不用GPU？
        batch_size = min(self.data.shape[0], self.data.shape[1], self.data.shape[2])    # 为了后面predict方便
        assert self.data.shape[0] == self.data.shape[1] and self.data.shape[0] == self.data.shape[2], "要求三维的长宽高一样，不一样的情况暂时未设计" # TODO 后面可以考虑加上，但会非常麻烦
        self.predict_data = np.zeros_like(self.data)
        self.move2device(device=device)
        coords = self.sampler.coords.to(device)
        # coords.shape[0]表示的是coords的第一维的长度，即coords的长度，全体坐标，eg: 512*512*512
        for index in tqdm(range(0, coords.shape[0], batch_size), desc='Decompressing', leave=False, file=sys.stdout):
            # 这里的coords是扁平化的坐标，相当于是用一个一维的坐标来表示三维的坐标
            # coords[index:index+batch_size]表示的是从index开始的batch_size个连续的坐标
            input = coords[index:index+batch_size]
            # predict是树调用的，predict_dfs的结果是计算出节点的predict_data
            self.predict_dfs(self.base_node, index, batch_size, input)
        self.merge()
        # predict_data是最下面一层叶节点那一块的预测值
        self.predict_data = self.predict_data.clip(self.side_info['scale_min'], self.side_info['scale_max'])
        self.predict_data = invnormalize_data(self.predict_data, **self.side_info)
        self.move2device(device=self.device)
        return self.predict_data

    def predict_dfs(self, node, index, batch_size, input):
        # 因为原设计是八叉树的每个节点都有MLP，新设计的只有叶节点有MLP
        # 这里后面需要修改以下，最小程度的修改，就是让非叶子节点的输出等于输入
        min_dimension_size = min(self.data.shape[0], self.data.shape[1], self.data.shape[2])
        assert min_dimension_size % batch_size == 0, "batch_size must be divisible by the smallest block dimension"
        if len(node.children) > 0:
            # input = node.net(input)
            # 需要要保证coords[index:index+batch_size]这batch_size在一个块内，以便处理
            # 这就需要保证batch_size可以被块的行，宽，高中的最小值整除
            # TODO 这里可能可以继续优化
            ds, hs, ws = self.data.shape[0]//(2**(level + 1)), self.data.shape[1]//(2**(level + 1)), self.data.shape[2]//(2**(level + 1))
            # 区间的左边界和右边界  1和2
            # self.d1, self.d2 = self.di*self.ds, (self.di+1)*self.ds
            # self.h1, self.h2 = self.hi*self.hs, (self.hi+1)*self.hs
            # self.w1, self.w2 = self.wi*self.ws, (self.wi+1)*self.ws
            z = index // (hs * ws)
            y = (index % (hs * ws)) // ws
            x = (index % (hs * ws)) % ws
            children = node.children             
            for child in children:
                if child.d1 <= z*ds and child.d2 > z*ds and child.h1 <= y*hs and child.h2 > y*hs and child.w1 <= x*ws and child.w2 > x*ws:
                    self.predict_dfs(child, index, batch_size, input)
        else:
            node.predict_data[index:index+batch_size] = node.net(input).detach().cpu().numpy()

    def merge(self):
        for node in self.leaf_node_list:
            chunk = node.predict_data  # 节点的predict_data
            chunk = rearrange(chunk, '(d h w) n -> d h w n', d=node.ds, h=node.hs, w=node.ws)
            # 下面的是树的predict_data，和上面那个区分开
            self.predict_data[node.d1:node.d2, node.h1:node.h2, node.w1:node.w2] = chunk

    """cal loss during training"""

    def l2loss(self, data_gt, data_hat):
        loss = F.mse_loss(data_gt, data_hat, reduction='none')
        weight = torch.ones_like(data_gt)
        l, h, scale = self.loss_weight
        weight[(data_gt >= l)*(data_gt <= h)] = scale
        loss = loss*weight
        loss = loss.mean()
        return loss

    def cal_loss(self, idxs, coords):
        self.loss = 0
        self.forward_dfs(self.base_node, idxs, coords)
        self.loss = self.loss.mean()
        return self.loss

    def forward_dfs(self, node, idxs, input):
        if len(node.children) > 0:
            input = node.net(input)
            children = node.children
            for child in children:
                self.forward_dfs(child, idxs, input)
        else:
            predict = node.net(input)
            # label = node.data[idxs:idxs+self.sampler.batch_size, :].to(self.device)
            label = node.data[idxs, :].to(self.device)
            self.loss = self.loss + self.l2loss(label, predict)
