import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log, sqrt

def normalize_point_batch(pc, NCHW=True):
    """
    normalize a batch of point clouds
    :param
        pc      [B, N, 3] or [B, 3, N]
        NCHW    if True, treat the second dimension as channel dimension
    :return
        pc      normalized point clouds, same shape as input
        centroid [B, 1, 3] or [B, 3, 1] center of point clouds
        furthest_distance [B, 1, 1] scale of point clouds
    """
    point_axis = 2 if NCHW else 1
    dim_axis = 1 if NCHW else 2
    centroid = torch.mean(pc, dim=point_axis, keepdim=True)
    pc = pc - centroid
    furthest_distance, _ = torch.max(
        torch.sqrt(torch.sum(pc ** 2, dim=dim_axis, keepdim=True)), dim=point_axis, keepdim=True)
    pc = pc / furthest_distance
    return pc

def __batch_distance_matrix_general(A, B):
    """
    :param
        A, B [B,N,C], [B,M,C]
    :return
        D [B,N,M]
    """
    r_A = torch.sum(A * A, dim=2, keepdim=True)
    r_B = torch.sum(B * B, dim=2, keepdim=True)
    m = torch.matmul(A, B.permute(0, 2, 1))
    D = r_A - 2 * m + r_B.permute(0, 2, 1)
    return D

def group_knn(k, query, points, unique=True, NCHW=True):
    """
    group batch of points to neighborhoods
    :param
        k: neighborhood size
        query: BxCxM or BxMxC
        points: BxCxN or BxNxC
        unique: neighborhood contains *unique* points
        NCHW: if true, the second dimension is the channel dimension
    :return
        neighbor_points BxCxMxk (if NCHW) or BxMxkxC (otherwise)
        index_batch     BxMxk
        distance_batch  BxMxk
    """
    if NCHW:
        batch_size, channels, num_points = points.size()
        points_trans = points.transpose(2, 1).contiguous()
        query_trans = query.transpose(2, 1).contiguous()
    else:
        points_trans = points.contiguous()
        query_trans = query.contiguous()

    batch_size, num_points, _ = points_trans.size()
    assert(num_points >= k
           ), "points size must be greater or equal to k"

    D = __batch_distance_matrix_general(query_trans, points_trans)
    if unique:
        # prepare duplicate entries
        points_np = points_trans.detach().cpu().numpy()
        indices_duplicated = np.ones(
            (batch_size, 1, num_points), dtype=np.int32)

        for idx in range(batch_size):
            _, indices = np.unique(points_np[idx], return_index=True, axis=0)
            indices_duplicated[idx, :, indices] = 0

        indices_duplicated = torch.from_numpy(
            indices_duplicated).to(device=D.device, dtype=torch.float32)
        D += torch.max(D) * indices_duplicated

    # (B,M,k)
    distances, point_indices = torch.topk(-D, k, dim=-1, sorted=True)
    # (B,N,C)->(B,M,N,C), (B,M,k)->(B,M,k,C)
    knn_trans = torch.gather(points_trans.unsqueeze(1).expand(-1, query_trans.size(1), -1, -1),
                             2,
                             point_indices.unsqueeze(-1).expand(-1, -1, -1, points_trans.size(-1)))

    if NCHW:
        knn_trans = knn_trans.permute(0, 3, 1, 2)

    return knn_trans, point_indices, -distances



# =========================Layers=========================

class DenseEdgeConv(nn.Module):
    """docstring for EdgeConv"""

    def __init__(self, in_channels, growth_rate, n, k, **kwargs):
        super(DenseEdgeConv, self).__init__()
        self.growth_rate = growth_rate
        self.n = n
        self.k = k
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(torch.nn.Conv2d(
            2 * in_channels, growth_rate, 1, bias=True))
        for i in range(1, n):
            in_channels += growth_rate
            self.mlps.append(torch.nn.Conv2d(
                in_channels, growth_rate, 1, bias=True))

    def get_local_graph(self, x, k, idx=None):
        """Construct edge feature [x, NN_i - x] for each point x
        :param
            x: (B, C, N)
            k: int
            idx: (B, N, k)
        :return
            edge features: (B, C, N, k)
        """
        if idx is None:
            # BCN(K+1), BN(K+1)
            knn_point, idx, _ = group_knn(k + 1, x, x, unique=True)
            idx = idx[:, :, 1:]
            knn_point = knn_point[:, :, :, 1:]

        neighbor_center = torch.unsqueeze(x, dim=-1)
        neighbor_center = neighbor_center.expand_as(knn_point)

        edge_feature = torch.cat(
            [neighbor_center, knn_point - neighbor_center], dim=1)
        return edge_feature, idx

    def forward(self, x, idx=None):
        """
        args:
            x features (B,C,N)
        return:
            y features (B,C',N)
            idx fknn index (B,C,N,K)
        """
        # [B 2C N K]
        for i, mlp in enumerate(self.mlps):
            if i == 0:
                y, idx = self.get_local_graph(x, k=self.k, idx=idx)
                x = x.unsqueeze(-1).repeat(1, 1, 1, self.k)
                y = torch.cat([nn.functional.relu_(mlp(y)), x], dim=1)
            elif i == (self.n - 1):
                y = torch.cat([mlp(y), y], dim=1)
            else:
                y = torch.cat([nn.functional.relu_(mlp(y)), y], dim=1)

        y, _ = torch.max(y, dim=-1)
        return y, idx

class Conv2d(nn.Module):
    """2dconvolution with custom normalization and activation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 activation=None, normalization=None, momentum=0.01):
        super(Conv2d, self).__init__()
        self.activation = activation
        self.normalization = normalization
        bias = not normalization and bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

        if normalization is not None:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm2d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm2d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            else:
                raise ValueError(
                    "only \"batch/instance\" normalization permitted.")

        # activation
        if activation is not None:
            if self.activation == 'relu':
                self.act = nn.ReLU()
            elif self.activation == 'elu':
                self.act = nn.ELU(alpha=1.0)
            elif self.activation == 'lrelu':
                self.act = nn.LeakyReLU(0.1)
            else:
                raise ValueError("only \"relu/elu/lrelu\" allowed")

    def forward(self, x, epoch=None):
        x = self.conv(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x


class Conv1d(nn.Module):
    """1dconvolution with custom normalization and activation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 activation=None, normalization=None, momentum=0.01):
        super(Conv1d, self).__init__()
        self.activation = activation
        self.normalization = normalization
        bias = not normalization and bias
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

        if normalization is not None:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            else:
                raise ValueError(
                    "only \"batch/instance\" normalization permitted.")

        # activation
        if activation is not None:
            if self.activation == 'relu':
                self.act = nn.ReLU()
            elif self.activation == 'elu':
                self.act = nn.ELU(alpha=1.0)
            elif self.activation == 'lrelu':
                self.act = nn.LeakyReLU(0.1)
            else:
                raise ValueError("only \"relu/elu/lrelu\" allowed")

    def forward(self, x, epoch=None):
        x = self.conv(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x

class Feature_Extraction(torch.nn.Module):
    """3PU per-level network"""

    def __init__(self, dense_n=3, growth_rate=12, knn=16):
        super(Feature_Extraction, self).__init__()
        self.dense_n = dense_n

        in_channels = 3
        self.layer0 = Conv2d(3, 24, [1, 1], activation=None)
        self.layer1 = DenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 84  # 24+(24+growth_rate*dense_n) = 24+(24+36) = 84
        self.layer2_prep = Conv1d(in_channels, 24, 1, activation="relu")
        self.layer2 = DenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 144  # 84+(24+36) = 144
        self.layer3_prep = Conv1d(in_channels, 24, 1, activation="relu")
        self.layer3 = DenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 204  # 144+(24+36) = 204
        self.layer4_prep = Conv1d(in_channels, 24, 1, activation="relu")
        self.layer4 = DenseEdgeConv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 264  # 204+(24+36) = 264


    def forward(self, xyz_normalized, previous_level4=None, **kwargs):
        """
        :param
            xyz             Bx3xN input xyz, unnormalized
            xyz_normalized  Bx3xN input xyz, normalized
            previous_level4 tuple of the xyz and feature of the final feature
                            in the previous level (Bx3xM, BxCxM)
        :return
            xyz             Bx3xNr output xyz, normalized
            l4_features     BxCxN feature of the input points
        """

        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis = {}

        x = self.layer0(xyz_normalized.unsqueeze(dim=-1)).squeeze(dim=-1)
        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis["layer_0"] = x

        y, idx = self.layer1(x)
        x = torch.cat([y, x], dim=1)
        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis["layer_1"] = x
            self.vis["nnIdx_layer_0"] = idx

        y, idx = self.layer2(self.layer2_prep(x))
        x = torch.cat([y, x], dim=1)
        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis["layer_2"] = x
            self.vis["nnIdx_layer_1"] = idx

        y, idx = self.layer3(self.layer3_prep(x))
        x = torch.cat([y, x], dim=1)
        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis["layer_3"] = x
            self.vis["nnIdx_layer_2"] = idx

        y, idx = self.layer4(self.layer4_prep(x))
        x = torch.cat([y, x], dim=1)
        if "phase" in kwargs and kwargs["phase"] == "vis":
            self.vis["layer_4"] = x
            self.vis["nnIdx_layer_3"] = idx

        return x


class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)

def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    nn.init.kaiming_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, gain=1):
        super().__init__()

        self.query = spectral_init(nn.Conv1d(in_channel, in_channel // 4, 1),
                                   gain=gain)
        self.key = spectral_init(nn.Conv1d(in_channel, in_channel // 4, 1),
                                 gain=gain)
        self.value = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out