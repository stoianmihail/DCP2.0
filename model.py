#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import quat2mat
from difficp import ICP6DoF
from hypericp import torch_compute_components, torch_compute_components_already_centered, torch_solve_batch

from difficp.utils.geometry_utils import euler_angles_to_rotation_matrix, rotation_matrix_to_euler_angles

torch.cuda.empty_cache()
# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class PointNet(nn.Module):
    def __init__(self, emb_dims=512):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []
        mus = []
        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)
            mus.append(rotation_matrix_to_euler_angles(r, "zyx"))

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3), torch.stack(mus, dim=0)


class DCP(nn.Module):
    def __init__(self, args):
        super(DCP, self).__init__()
        self.emb_dims = args.emb_dims
        self.cycle = args.cycle
        if args.emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif args.emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')

        if args.pointer == 'identity':
            self.pointer = Identity()
        elif args.pointer == 'transformer':
            self.pointer = Transformer(args=args)
        else:
            raise Exception("Not implemented")

        if args.head == 'mlp':
            self.head = MLPHead(args=args)
        elif args.head == 'svd':
            self.head = SVDHead(args=args)
        else:
            raise Exception('Not implemented')

        icp_kwargs = {"verbose": False}
        self.difficp = ICP6DoF(differentiable=True, iters_max=1, **icp_kwargs)
        self.icp = ICP6DoF(differentiable=False, **icp_kwargs)
        self.failures = 0

    # def _refine_with_icp(self, src, tgt, rotation, translation, full_icp=False):
    #     batch_size = src.size()[0]
    #     rotations, translations = [], []
    #     icp = self.icp if full_icp else self.difficp
    #     num_iters = []

    #     # print(f'src.shape={src.shape}, tgt.shape={tgt.shape}')
    #     for i in range(batch_size):
    #         icp_init_pose = torch.eye(4, dtype=rotation.dtype, device=rotation.device)
    #         icp_init_pose[:3, :3] = rotation[i]
    #         icp_init_pose[:3, 3] = translation[i]
    #         try:
    #             pred_pose, num_iter, mse = icp(
    #                 src[i].transpose(0, 1),
    #                 tgt[i].transpose(0, 1),
    #                 icp_init_pose,
    #             )
    #             num_iters.append(num_iter)
    #             rotations.append(pred_pose[:3, :3])
    #             translations.append(pred_pose[:3, 3])
    #         except Exception as e:
    #             print(e)
    #             rotations.append(rotation[i])
    #             translations.append(translation[i])
    #             self.failures += 1
    #             print(self.failures)
    #     print(f'num_iters={num_iters}')
    #     return torch.stack(rotations, 0), torch.stack(translations, 0), torch.Tensor(num_iters)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        rotation_ab, translation_ab, _ = self.head(src_embedding, tgt_embedding, src, tgt)

        # if not self.training:
        #     rotation_ab, translation_ab, num_iters  = self._refine_with_icp(
        #         src, tgt, rotation_ab, translation_ab#, full_icp=True
        #     )

        #     print(f'torch.mean(num_iters)={torch.mean(num_iters)}')

        if self.cycle:
            rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src)
        else:
            rotation_ba = rotation_ab.transpose(2, 1).contiguous()
            translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, 0

class MyBatchNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fn = nn.BatchNorm1d(dim)

    def forward(self, x):
        if x.size(0) > 1:
            return self.fn(x) 
        return x

class DCP_DiffICP(nn.Module):
    def __init__(self, args):
        super(DCP_DiffICP, self).__init__()
        self.emb_dims = args.emb_dims
        self.cycle = args.cycle
        if args.emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif args.emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')

        if args.pointer == 'identity':
            self.pointer = Identity()
        elif args.pointer == 'transformer':
            self.pointer = Transformer(args=args)
        else:
            raise Exception("Not implemented")

        if args.head == 'mlp':
            self.head = MLPHead(args=args)
        elif args.head == 'svd':
            self.head = SVDHead(args=args)
        else:
            raise Exception('Not implemented')

        # self.eval = args.eval

        # TODO: but don't we need the `InitPoseICP`???
        icp_kwargs = {"verbose": False}
        self.difficp = ICP6DoF(differentiable=True, iters_max=1, **icp_kwargs)
        self.icp = ICP6DoF(differentiable=False, **icp_kwargs)
        self.failures = 0

    def _refine_with_icp(self, src, tgt, rotation, translation, full_icp=False):
        batch_size = src.size()[0]
        rotations, translations = [], []
        icp = self.icp if full_icp else self.difficp
        num_iters = []

        # print(f'src.shape={src.shape}, tgt.shape={tgt.shape}')
        for i in range(batch_size):
            icp_init_pose = torch.eye(4, dtype=rotation.dtype, device=rotation.device)
            icp_init_pose[:3, :3] = rotation[i]
            icp_init_pose[:3, 3] = translation[i]
            try:
                pred_pose, num_iter, mse = icp(
                    src[i].transpose(0, 1),
                    tgt[i].transpose(0, 1),
                    icp_init_pose,
                )
                num_iters.append(num_iter)
                rotations.append(pred_pose[:3, :3])
                translations.append(pred_pose[:3, 3])
            except Exception as e:
                print(e)
                rotations.append(rotation[i])
                translations.append(translation[i])
                self.failures += 1
                print(self.failures)
        print(f'num_iters={num_iters}')
        return torch.stack(rotations, 0), torch.stack(translations, 0), torch.Tensor(num_iters)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]

        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        rotation_ab, translation_ab, _ = self.head(src_embedding, tgt_embedding, src, tgt)

        if self.training:
            rotation_ab, translation_ab = self._refine_with_icp(
                src, tgt, rotation_ab, translation_ab
            )
        else:
            rotation_ab, translation_ab, num_iters  = self._refine_with_icp(
                src, tgt, rotation_ab, translation_ab, full_icp=True
            )

            print(f'torch.mean(num_iters)={torch.mean(num_iters)}')

        rotation_ba = rotation_ab.transpose(2, 1).contiguous()
        translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

        return rotation_ab, translation_ab, rotation_ba, translation_ba, 0

def get_important_singular_values(U1, S1, U2, S2):
    def f(svs):
        if svs[0] > svs[1] + 5e-1:
            return 0
        return 2
 
    us1, us2 = [], []
    bs = U1.shape[0]
    for i in range(U1.shape[0]):
        col = f(S1[i])
        # us1.append(S1[i][col] / S2[i][col] * torch.gather(U1[i], 1, torch.tensor([[col]] * 3, device=U1.device)))
        # Or the other way around, i.e., S2[i][col] / S1[i][col]!
        # Or any combination: also put S1[i][col] * in the second line, for `U2`.
        t1 = torch.gather(U1[i], 1, torch.tensor([[col]] * 3, device=U1.device))
        t2 = torch.gather(U2[i], 1, torch.tensor([[col]] * 3, device=U2.device))
        # print(f't1.shape={t1.shape}, t2.shape={t2.shape}, S1[i]={S1[i]}, S2[i]={S2[i]}')
        # t1 *= S1[i][col]# S1[i]
        # factor = S2[i] / S1[i]
        # print(f'factor.shape={factor.shape}')
        # t2 *= S2[i][col]
        # t2 *= (S2[i] / S1[i]).unsqueeze(1)
        # print(f't1.shape={t1.shape}, t2.shape={t2.shape}')
        us1.append(t1)#torch.gather(U1[i], 1, torch.tensor([[col]] * 3, device=U1.device)))
        us2.append(t2)#torch.gather(U2[i], 1, torch.tensor([[col]] * 3, device=U2.device)))
    return torch.concat(us1, axis=0).reshape(bs, 3), torch.concat(us2, axis=0).reshape(bs, 3)

def bdot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

class Sprinter(nn.Module):
    def __init__(self, args):
        super(Sprinter, self).__init__()
        self.emb_dims = args.emb_dims
        self.cycle = args.cycle
        self.formula = args.formula
        print(f'Formula!!!!!!!!!!!!!! -> {self.formula}')
        if args.emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif args.emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')

        if args.pointer == 'identity':
            self.pointer = Identity()
        elif args.pointer == 'transformer':
            self.pointer = Transformer(args=args)
        else:
            raise Exception("Not implemented")

        if args.head == 'mlp':
            self.head = MLPHead(args=args)
        elif args.head == 'svd':
            self.head = SVDHead(args=args)
        else:
            raise Exception('Not implemented')

        # self.eval = args.eval

        # TODO: but don't we need the `InitPoseICP`???
        icp_kwargs = {"verbose": False}
        self.difficp = ICP6DoF(differentiable=True, iters_max=1, **icp_kwargs)
        self.icp = ICP6DoF(differentiable=False, **icp_kwargs)
        self.failures = 0
        
        self.nn = nn.Sequential(
            nn.Linear(self.emb_dims * 2, self.emb_dims // 2),
            MyBatchNorm1d(self.emb_dims // 2),
            nn.LeakyReLU(),
            nn.Linear(self.emb_dims // 2, self.emb_dims // 8),
            MyBatchNorm1d(self.emb_dims // 8),
            nn.LeakyReLU(),
            nn.Linear(self.emb_dims // 8, 3 if self.formula == 'diag' else 6))

    def _refine_with_icp(self, src, tgt, rotation, translation, full_icp=False):
        batch_size = src.size()[0]
        rotations, translations = [], []
        icp = self.icp if full_icp else self.difficp
        # mse = []

        # print(f'src.shape={src.shape}, tgt.shape={tgt.shape}')
        for i in range(batch_size):
            icp_init_pose = torch.eye(4, dtype=rotation.dtype, device=rotation.device)
            icp_init_pose[:3, :3] = rotation[i]
            icp_init_pose[:3, 3] = translation[i]
            try:
                pred_pose, _, _ = icp(
                    src[i].transpose(0, 1),
                    tgt[i].transpose(0, 1),
                    icp_init_pose,
                )
                # mse.append(mse_i)
                rotations.append(pred_pose[:3, :3])
                translations.append(pred_pose[:3, 3])
            except Exception as e:
                print(e)
                rotations.append(rotation[i])
                translations.append(translation[i])
                self.failures += 1
                print(self.failures)
        # print(f'num_iters={num_iters}')
        return torch.stack(rotations, 0), torch.stack(translations, 0)#, mse

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        # debug = input[2]

        src_mean = src.mean(dim=2, keepdim=True)
        tgt_mean = tgt.mean(dim=2, keepdim=True)

        # print(f'src_mean={src_mean}, shape={src_mean.shape}, ttgt_mnea={tgt_mean}, tgt_mean.shape={tgt_mean.shape}')

        # print(f'src.shape={src.shape}, tgt.shape={tgt.shape}')

        src = (src - src_mean)
        tgt = (tgt - tgt_mean)

        # print(f'src.shape={src.shape}, tgt.shape={tgt.shape}')

        # print(f'src_mean={src.mean(dim=2, keepdim=True)}')

        assert torch.allclose(src.mean(dim=2, keepdim=True), torch.zeros_like(src_mean, device=src_mean.device), atol=1e-6)
        assert torch.allclose(tgt.mean(dim=2, keepdim=True), torch.zeros_like(tgt_mean, device=tgt_mean.device), atol=1e-6)

        # (U1, S1), (U2, S2) = torch_compute_components_already_centered(src, tgt)

        # I1, I2 = get_important_singular_values(U1, S1, U2, S2)

        # print(f'U1={U1}, S1={S1}')

        if True:
            src_embedding = self.emb_nn(src)
            tgt_embedding = self.emb_nn(tgt)

            src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

            src_embedding = src_embedding + src_embedding_p
            tgt_embedding = tgt_embedding + tgt_embedding_p

            _, tmp, mu = self.head(src_embedding, tgt_embedding, src, tgt)

            # print(f'mu={mu}')

            translation_ab = torch.zeros_like(tmp)
            # if self.cycle:
            #     rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src)
            # else:

            combined_embedding = torch.cat([src_embedding, tgt_embedding], 1)
            combined_embedding = F.adaptive_max_pool1d(combined_embedding, 1).squeeze(-1)

            # print(f'combined_embedding={combined_embedding}, shape={combined_embedding.shape}')

            # TODO: implement rotation matrix trick, that we transform the delta angles into rotation matrix.
            # TODO: then we don't need to apply `rotation_matrix_to_euler_angles` anymore in `self.head` <- more precise + more efficient.
            # TODO: but we lose the gaussian structure!

            kl = None
            if self.formula == 'diag':
                log_var = self.nn(combined_embedding)
                std = torch.exp(0.5 * log_var)

                # log_var = torch.zeros_like(log_var)
                # assert not log_var.requires_grad

                # print(f'std={std}')

                # print(f'sigma.shape={std}')
                # print(f'diag={torch.diag(sigma)}')
                
                # std = torch.exp(0.5 * logvar)
                z = torch.rad2deg(mu)
                if self.training:
                # TODO: wait, is this the same for all?
                    eps = torch.randn_like(std)
                    # print(f'eps={eps.shape}')
                    z += eps * std

                # Enforce positive angles <-- what if we're at pi / 2 <-- 
                z = torch.abs(z)
                # print(f'z={z}')
                
                # dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=torch.diag(sigma))
                # z = dist.rsample()
                # print(f'z.shape={z.shape}')
                rotation_ab = euler_angles_to_rotation_matrix(torch.deg2rad(z), "zyx")
                kl = -torch.mean(0.5 * torch.sum(log_var.exp() - 1 - log_var, dim = 1), dim = 0)
            else:
                log_cholesky = self.nn(combined_embedding)
                idx = torch.tensor([[0, 1, 2, 1, 2, 2],
                                    [0, 1, 2, 0, 0, 1]], dtype=torch.int64)
                                    # dtype=torch.int64)
                # TODO: add requires grad.
                A = torch.zeros((src.shape[0], 3, 3), device=log_cholesky.device)
                # ret = torch.rand((6,))# if False else torch.tensor([0, 0, 0, -torch.inf, -torch.inf, -torch.inf], dtype=torch.float32)
                # print(ret.dtype)

                # TODO: do the others entries of `A` need by `np.inf`.
                A[:, idx[0], idx[1]] = (0.5 * log_cholesky).exp()
                assert A.requires_grad_
                # S = A @ A.transpose(1, 2).contiguous()

                # print(f'log_cholesky={log_cholesky}')

                # print(f'A.shape={A.shape}, eps.shape={eps.shape}, eps.uns={eps.unsqueeze(-1).shape}')
                
                z = torch.rad2deg(mu)
                if self.training:
                    eps = torch.randn_like(mu)
                    assert A.shape[0] == eps.shape[0]
                    z += torch.bmm(A, eps.unsqueeze(-1)).squeeze(-1)
                
                assert z.shape == mu.shape
                z = torch.abs(z)

                rotation_ab = euler_angles_to_rotation_matrix(torch.deg2rad(z), "zyx")

                kl = -torch.mean(0.5 * (log_cholesky.exp().sum(dim=1) - log_cholesky[:, :3].sum(dim=1) - 3))
                # test_kl = -torch.mean(0.5 * torch.sum(log_cholesky.exp() - 1 - log_cholesky, dim = 1), dim = 0)
                # print(f'kl={kl}, test_kl={test_kl}')
                # assert torch.isclose(kl, test_kl, atol=1e-6)
                # test_loss = 0.5 * (S.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - torch.log(torch.linalg.det(S)) - 3)
                # assert torch.allclose(loss, test_loss)
                # return loss

            


            # print(f'pre_sigma={pre_sigma.shape}')

            # input = torch.cat([pre_sigma, mu], 1)
            # print(f'input={input.shape}')
            # sigma = self.compressor(input)
            # assert sigma.shape == pre_sigma.shape

            
            # Sample.
            # print(f'mu={mu}')

                    # self.candidates = self.mus + self.sigmas * torch.randn(self.planning_horizon, self.batch_size, self.n_candidates, self.action_size).to(self.device)  # (H, B, C, 6)

            # sigma = torch.exp(self.nn(com))

            # curr_num_cands = self.num_cands if self.training else 1
            # z, curr_sigmas, ratio = self.reparameterize(batch_size, curr_num_cands, mu, sigma)

            # TODO: compute translation based on rotation.
            # TODO: subtract means at the beginning.

        
            # TODO: this could fail.
            if self.training:
                assert not translation_ab.requires_grad
                rotation_ab, translation_ab = self._refine_with_icp(
                    src, tgt, rotation_ab, translation_ab
                )
            # else:
                # best_loss, best_z = [None for _ in range(z.shape[0])], torch.zeros_like(z)
                # for _ in range(5):
                #     eps = torch.randn_like(std)
                #     z_i = z + eps * std
                #     rotation_ab = euler_angles_to_rotation_matrix(torch.deg2rad(z_i), "zyx")
                #     _, _, loss_i = self._refine_with_icp(
                #         src, tgt, rotation_ab, translation_ab, full_icp=True
                #     )
                #     for i in range(z.shape[0]):
                #         if best_loss[i] is None or best_loss[i] > loss_i[i]:
                #             best_z[i], best_loss[i] = z_i[i], loss_i[i]
                # z = best_z
            # else:
            #     rotation_ab, translation_ab  = self._refine_with_icp(
            #         src, tgt, rotation_ab, translation_ab, full_icp=True
            #     )

            # print(f'torch.mean(num_iters)={torch.mean(num_iters)}')
        else:
            kl = 0
            rotation_ab, translation_ab = torch_solve_batch(src, tgt, verbose=False)


        # TODO: should be the other way around?
        # print(f'rotation_ab.shape={rotation_ab.shape}, I1.shape={I1.shape}')
        # mm1 = torch.matmul(rotation_ab, I1.unsqueeze(2)).squeeze(2)
        # print(f'mm1={mm1}, I2={I2}')
        # print(f'mm -')
        # TODO: also account for sign (axis rotation)

        # diff = 1 - torch.mean(bdot(mm1, I2))

        # diff = torch.mean(1 - torch.dot())

        # diff = torch.mean(torch.mean((mm1 - I2)**2, dim=1))
        # print(f'diff={diff}')

        # TODO: should we reocompute translation?
        # TODO: yes, since the compute translation would be actually zero!!!
        # assert torch.allclose(translation_ab, torch.zeros)
        # print(f'translation_ab[]={translation_ab[0]}\nbefore: {translation_ab.shape}')
        translation_ab = (torch.matmul(-rotation_ab, src_mean) + tgt_mean).squeeze(2)

        # print(f'trasnlation-ab.shape={translation_ab.shape}')

        # TODO: really use this during training?
        # if not self.training:
        #     rotation_ab, translation_ab = self._refine_with_icp(
        #         src, tgt, rotation_ab, translation_ab, full_icp=True,
        #     )

        rotation_ba = rotation_ab.transpose(2, 1).contiguous()
        translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

        return rotation_ab, translation_ab, rotation_ba, translation_ba, kl