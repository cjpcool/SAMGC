import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, concat=True):
        h = torch.mm(input, self.W)
        #前馈神经网络
        attn_for_self = torch.mm(h,self.a_self)       #(N,1)
        attn_for_neighs = torch.mm(h,self.a_neighs)   #(N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs,0,1)
        M = adj.bool()*1.0
        attn_dense = torch.mul(attn_dense,M)
        attn_dense = self.leakyrelu(attn_dense)            #(N,N)

        #掩码（邻接矩阵掩码）
        zero_vec = -9e15*torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention,h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
            return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LatentMappingLayer, self).__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z = self.encode(x)
        return z

    def encode(self, x):
        h = self.enc1(x)
        # h = torch.dropout(h, 0.2, train=self.training)
        h = F.elu(h)
        h = self.enc2(h)
        h = F.elu(h)
        return h



class GraphEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, order=4, alpha=0.2):
        super(GraphEncoder, self).__init__()
        self.order = order
        self.SAL = GlobalSelfAttentionLayer(feat_dim, hidden_dim)
        self.la = nn.Parameter(torch.ones(self.order))
        nn.init.normal_(self.la.data, mean=1., std=.001)
        # if cora, citeseer, pubmed
        # self.la = torch.ones(self.order)
        # self.la[1] = 2

    def forward(self, x, adj):
        # sattn = self.SAL(x)# + torch.eye(adj.shape[0], device=x.device)
        # a = [1. for i in range(self.order)]
        if self.order != 0:
            adj_temp = self.la[0] * adj.clone().detach()
            for i in range(self.order-1):
                adj_temp += self.la[i+1] * torch.matmul(adj, adj_temp).detach_()
            attn = adj_temp / self.order
            h2 = torch.mm(attn, x)
        else:
            h2 = x

        # h1 = torch.mm(sattn, x)
        if True or x.shape[0] not in [3327, 19717]:
            h1 = self.SAL(x)
        else:
            h1 = h2
        h = torch.cat([h2, h1], dim=-1)
        return h


class GlobalSelfAttentionLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(GlobalSelfAttentionLayer, self).__init__()
        self.feat_dim = feat_dim
        # self.Q = nn.Linear(feat_dim, hidden_dim, bias=False)
        # self.K = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.Q = nn.Parameter(torch.zeros(feat_dim, hidden_dim))
        nn.init.xavier_normal_(self.Q.data, gain=1.141)
        self.K = nn.Parameter(torch.zeros(feat_dim, hidden_dim))
        nn.init.xavier_normal_(self.K.data, gain=1.141)
        # self.V = nn.Parameter(torch.zeros(feat_dim, hidden_dim))
        # nn.init.xavier_normal_(self.V.data, gain=1.141)

    def forward(self, x):
        # x = x.bool().float()
        k = torch.matmul(x, self.K)
        q = torch.matmul(x, self.Q)
        # k = F.relu(k)
        # q = F.relu(q)
        attn = torch.matmul(q, k.T)
        # attn = torch.relu(attn)
        # attn = F.normalize(attn, p=2, dim=-1)
        attn = torch.softmax(attn,-1)
        # print(attn)
        # v = torch.matmul(x, self.V)
        h = torch.mm(attn, x)
        # h = F.elu(h)
        h = F.relu(h)
        # print(attn)
        return h

