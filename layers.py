import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn


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
    def __init__(self, feat_dim, hidden_dim, lam_emd=1., order=4):
        super(GraphEncoder, self).__init__()
        self.order = order
        self.SAL = GlobalSelfAttentionLayer(feat_dim, hidden_dim)
        self.lam_emd = lam_emd
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

        if True:  # or x.shape[0] not in [3327, 19717]:
            h1 = self.SAL(x)
        else:
            h1 = h2
        h = torch.cat([h2, self.lam_emd * h1], dim=-1)
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
        k = torch.matmul(x, self.K)
        q = torch.matmul(x, self.Q)
        k = F.elu(k)
        q = F.elu(q)
        attn = torch.matmul(q, k.T)
        attn = F.normalize(attn, p=2, dim=-1)

        h = torch.mm(attn, x)

        h = F.elu(h)
        # print(attn)
        return h

