import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn

from layers import GraphEncoder, LatentMappingLayer


class MultiGraphAutoEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, latent_dim, class_num, lam_emd=1., alpha=0.2, order=5, view_num=1):
        super(MultiGraphAutoEncoder, self).__init__()
        self.hidden_size = hidden_dim
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.class_num = class_num


        self.cluster_layer = [Parameter(torch.Tensor(class_num, latent_dim)) for _ in range(view_num)]
        self.cluster_layer.append(Parameter(torch.Tensor(class_num, view_num * latent_dim)))
        # self.cluster_layer.append(torch.cat(self.cluster_layer, dim=-1))
        self.GraphEnc = [GraphEncoder(feat_dim, hidden_dim, lam_emd=lam_emd, order=order) for _ in range(view_num)]
        self.LatentMap = [LatentMappingLayer(2*feat_dim, hidden_dim, latent_dim) for _ in range(view_num)]
        self.FeatDec = [LatentMappingLayer(latent_dim, hidden_dim, feat_dim) for _ in range(view_num)]

        for i in range(view_num):
            self.register_parameter('centroid_{}'.format(i), self.cluster_layer[i])
            self.add_module('graphenc_{}'.format(i), self.GraphEnc[i])
            self.add_module('latentmap_{}'.format(i), self.LatentMap[i])
            self.add_module('featdec_{}'.format(i), self.FeatDec[i])
        self.register_parameter('centroid_{}'.format(view_num), self.cluster_layer[view_num])

    def forward(self, x, adj, view=0):
        # x = torch.dropout(x, 0.2, train=self.training)
        e = self.GraphEnc[view](x, adj)
        # print('view:', view, self.GraphEnc[view].la)
        # print(self.cluster_layer[view])
        z = self.LatentMap[view](e)
        z_norm = F.normalize(z, p=2, dim=1)
        A_pred = self.decode(z_norm)
        q = self.predict_distribution(z_norm, view)

        x_prim = self.FeatDec[view](z)
        x_pred = torch.sigmoid(x_prim)
        return A_pred, z_norm, q, x_pred

    @staticmethod
    def decode(z):
        rec_graph = torch.sigmoid(torch.matmul(z, z.T))
        return rec_graph

    def predict_distribution(self, z, v, alpha=1.0):
        c = self.cluster_layer[v]
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()


    def get_graph_embedding(self, x, adj,view):
        e = self.GraphEnc[view](x, adj)
        e_norm = F.normalize(e, p=2, dim=1)

        return e_norm

