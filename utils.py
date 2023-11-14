import sys

import numpy as np
import scipy.sparse as sp
import torch
import _pickle as pkl
import networkx as nx
from scipy.linalg import fractional_matrix_power, inv
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio

''' Compute Personalized Page Ranking'''
def compute_ppr(a, dataset, alpha=0.2, self_loop=True, epsilon=0.01):
    '''
    :param a: numpy; adj_noeye. (adj without self loop) dense
    :param alpha:
    :param self_loop: bool
    :return: adj_ppr;
    '''
    print('computing ppr......')
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    ppr_adj = alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1
    # ppr_adj[ppr_adj < epsilon] = 0
    if dataset == 'citeseer':
        print('additional processing')
        ppr_adj[ppr_adj < epsilon] = 0
        scaler = MinMaxScaler()
        scaler.fit(ppr_adj)
        ppr_adj = scaler.transform(ppr_adj)
    print('complete ppr')
    ppr_adj_labels = torch.FloatTensor(ppr_adj > epsilon).contiguous()
    ppr_adj = torch.FloatTensor(ppr_adj)

    return ppr_adj, ppr_adj_labels

def sample_graph(adj, drop_rate):
    drop_rate = torch.FloatTensor(np.ones(adj.shape[0]) * drop_rate)
    masks = torch.bernoulli(1. - drop_rate).unsqueeze(1)
    adj_droped = masks * adj
    adj_noeye = adj.mul(adj_droped.T)
    adj_droped = adj_noeye + torch.eye(adj_droped.shape[0])

    rowsum = adj_droped.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    adj_droped = adj_droped.mm(r_mat_inv_sqrt).T.mm(r_mat_inv_sqrt)
    return adj_droped, adj_noeye

def normalize_spadj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_weight(weights, p=1/2, eps=1e-12):
    '''
    :param weights:  a list [w1, w2, w3]
    :param p: default=1
    :param eps:
    :return:
    '''
    ws = np.array(weights)
    ws = np.power(ws, p) # label soft
    ws = ws / ws.max()
    # r = max(np.power(np.power(ws, p).sum(), 1/p), eps)
    # ws = ws / r
    return ws


def normalize_spfeatures(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def normalize_features(x):
    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    x = x * r_inv
    return x

def normalize_adj(x):
    rowsum = np.array(x.sum(1))
    colsum = np.array(x.sum(0))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    c_inv = np.power(colsum, -0.5).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    c_inv = c_inv.reshape((-1, x.shape[1]))
    x = x * r_inv * c_inv
    return x


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_planetoid(dataset, path):
    path= path + dataset
    print('data loading.....')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features_unorm = sp.vstack((allx, tx)).tolil()
    features_unorm[test_idx_reorder, :] = features_unorm[test_idx_range, :]
    features = normalize_spfeatures(features_unorm)
    adj_noeye = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj_temp = adj_noeye

    # norm
    adj = adj_noeye + adj_noeye.T.multiply(adj_noeye.T > adj_noeye) - adj_noeye.multiply(adj_noeye.T > adj_noeye)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_spadj(adj)

    # D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    # D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    # D1 = sp.diags(D1[:, 0], format='csr')
    # D2 = sp.diags(D2[0, :], format='csr')
    # adj = adj.dot(D1)
    # adj = D2.dot(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj = torch.FloatTensor(np.array(adj.todense()))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    adj_labels = torch.FloatTensor(np.array(adj_noeye.todense()) != 0)
    feature_labels = torch.FloatTensor(np.array(features_unorm.todense()))
    print('complete data loading')

    return labels, adj, features, adj_labels, feature_labels


def load_multi(dataset, root):
    # load the data: x, tx, allx, graph
    if dataset == 'acm':
        path = root + 'ACM3025.mat'
    elif dataset == 'dblp':
        path = root + 'DBLP4057.mat'
    elif dataset =='imdb':
        path = root + 'imdb5k.mat'
    data = sio.loadmat(path)
    # print(dataset)
    # print(data)
    # rownetworks = np.array([(data['PLP'] - np.eye(N)).tolist()]) #, (data['PLP'] - np.eye(N)).tolist() , (data['PTP'] - np.eye(N)).tolist()])

    if dataset == "acm":
        truelabels, truefeatures = data['label'], data['feature'].astype(float)
        N = truefeatures.shape[0]
        rownetworks = np.array([(data['PAP']).tolist(), (data['PLP']).tolist()])
    elif dataset == "dblp":
        truelabels, truefeatures = data['label'], data['features'].astype(float)
        N = truefeatures.shape[0]
        rownetworks = np.array([(data['net_APA']).tolist(), (data['net_APCPA']).tolist(), (data['net_APTPA']).tolist()])
        # rownetworks = rownetworks[:2]
    elif dataset == 'imdb':
        truelabels, truefeatures = data['label'], data['feature'].astype(float)
        N = truefeatures.shape[0]
        rownetworks = np.array([(data['MAM']).tolist(), (data['MDM']).tolist(), (data['MYM']).tolist()])
        rownetworks = rownetworks[:2]

    numView = rownetworks.shape[0]
    adjs_labels = []
    adjs = []
    feature_labels = torch.FloatTensor(np.array(truefeatures)).contiguous()
    features = torch.FloatTensor(normalize_features(truefeatures)).contiguous()
    for i in range(numView):
        adjs_labels.append(torch.FloatTensor(np.array(rownetworks[i])))
        adjs.append(torch.FloatTensor(normalize_adj(np.array(rownetworks[i]))))

    labels = torch.LongTensor(np.argmax(truelabels, -1)).contiguous()

    return labels, adjs, features, adjs_labels, feature_labels, numView


def load_data(dataset, path):
    """Load data."""
    if dataset in ['cora', 'citeseer', 'pubmed']:
        num_graph = 1
        labels, adjs, features, adjs_labels, feature_labels = load_planetoid(dataset, path)
    elif dataset in ['acm', 'dblp', 'imdb']:
        labels, adjs, features, adjs_labels, feature_labels, num_graph = load_multi(dataset, path)
    else:
        assert 'Dataset is not exist.'

    return labels, adjs, features, adjs_labels, feature_labels, num_graph

def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_noeye = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj_noeye + sp.eye(adj_noeye.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_noeye)

    return adj, adj_label


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_sharp_common_z(zs, temp=0.5):
    sum = 0.
    for z in zs:
        sum = sum + z
    avg_z = sum / len(zs)
    sharp_z = (torch.pow(avg_z, 1./temp) / torch.sum(torch.pow(avg_z, 1./temp), dim=1, keepdim=True)).detach()
    return sharp_z
