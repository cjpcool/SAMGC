import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from evaluation import eva
from utils import load_data, compute_ppr, get_sharp_common_z, sample_graph, normalize_weight
from torch.optim import Adam
from models import MultiGraphAutoEncoder
# ============================ 1.parameters ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='acm', help='acm, dblp, imdb, cora, citeseer, pubmed')
parser.add_argument('--train', type=bool, default=False, help='training mode')
parser.add_argument('--model_name', type=str, default='samgc_acm', help='model name')

parser.add_argument('--path', type=str, default='./data/', help='')
parser.add_argument('--order', type=int, default=16, help='aggregation orders')  # cora=[8,6] citeseer=4 acm=[16] dblp=9
parser.add_argument('--weight_soft', type=float, default=0.5, help='parameter of p')  # acm=0, dblp=[0.2,0.3]
parser.add_argument('--lam_emd', type=float, default=1., help='trade off between global self-attention and gnn')
parser.add_argument('--kl_step', type=float, default=10., help='lambda kl')

parser.add_argument('--lam_consis', type=float, default=10., help='lambda consis')
parser.add_argument('--hidden_dim', type=int, default=256, help='lambda consis')  # citeseer=[512] others=default
parser.add_argument('--latent_dim', type=int, default=64, help='lambda consis')  # citeseer=[16] others=default

parser.add_argument('--epoch', type=int, default=20000, help='')
parser.add_argument('--patience', type=int, default=100, help='')
parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='')
parser.add_argument('--temperature', type=float, default=0.5, help='')
parser.add_argument('--cuda_device', type=int, default=1, help='')
parser.add_argument('--use_cuda', type=bool, default=True, help='')
parser.add_argument('--update_interval', type=int, default=1, help='')
parser.add_argument('--random_seed', type=int, default=2022, help='')
parser.add_argument('--add_graph', type=bool, default=True, help='')


args = parser.parse_args()

train = args.train
dataset = args.dataset  # [imdb, dblp, acm]  [cora, citeseer, pubmed]
path = args.path
order = args.order  # acm=16, dblp=9,10, imdb=0,1,2
weight_soft = args.weight_soft # acm=0., dblp= [0.0-0.5], imdb
kl_step = args.kl_step  # acm=0.09,10. dblp = 1., imdb
kl_max = kl_step  # acm=10 dblp=100, imdb
lam_consis = args.lam_consis  # acm=10 dblp=1 current0.5, imdb
lam_emd = args.lam_emd

add_graph=args.add_graph
hidden_dim = args.hidden_dim
latent_dim = args.latent_dim
epoch = args.epoch
patience = args.patience
lr = args.lr
weight_decay = args.weight_decay
temprature = args.temperature
cuda_device = args.cuda_device
use_cuda = args.use_cuda
update_interval = args.update_interval
random_seed = args.random_seed

torch.manual_seed(random_seed)

# ============================ 2.dataset and model preparing ==========================
labels, adjs, features, adjs_labels, feature_labels, graph_num = load_data(dataset, path)
# graph_num = 1
# adjs=adjs[:1]
# adjs_labels = adjs_labels[:1]

class_num = int(labels.max()+1)
feat_dim = features.shape[1]

if dataset in ['cora', 'citeseer', 'pubmed']:
    # drop_adj, drop_adj_labels = sample_graph(adj_labels*1.0, drop_rate=drop_rate)
    if add_graph:
        # ppr_adj, ppr_adj_labels = compute_ppr(adjs_labels[0].numpy(), dataset=dataset)
        graph_num+=1
        adjs = [adjs, adjs.clone()]
        # adjs.append(adjs)
        adjs_labels = [adjs_labels, adjs_labels.clone()]
        # adjs_labels.append(ppr_adj_labels)
    # adj_labels = ppr_adj_labels


print(
    'dataset informations:\n',
    'class_num:{}\n'.format(class_num),
    'graph_num:{}\n'.format(graph_num),
    'feat_dim:{}\n'.format(feat_dim),
    'node_num:{}'.format(features.shape[0]),end='\n'
)
for i in range(graph_num):
    print('G^{} edge num:{}'.format(i+1, int(adjs_labels[i].sum())))


model = MultiGraphAutoEncoder(feat_dim, hidden_dim, latent_dim, class_num, lam_emd=lam_emd, order=order, view_num=graph_num)

if use_cuda:
    torch.cuda.set_device(cuda_device)
    torch.cuda.manual_seed(random_seed)
    model = model.cuda()
    adjs = [a.cuda() for a in adjs]
    adjs_labels = [adj_labels.cuda() for adj_labels in adjs_labels]
    features = features.cuda()
    feature_labels = feature_labels.cuda()

device = features.device

# ------------------------------------------- optimizer -------------------------------
# optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
param_ge = []
param_ae = []
for i in range(graph_num):
    param_ge.append({'params': model.GraphEnc[i].parameters()})
    param_ae.append({'params': model.FeatDec[i].parameters()})
    param_ae.append({'params': model.LatentMap[i].parameters()})
    param_ae.append({'params': model.cluster_layer[i]})
param_ae.append({'params': model.cluster_layer[graph_num]})

optimizer_ge = Adam(param_ge + param_ae,
                 lr=lr, weight_decay=weight_decay)

# cluster parameter initiate
y = labels.cpu().numpy()



# ============================ 3.Training ==========================
if train:
    with torch.no_grad():
        zs = []
        kmeans = KMeans(n_clusters=class_num, n_init=3)
        for i in range(graph_num):
            _, z, _, _ = model(features, adjs[i])
            zs.append(z)
            y_pred = kmeans.fit_predict(z.data.cpu().numpy())
            y_pred_last = y_pred
            model.cluster_layer[i].data = torch.tensor(kmeans.cluster_centers_).to(device)
            eva(y, y_pred, 'K{}'.format(i))

        z = torch.cat(zs, dim=-1)
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        y_pred_last = y_pred
        model.cluster_layer[-1].data = torch.tensor(kmeans.cluster_centers_).to(device)
        eva(y, y_pred, 'Kz')
        print()

    bad_count = 0
    best_loss = 100
    best_acc = 1e-12
    best_nmi = 1e-12
    best_ari = 1e-12
    best_f1 = 1e-12
    best_epoch = 0
    l = 0.0
    best_a = [1e-12 for i in range(graph_num)]
    weights = normalize_weight(best_a)

    # for i in range(num_graph+1):
    #     model.cluster_layer[i].requires_grad = False

    for epoch_num in range(epoch):
        # drop_adj, drop_adj_labels = sample_graph(adj_labels.clone(), drop_rate=drop_rate)
        # adjs[1] = drop_adj.to(device)
        # adjs_labels[1] = drop_adj_labels.to(device)
        model.train()

        zs = []
        x_preds = []
        qs = []
        re_loss = 0.
        consis_loss = 0.
        re_feat_loss = 0.
        kl_loss = 0.

        # ----------------------------- compute reconstruct loss for each view---------------------
        for v in range(graph_num):
            A_pred, z, q, x_pred = model(features, adjs[v], view=v)
            zs.append(z)
            qs.append(q)
            x_preds.append(x_pred.unsqueeze(0))
            re_loss += F.binary_cross_entropy(A_pred.view(-1), adjs_labels[v].view(-1))
            re_feat_loss += F.binary_cross_entropy(x_pred.view(-1), feature_labels.view(-1))

        # ------------------------------- weight assignment with pseudo labels ---------------------------------------
        with torch.no_grad():
            # h_prim = torch.cat(zs, dim=-1).detach()
            h_prim = torch.cat([zs[i] * weights[i] for i in range(graph_num)], dim=-1).detach()
            kmeans = KMeans(n_clusters=class_num, n_init=3)
            y_prim = kmeans.fit_predict(h_prim.cpu().numpy())
            for v in range(graph_num):
                y_pred = kmeans.fit_predict(zs[v].detach().cpu().numpy())
                a = eva(y_prim, y_pred, visible=False, metrics='nmi')
                # if a > best_a[i]:
                best_a[v] = a
            weights = normalize_weight(best_a, p=weight_soft)

        # ------------------------------------- consistency -----------------------------------
        # compute consis_loss and common_z
        common_z = get_sharp_common_z(zs, temp=temprature)
        for z in zs:
            # consis_loss += F.mse_loss(zs[0], zs[1])
            consis_loss += F.mse_loss(common_z, z)
        # consis_loss /= graph_num
        consis_loss *= lam_consis

        # ---------------------------------------- kl loss------------------------------------
        h = torch.cat([zs[i] * weights[i] for i in range(graph_num)], dim=-1)

        qh = model.predict_distribution(h, -1)
        p = model.target_distribution(qh)
        kl_loss += F.kl_div(qh.log(), p, reduction='batchmean')
        for i in range(graph_num):
            kl_loss += F.kl_div(qs[i].log(), p, reduction='batchmean')
            # kl_loss += F.kl_div(qs[i].log(), model.target_distribution(qs[i]), reduction='batchmean')

        if l < kl_max:
            l = kl_step * epoch_num
        else:
            l = kl_max
        kl_loss *= l
        # -----------------------------------------------------------------------

        loss = re_loss + kl_loss + consis_loss + re_feat_loss

        optimizer_ge.zero_grad()
        loss.backward()
        optimizer_ge.step()

    # ============================ 4.evaluation ==========================
        if epoch_num % update_interval == 0:  # [1,3,5]
            model.eval()
            with torch.no_grad():
                # update_interval
                zs = []
                qs = []
                q = 0.
                for v in range(graph_num):
                    _, z, tmp_q, _ = model(features, adjs[v], view=v)
                    zs.append(weights[v] * z)
                    qs.append(tmp_q)

            z = torch.cat(zs, dim=-1)
            q = model.predict_distribution(z, -1)
            kmeans = KMeans(n_clusters=class_num, n_init=20)
            res2 = kmeans.fit_predict(z.data.cpu().numpy())
            nmi, acc, ari, f1 = eva(y, res2, str(epoch_num) + 'Kz')

            # for i in range(graph_num):
            #     res1 = kmeans.fit_predict(zs[i].data.cpu().numpy())
            #     _, _, _, _ = eva(y, res1, str(epoch_num) + 'K'+str(i))

            for i in range(graph_num):
                print('view:', str(i), np.around(model.GraphEnc[i].la.data.cpu().numpy(), 3))
            print(weights)
            model.train()
    # ======================================= 5. postprocess ======================================
        print(#'Epoch:{}'.format(epoch_num),
              'bad_count:{}'.format(bad_count),
              'kl:{:.4f}'.format(kl_loss),
              'consis:{:4f}'.format(consis_loss),
              'rec:{:.4f}'.format(re_loss.item()),
              're_feat:{:.4f}'.format(re_feat_loss.item()),
              end='\n')

        if nmi > best_nmi:
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
            best_epoch = epoch_num
            if loss < best_loss:
                best_loss = loss
            print('saving model epcoh:{}'.format(epoch_num))
            torch.save({'state_dict':model.state_dict(),
                        'weights': weights}, 'samgc_{}.pkl'.format(dataset))
            bad_count = 0
        else:
            bad_count += 1

        print('best acc:{}, best nmi:{}, best ari:{}, best f1:{},best loss:{}, bestepoch:{}'.format(
            best_acc, best_nmi, best_ari, best_f1, best_loss, best_epoch))
        print()

        if bad_count >= patience:
            print('complete training, best acc:{}, best nmi:{}, best ari:{}, best f1:{},best loss:{}, bestepoch:{}'.format(
                best_acc, best_nmi, best_ari, best_f1, best_loss.item(), best_epoch))
            break


# ============================================== Test =====================================================
if not train:
    model_name = args.model_name
else:
    model_name = 'samgc_{}.pkl'.format(dataset)
print('Loading model:{}...'.format(model_name))
best_model = torch.load(model_name, map_location=features.device)
weights = best_model['weights']
print(weights)
state_dict = best_model['state_dict']
model.load_state_dict(state_dict)
print('Evaluating....')
with torch.no_grad():
    # update_interval
    zs = []
    qs = []
    q = 0.
    for v in range(graph_num):
        _, z, tmp_q, _ = model(features, adjs[v], view=v)
        zs.append(z)
        qs.append(tmp_q)

z = torch.cat([zs[i] * weights[i] for i in range(graph_num)], dim=-1)
kmeans = KMeans(n_clusters=class_num, n_init=100)
res2 = kmeans.fit_predict(z.data.cpu().numpy())
nmi, acc, ari, f1 = eva(y, res2, str('eva:') + 'Kz')

# for i in range(graph_num):
#     res1 = kmeans.fit_predict(zs[i].data.cpu().numpy())
#     eva(y, res1, str('eva:') + 'K' + str(i))

print('Results: acc:{},  nmi:{},  ari:{},  f1:{}, '.format(
            acc, nmi, ari, f1))
