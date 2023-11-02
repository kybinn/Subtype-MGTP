'''
Deep Subspace Contrastive Clustering Module
This module takes the predicted protein data of 3851 samples as input. 
'''

import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from os.path import isfile
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from models import Encoder, Decoder, Model,DAEGC
from utils import  gen_adj_mat_tensor,get_M
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans



def gen_trte_adj_mat(data_tr,num_class=6):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_train_list.append(gen_adj_mat_tensor(data_tr, num_class, adj_metric)) 

    return adj_train_list 

def target_distribution(q):
    p = q**2 / q.sum(0)
    return (p.t() / p.sum(1)).t()



class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while stop == False:
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp

def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
    f = lambda x: torch.exp(x / 1)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def instanceloss(z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
    l1 = semi_loss(z1, z2)
    l2 = semi_loss(z2, z1)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret

def soft_assign(z, n_clusters, alpha=1.0):
    mu = torch.nn.Parameter(torch.Tensor(n_clusters, z.size(1)))  
    device = z.device  
    mu = mu.to(device)  
    torch.nn.init.xavier_uniform_(mu)  
    diff = z.unsqueeze(1) - mu  
    squared_distance = torch.sum(diff ** 2, dim=2)  
    q = 1.0 / (1.0 + squared_distance / alpha) 
    q = q ** ((alpha + 1.0) / 2.0) 
    q = (q.t() / torch.sum(q, dim=1)).t() 
    return q

def cluster_loss(p, q):
    def kld(target, pred):
        return torch.sum(torch.sum(target*torch.log(target/(pred+1e-8)), dim=-1))
    kldloss = kld(p, q)
    # kldloss = F.kl_div(q, p)
    return kldloss 




def train(x,adj,n_clusters):
    model.train()
    optimizer.zero_grad()
    H, CH, Coefficient, X_ = model(x, adj) 
  
    rec_loss = torch.sum(torch.pow(x - X_, 2))
    #  loss of reconstruction
    loss_instance = instanceloss(H, CH)
    #  contrastive self-expression loss 

    loss_coef = torch.sum(torch.pow(Coefficient, 2))
    #  regularization term
  
    loss = 1.0 * loss_instance + 1.0 * loss_coef + 0.01 * rec_loss
    loss.backward()
    optimizer.step()
    return loss_instance.item(), loss_coef.item(),  rec_loss.item(), loss.item(), Coefficient,H

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest='type', default="ALL", help="cancer type: BRCA, GBM")
    parser.add_argument("-learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("-epochs", type=int, default=600, help="number of epochs")
    parser.add_argument("-weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("-tau", type=float, default=1, help="temperature")
    parser.add_argument("-n", dest='cluster_num', type=int, default=-1, help="cluster number")
    args = parser.parse_args()


    # args.type='KIRC'
    cancer_type = args.type
    cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
                   'LUAD': 3,'SKCM': 4, 
                    'STAD': 3, 'UCEC': 4, 'UVM': 4}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    fea_tmp_file = './all_samples_fea/' + cancer_type + '/protein.fea' 
    if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
    elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
    
    print("python main.py -t"+ cancer_type)

    if isfile(fea_tmp_file):
        df_new = pd.read_csv(fea_tmp_file, sep=',', header=0, index_col=0)   
    
    protein= df_new
    data = torch.FloatTensor(df_new.values.astype(float))
    graph = gen_trte_adj_mat(data, args.cluster_num)[0]
    data = data.to(device)
    graph = graph.to(device)

    criterion_instance = InstanceLoss(len(df_new), args.tau, device).to(
        device)

    encoder = Encoder(df_new.shape[1], 150, dropout=0.1).to(device)
    decoder = Decoder(150, df_new.shape[1], dropout=0.1).to(device)
    model = Model(encoder, decoder, len(df_new)).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    
    alpha = max(0.4 - (args.cluster_num - 1) / 10 * 0.1, 0.1)
    

    acclist = []
    nmilist = []
    arilist = []

    for epoch in range(1, args.epochs+1):       
        loss_instance, loss_c, rec_loss, loss, C,H = train(data,graph,n_clusters=args.cluster_num)
        print(f'Epoch={epoch:03d}, loss={loss:.4f}, loss_instance = {loss_instance:.4f}, rec_loss = {rec_loss:.4f}, loss_c={loss_c:.4f}')

    C = C.detach().cpu().numpy()
    commonZ = thrC(C, alpha) 
    # Threshold processing helps achieve sparse representation, extract important features, and filter noise
  
    y_pred, _ = post_proC(commonZ, args.cluster_num)
    # y_pred represents the clustering result
    
    choice='cluster'
    protein[choice]=y_pred  
    score = silhouette_score(protein, y_pred)
    print(f'silhouette score= {score:.2f}')
           
    X = protein.loc[:, [choice]] 
    out_file = './analysis/results/' + cancer_type + '.'+ choice
    X.to_csv(out_file, header=True, index=True, sep='\t')


    print("=================== Final =====================")
    

    
