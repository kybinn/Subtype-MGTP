import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())



def graph_from_dist_tensor(dist, num_class, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    
    k=int(dist.shape[0]/num_class+1)  #k= #samples/#num_class+1, to leave 0 on the diagonal
    new_matrix = np.ones_like(dist)
    for idx, row in enumerate(dist):
        kth_smallest = np.partition(row, k-1)[:k]
        new_row = np.where(np.isin(row, kth_smallest), row, 1)
        new_matrix[idx] = new_row

    return new_matrix



def gen_adj_mat_tensor(data, num_class, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)   
    g = graph_from_dist_tensor(dist, num_class, self_dist=True) 

    if metric == "cosine":
        adj = 1-g    
    else:
        raise NotImplementedError
    
    diag_idx = np.diag_indices(adj.shape[0])
    adj[diag_idx[0], diag_idx[1]] = 0

    row_sums = adj.sum(axis=1)                          
    row_sums_expanded = np.expand_dims(row_sums, axis=1)
    adj = adj / row_sums_expanded      
    # adj_T = adj.transpose(0,1)            
    adj_T=adj.T
    adj=adj+adj_T                         

    adj = F.normalize(torch.from_numpy(adj), p=1)  

    I = torch.eye(adj.shape[0])
    # if cuda:
    #     I = I.cuda()
    adj=adj+I
    adj = to_sparse(adj)    
    return adj


def get_M(adj):
    adj_numpy = adj.to_dense().cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)
