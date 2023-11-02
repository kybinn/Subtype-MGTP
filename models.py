import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import normalize
from torch_geometric.nn import GCNConv
from torch.nn.parameter import Parameter


# GCN layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)) 
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
          
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight) 
        output = torch.sparse.mm(adj, support) 
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# GCN model
class GraphGCN(nn.Module):
    def __init__(self, nfeat, nhid, dim_final, dropout,npatient):
        super(GraphGCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(nfeat, nhid)        #GCN1
        self.gc2 = GraphConvolution(nhid , dim_final)   #GCN2

    def forward(self, x, adj):
        z = self.gc1(x, adj)
        z = F.leaky_relu(z, 0.25)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        z = F.leaky_relu(z, 0.25)
        return z


################ PART1 : Used for Module A
class multiGCNEncoder(nn.Module):
    def __init__(self, dim_list, dim_hid_list, dim_final, dropout,npatient):
        super(multiGCNEncoder, self).__init__()
        self.dropout = dropout
        self.RNAseq = GraphGCN(dim_list[0], dim_hid_list[0], dim_final, dropout,npatient)
        self.miRNA = GraphGCN(dim_list[1], dim_hid_list[1], dim_final, dropout,npatient)
        self.CN = GraphGCN(dim_list[2], dim_hid_list[2], dim_final, dropout,npatient)
        self.meth = GraphGCN(dim_list[3], dim_hid_list[3], dim_final, dropout,npatient)

        
    def forward(self, RNAseq, dnam, cn, mic, adj_1, adj_2, adj_3, adj_4):
        x_r = self.RNAseq(RNAseq, adj_1)
        x_d = self.meth(dnam, adj_2)
        x_c = self.CN(cn, adj_3)
        x_mic = self.miRNA(mic,adj_4)
        z = (x_r+ x_d+ x_c + x_mic) / 4
        return z



################ PART2: Used for Module B
class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(in_channels, out_channels*2)     #GCN1
        self.gc2 = GraphConvolution(out_channels*2 , out_channels)   #GCN2

    def forward(self, x, adj):
        z = self.gc1(x, adj)
        z = F.leaky_relu(z, 0.25)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        z = F.leaky_relu(z, 0.25)
        return z

class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(in_channels, in_channels*2)     #GCN1
        self.gc2 = GraphConvolution(in_channels*2 , out_channels)   #GCN2

    def forward(self, x, adj):
        z = self.gc1(x, adj)
        z = F.leaky_relu(z, 0.25)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        z = F.leaky_relu(z, 0.25)
        return z

class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, num_sample: int):
        super(Model, self).__init__()
        self.n = num_sample
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(self.n, self.n, dtype=torch.float32), requires_grad=True)


    def forward(self, x, edge_index):
        H = self.encoder(x, edge_index) 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        C_diag = torch.diag(torch.diag(self.Coefficient)).to(device)
        Coefficient = self.Coefficient
        CH = torch.matmul(Coefficient, H) 
        X_ = self.decoder(CH, edge_index) 
        return H, CH, Coefficient, X_



