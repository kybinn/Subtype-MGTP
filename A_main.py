'''
Import the trained translation model and apply it to 3085 omics samples to generate corresponding proteomics.
'''

import numpy as np
import pandas as pd
import torch
from models import multiGCNEncoder
from utils import  gen_adj_mat_tensor
from os.path import isfile

cuda = True if torch.cuda.is_available() else False
dataType = ['RNA' ,'miRNA','CN','meth'] 
data_fold ='./all_samples_fea/all/'      #samples = 3851


def prepare_train_data(data_type): 
    file_input = data_fold
    fea_save_file =file_input + data_type + '.fea'
    if isfile(fea_save_file):
        df = pd.read_csv(fea_save_file, sep='\t', header=0, index_col=0)   
    tensor = torch.FloatTensor(df.values.astype(float))
    return tensor


def gen_trte_adj_mat(data_tr,num_class=6):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_train_list.append(gen_adj_mat_tensor(data_tr, num_class, adj_metric))

    return adj_train_list 


omic_dic ={} 
graph_dic ={} 
num_class = 32 

############## load data/graph/model #####################
print("\n############## load model ...##############\n")
for type in dataType:
    omic_dic[type] = prepare_train_data(type)
    graph_dic[type] = gen_trte_adj_mat(omic_dic[type], num_class)

dim_list = [omic_dic[x].shape[1] for x in dataType]
dim_hid_list = [800,200,800,800]
dim_final = 455
dropout = 0.1
npatient = omic_dic['RNA'].shape[0]
model = multiGCNEncoder(dim_list, dim_hid_list,dim_final, dropout,npatient)
model.load_state_dict(torch.load('model.pth'))

if cuda:
    model.cuda()
for type in dataType:
    if cuda:
        omic_dic[type]=omic_dic[type].cuda()
        graph_dic[type]=graph_dic[type][0].cuda()
    else:
        graph_dic[type]=graph_dic[type][0]

############## generate protein #######################
print("\n############## computing start ...##############\n")

protein = model(RNAseq=omic_dic['RNA'], dnam=omic_dic['meth'], cn=omic_dic['CN'], mic=omic_dic['miRNA'], adj_1=graph_dic['RNA'], adj_2=graph_dic['meth'], adj_3=graph_dic['CN'], adj_4=graph_dic['miRNA'])

print("\n############## computing end ...##############\n")

############## save protein data #######################
pd.DataFrame(protein.detach().cpu().numpy()).to_csv(data_fold+'protein_generate.fea',sep='\t')
# Predicted protein data of #3851 samples
