""" 
  Training of the module A to obtain the translation model, and output the model as model.pth
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import multiGATEncoder,multiGCNEncoder
from utils import  gen_adj_mat_tensor
from os.path import isfile
import datetime


cuda = True if torch.cuda.is_available() else False
# cuda = False
dataType = ['RNA' ,'miRNA','CN','meth'] 
data_fold1 ='./partial_samples_fea/all/' # training data, #samples=3025



def prepare_train_data(data_type): 
    file_input = data_fold1
    fea_save_file =file_input + data_type + '.fea'
    if isfile(fea_save_file):
        df = pd.read_csv(fea_save_file, sep='\t', header=0, index_col=0)
    tensor = torch.FloatTensor(df.values.astype(float))
    return tensor

# if #clusters is unknown, using 6 as a crude estimate for the number of clusters observed in cancer datasets
def gen_trte_adj_mat(data_tr,num_class=6):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_train_list.append(gen_adj_mat_tensor(data_tr, num_class, adj_metric)) 
    return adj_train_list 



omic_dic ={} 
graph_dic ={} 
num_class = 32 
############## load data/graph #####################
for type in dataType:
    omic_dic[type] = prepare_train_data(type)
    graph_dic[type] = gen_trte_adj_mat(omic_dic[type], num_class)

fea_protein_file = data_fold1 + 'protein.fea'
df = pd.read_csv(fea_protein_file, sep='\t', header=0, index_col=0) 
protein = torch.FloatTensor(df.values.astype(float))
####################################################

############## Initialize a model with 4 GCNs, each with two layers of GCN-layer #####################
dim_list = [omic_dic[x].shape[1] for x in dataType] # dimensions of each omics [3217, 383, 3105, 3139]
dim_hid_list = [800,200,800,800] # different hidden layer dimensions for different omics
dim_final = 455 # The dimension of the final protein

dropout = 0.1   #0.1-0.5
lr_e = 5e-4
npatient = omic_dic['RNA'].shape[0]
model = multiGCNEncoder(dim_list, dim_hid_list,dim_final, dropout,npatient)
optim = torch.optim.Adam(
            list(model.parameters()),lr=lr_e) 

################  GPU ############################
if cuda:
    model.cuda()
for type in dataType:
    if cuda:
        omic_dic[type]=omic_dic[type].cuda()
        graph_dic[type]=graph_dic[type][0].cuda()
        protein=protein.cuda()
    else:
        graph_dic[type]=graph_dic[type][0]


############## training ###########################
epoch = 175
loss_list = []
torch.autograd.set_detect_anomaly(True)
criterion = torch.nn.MSELoss()# MSE-loss
model.train()
print("\n############## training ...##############\n")

for i in range(epoch):
    i_loss = 0
    optim.zero_grad()
    z = model(RNAseq=omic_dic['RNA'], dnam=omic_dic['meth'], cn=omic_dic['CN'], mic=omic_dic['miRNA'], adj_1=graph_dic['RNA'], adj_2=graph_dic['meth'], adj_3=graph_dic['CN'], adj_4=graph_dic['miRNA'])
    # z denotes predicted protein
    i_loss = criterion( z , protein)
    i_loss = torch.sqrt(i_loss)
    i_loss.backward()
    optim.step()
    loss_list.append(i_loss.tolist())
print("############## end ##############")

print(loss_list)
pd.DataFrame(z.detach().cpu().numpy()).to_csv(data_fold1+'protein_new.fea',sep='\t')
# Save predicted proteins for visualization

plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"{time_str}"
plt.savefig('./loss_png/'+ file_name)

torch.save(model.state_dict(), 'model.pth')
