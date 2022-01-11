
import logging
import anndata as ad
import os
from scipy.sparse import csc_matrix
import sys
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from torch.serialization import save

# addition package 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch.nn as nn 
import torch.nn.functional as F
import torch
from torch.nn.modules import flatten
from torch.nn.modules.activation import ReLU
import optuna 

# ======================= MMD loss ==================== #
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) 
   
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
   
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
   
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) 
 
def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,    
                                kernel_num=kernel_num,  
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size] # Source<->Source
    YY = kernels[batch_size:, batch_size:] # Target<->Target
    XY = kernels[:batch_size, batch_size:] # Source<->Target
    YX = kernels[batch_size:, :batch_size] # Target<->Source
    loss = torch.mean(XX + YY - XY -YX) 
                                                                            
    return loss
# ============https://www.codenong.com/cs105876584/============== # 

#================ model and loss function =================== #
class EncoderBias(nn.Module):
    def __init__(self, input_dim1, input_dim2, batch_feature, latent_dim,  bias=False):
        """[summary]
        Args:
            input_dim1 ([type]): [mod1 dimemsion]
            input_dim2 ([type]): [mod2 dimemsion]
            batch_feature ([type]): [batch dimemsion]
            latent_dim ([type]): [latent dimemsion]
            bias (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.dim1_encoder_weight = nn.Parameter(torch.randn(input_dim1, latent_dim))
        self.dim2_encoder_weight = nn.Parameter(torch.randn(input_dim2, latent_dim))
        self.batch_encoder_weight   = nn.Parameter(torch.randn(batch_feature, latent_dim))

        if bias:
            self.dim1_bias = nn.Parameter(torch.randn(latent_dim))
            self.dim2_bias = nn.Parameter(torch.randn(latent_dim))
        else:
            self.dim1_bias = 0
            self.dim2_bias = 0
            
    # x : [mod1_value, mod2_value, batch_one_hot]
    # dim1_encoder: (batch size, 100), (batch size, 100), as reuslt of first hidden layer
    def forward(self, x):
        self.dim1_weight = torch.cat([self.dim1_encoder_weight, self.batch_encoder_weight], dim=0) # concatenate by default
        self.dim2_weight = torch.cat([self.dim2_encoder_weight, self.batch_encoder_weight], dim=0)
        x1 = x[0] # x1
        x2 = x[1] # x2
        x3 = x[2] # batch
        dim1_input = torch.cat([x1,x3],dim=1) # mod1 value concat with batch information 
        dim2_input = torch.cat([x2,x3],dim=1)
        dim1_encoder= dim1_input @ self.dim1_weight + self.dim1_bias # linear operation 
        dim2_encoder= dim2_input @ self.dim2_weight + self.dim2_bias # 100 dimension
        return dim1_encoder, dim2_encoder

class DecoderBias(nn.Module):
    def __init__(self, dim1_batch, latent_dim, bias=False):
        # as EncoderBias
        super().__init__()
        self.dim1_latent_decoder = nn.Parameter(torch.randn(latent_dim,latent_dim))
        self.dim2_latent_decoder = nn.Parameter(torch.randn(latent_dim,latent_dim))
        self.batch_decoder_weight = nn.Parameter(torch.randn(dim1_batch,latent_dim))

        if bias:
            self.dim1_bias = nn.Parameter(torch.randn(latent_dim))  
            self.dim2_bias = nn.Parameter(torch.randn(latent_dim))  
        else:
            self.dim1_bias = 0
            self.dim2_bias = 0
    # x
    # x[0]: modality 1 latent space embed without batch one hot
    # x[1]: modality 2 latent space embed without batch one hot
    # x[2]: batch_one_hot
    def forward(self, x):
        self.dim1_decoder_weight = torch.cat([self.dim1_latent_decoder, self.batch_decoder_weight],dim=0)
        self.dim2_decoder_weight = torch.cat([self.dim2_latent_decoder, self.batch_decoder_weight],dim=0)
        dim1_latent = x[0]
        dim2_latent = x[1]
        batch  = x[2]
        dim1_input = torch.cat([dim1_latent, batch],dim=1)
        dim2_input = torch.cat([dim2_latent, batch],dim=1)
        dim1_output = dim1_input @ self.dim1_decoder_weight + self.dim1_bias
        dim2_output = dim2_input @ self.dim2_decoder_weight + self.dim2_bias
        return dim1_output,dim2_output
    
"""[summary]
A autoencoder moddel in step 1.
Inputs are original gex and adt wiith batch one hot vector.
To get latent space embeddings of gex and adt considered as 
new expressions without batch effect ready for MLP model training.
"""
class AutoEncoder(nn.Module):
    def __init__(self, input_dim1, input_dim2, dim1_batch, latent_dim):
        """[summary]
        Args:
            input_dim1 ([type]): [mod1 dimemsion]
            input_dim2 ([type]): [mod2 dimemsion]
            dim1_batch ([type]): batch feature dimesion
            latent_dim ([type]): [latent dimemsion]
        """
        super().__init__()
        self.encoder = EncoderBias(input_dim1, input_dim2, dim1_batch, latent_dim)
        self.latent1 = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
        )
        self.latent2 = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
        )
        self.batch_decoder = DecoderBias(dim1_batch, latent_dim)
        self.dim1_decoder = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, input_dim1),
            nn.ReLU()
        )
        self.dim2_decoder = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, input_dim2),
            nn.ReLU()
        )
    
    def get_encoder(self,x):
        batch = x[2]
        dim1_encoder, dim2_encoder = self.encoder(x) # shape (batch size,100) hidden layer 
        dim1_encoder_latent = self.latent1(dim1_encoder) # hidden layer 2 latent layer of mod1
        dim2_encoder_latent = self.latent2(dim2_encoder) # hidden layer 2 latent layer of mod2
        return dim1_encoder_latent,dim2_encoder_latent

    def forward(self,x):
        # x : [mod1_value, mod2_value, batch_one_hot]
        batch = x[2]
        dim1_encoder, dim2_encoder = self.encoder(x) # shape (batch size,100), as output of first hidden layer 
        dim1_encoder_latent = self.latent1(dim1_encoder) # hidden layer 2, latent layer of mod1
        dim2_encoder_latent = self.latent2(dim2_encoder) # hidden layer 2, latent layer of mod2
        # dim1_encoder_latent_with_batch = torch.cat([dim1_encoder_latent,batch],dim=1) # latent representation + batch information mod1
        # dim2_encoder_latent_with_batch = torch.cat([dim2_encoder_latent,batch],dim=1) # latent representation + batch information mod2
        dim1_latent_decoder, dim2_latent_decoder = self.batch_decoder(
                                                    [dim1_encoder_latent,
                                                    dim2_encoder_latent,
                                                    batch]) # latent layer 2, hidden layer with batch 
        reconstruct_dim1 = self.dim1_decoder(dim1_latent_decoder) # output layer 
        reconstruct_dim2 = self.dim2_decoder(dim2_latent_decoder)
        return reconstruct_dim1, reconstruct_dim2,dim1_encoder_latent,dim2_encoder_latent
    
    # only 1 encoder
    # def forward(self,x):
    #     # x : [mod1_value, mod2_value, batch_one_hot]
    #     batch = x[2]
    #     dim1_encoder, dim2_encoder = self.encoder(x) # shape (batch size,100), as output of first hidden layer 
    #     dim1_encoder_latent = self.latent1(dim1_encoder) # hidden layer 2, latent layer of mod1
    #     # dim2_encoder_latent = self.latent2(dim2_encoder) # hidden layer 2, latent layer of mod2
    #     dim1_latent_decoder, dim2_latent_decoder = self.batch_decoder(
    #                                                 [dim1_encoder_latent,
    #                                                 dim1_encoder_latent,
    #                                                 batch]) # latent layer 2, hidden layer with batch 
    #     reconstruct_dim1 = self.dim1_decoder(dim1_latent_decoder) # output layer 
    #     reconstruct_dim2 = self.dim2_decoder(dim2_latent_decoder)
    #     return reconstruct_dim1, reconstruct_dim2, dim1_encoder_latent, dim1_encoder_latent

def double_autoencoder_loss(pred, target, weights=(0.5, 0.5, 1),kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # target is x
    dim1_rec_loss = (pred[0] - target[0]).pow(2).mean().sqrt()
    dim2_rec_loss = (pred[1] - target[1]).pow(2).mean().sqrt() 
    mmd_loss      = mmd(pred[2],pred[3],kernel_mul,kernel_num,fix_sigma)
    return weights[0] * dim1_rec_loss + weights[1] * dim2_rec_loss + weights[2] * mmd_loss

#================ step2 : mlp ============================== # 
class LatentMLP(nn.Module):
    def __init__(self, latent_dim, hidden_dim=50):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,latent_dim),
            nn.ReLU(),
        )
        
    def forward(self,x):
        x = self.mlp(x)
        return x
    
def mlp_loss(pred,target):
    return (pred-target).abs().sum(dim=-1).mean()
# ========================================================== #

# ============== step3 1: finetune batch effect ============== # 

class Mod1AutoEncoderFinetune(nn.Module):
    # as before
    def __init__(self,input_dim1,dim1_batch,latent_dim):
        super().__init__()
        self.dim1_encoder_weight = nn.Parameter(torch.randn(input_dim1,latent_dim))
        self.test_batch_encoder_weight   = nn.Parameter(torch.randn(dim1_batch,latent_dim))
        
        self.latent1 = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
        )
        self.dim1_latent_decoder = nn.Parameter(torch.randn(latent_dim,latent_dim))
        self.test_batch_decoder_weight = nn.Parameter(torch.randn(dim1_batch,latent_dim))
        
        self.dim1_decoder = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, input_dim1),
            nn.ReLU()
        )   
    
    def get_encoder(self, x):
        # as forward
        value, batch = x
        self.dim1_weight = torch.cat([self.dim1_encoder_weight, self.test_batch_encoder_weight], dim=0)
        self.dim1_decoder_weight = torch.cat([self.dim1_latent_decoder, self.test_batch_decoder_weight], dim=0)
        dim1_input  = torch.cat([value,batch],dim=1)
        dim1_encoder= dim1_input @ self.dim1_weight 
        dim1_latent = self.latent1(dim1_encoder)
        return dim1_latent
    
    def forward(self, x):
        value, batch = x # x is constructed by value and ont hot encoding 
        # encoder hidden layer 
        self.dim1_weight = torch.cat([self.dim1_encoder_weight, self.test_batch_encoder_weight],dim=0)
        # decoder hidden layer
        self.dim1_decoder_weight = torch.cat([self.dim1_latent_decoder, self.test_batch_decoder_weight],dim=0)
        
        dim1_input  = torch.cat([value,batch],dim=1)
        dim1_encoder= dim1_input @ self.dim1_weight  # encoder hidden layer 
        dim1_latent = self.latent1(dim1_encoder)     # hidden layer 2 latent space 
        dim1_decoder_input = torch.cat([dim1_latent,batch],dim=1)
        dim1_decoder = dim1_decoder_input @ self.dim1_decoder_weight # decoder hidden layer 
        decoder = self.dim1_decoder(dim1_decoder)    # decoder hidden layer 2 output space 
        return decoder

def rec_loss(pred,target):
    return (pred - target).pow(2).mean().sqrt()

# ============= step3 2: predict =========================== #

class Mod2Predict(nn.Module):
    def __init__(self,input_dim2,dim1_batch,latent_dim):
        super().__init__()
        self.dim2_latent_decoder = nn.Parameter(torch.randn(latent_dim,latent_dim))
        self.test_batch_decoder_weight = nn.Parameter(torch.randn(dim1_batch,latent_dim))
        self.dim2_decoder_weight = torch.cat([self.dim2_latent_decoder,self.test_batch_decoder_weight],dim=0)
        self.dim2_decoder = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, input_dim2),
            nn.ReLU()
        )
        
    def forward(self,x):
        # x is mlp predict mod2 latent space and one hot
        dim2_encoder_latent,batch = x
        dim2_encoder_latent_with_batch = torch.cat([dim2_encoder_latent,batch],dim=1)
        dim2_latent_decoder = dim2_encoder_latent_with_batch @ self.dim2_decoder_weight
        reconstruct_dim2 = self.dim2_decoder(dim2_latent_decoder)
        return reconstruct_dim2

# ============ step3 parameter setting ===================== # 

def parameter_modify(ae_static_dict):
    modify = ['encoder','batch_decoder']
    name_list = list(ae_static_dict.keys())
    for i in name_list:
        if i.split(".")[0] in modify:
            ae_static_dict[".".join(i.split(".")[1:])] = ae_static_dict[i]

def Mod1AutoEncoderFinetuneParameterSetting(model,ae_static_dict):
    model.load_state_dict(ae_static_dict,strict=False)
    for name, param in model.named_parameters():
        if "test" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def Mod2PredictParameterSetting(mod2_model,mod1_model,ae_static_dict):
    ae_static_dict["test_batch_decoder"] = mod1_model.state_dict()['test_batch_decoder_weight']
    mod2_model.load_state_dict(ae_static_dict,strict=False)

# ============ dataset & dataloader  ===================== # 
class pairDataset(torch.utils.data.Dataset):
    
    def __init__(self,*pairs,obs=list(range(1000000))):
        super().__init__()
        self.pairs = pairs
        self.obs = obs # batch information 
    
    def __len__(self,):
        return self.pairs[0].size(0)

    def __getitem__(self,index):
        return [pair[index] for pair in self.pairs],self.obs[index]
# =======================tsne======================== #

def tnse_plot_embedding(sample_data, sample_label, title='t-SNE embedding',savepath=""):
    # sample_data : numpy ndarray 
    # sample_label: numpy ndarray 
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(sample_data)
    label= sample_label
    
    #x_min, x_max = np.min(data, 0), np.max(data, 0)
    #data = (data - x_min) / (x_max - x_min)
    data_index_col = {i:[idx for idx,j in enumerate(label) if j == i] for i in set(label)}
    
    data_col = {i:data[data_index_col[i]] for i in data_index_col.keys()}
    plt.figure()
    ax = plt.subplot(111)
    length = len(data_col.keys())
    for i in data_col.keys():
        ax.scatter(data_col[i][:, 0], data_col[i][:, 1],s=0.1,color=plt.cm.Set1(i / length),label=str(i))
    ax.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(savepath+".jpg")

def sample_data_from_total(total_data,total_label,sample_rate=0.5):
    # total_data : numpy ndarray 
    # total_label: numpy ndarray 
    idx = list(range(total_data.shape[0]))
    sample_len=int(len(idx) * sample_rate)
    sample_idx=np.random.choice(idx,size=sample_len,replace=False)
    return total_data[sample_idx], total_label[sample_idx]

def sample_data_2_tnse_plot(total_data,total_label,sample_rate=0.5,title='t-SNE embedding',savepath=""):
    sample_data,sample_label = sample_data_from_total(total_data,total_label,sample_rate=sample_rate)
    tnse_plot_embedding(sample_data,sample_label,title=title,savepath=savepath)

# ========================= optuna ========================== #
def register_search_space_by_parameters(parameters,prefix = "search_",postfix = "_list"):
    """[parameters has some keys starting with 'search'] 
    Args:
        search_space ([type]): [description]
        parameters ([type]): [description]
    """
    search_space = {}
    for key in parameters.keys():
        if key.startswith(prefix) and parameters[key]:
            search_space[key[len(prefix):]] = parameters[key[len(prefix):]+postfix]
    return search_space

def get_parameters_by_trial_or_not(parameters,trial,prefix = "search_",postfix = "_list"):
    return_dict = {}
    search_space = register_search_space_by_parameters(parameters,prefix,postfix)
    for key in search_space:
        return_dict[key] = trial_auto_generator(search_space[key],trial,key)

    for key in parameters.keys():
        if key not in return_dict:
            return_dict[key] = parameters[key]
    return return_dict

def trial_auto_generator(parameter_list,trial,name):
    if type(parameter_list[0]) == int:
        return trial.suggest_int(name,min(parameter_list),max(parameter_list))
    elif type(parameter_list[0]) == float:
        return trial.suggest_float(name,min(parameter_list),max(parameter_list))
    else:
        return trial.suggest_categorical(name,parameter_list)

def prity_print_dict(dict_):
    res = []
    content = ""
    for i in dict_.keys():
        content = ">>>> {:50}: \t {}".format(i , str(dict_[i]))
        res.append(content)
    return "\n".join(res)

# ================================================================ # 

debug = True

if debug:
    path = "output/datasets_phase1v2/predict_modality/openproblems_bmmc_cite_phase1v2_mod2/"
    logging.basicConfig(level=logging.INFO,filename="./log/log.log",filemode='w')
else:
    path = sys.argv[1]
    logging.basicConfig(level=logging.INFO)



pathlist = os.listdir(path)
if "sample_data" not in path:
    train_mod1_path = path + [i for i in pathlist if "output_train_mod1" in i ][0]
    train_mod2_path = path + [i for i in pathlist if "output_train_mod2" in i][0]
    test_mod1_path  = path + [i for i in pathlist if "output_test_mod1" in i][0]
    test_mod2_path  = path + [i for i in pathlist if "output_test_mod2" in i][0]
else:
    train_mod1_path = path + [i for i in pathlist if "train_mod1" in i ][0]
    train_mod2_path = path + [i for i in pathlist if "train_mod2" in i][0]
    test_mod1_path  = path + [i for i in pathlist if "test_mod1" in i][0]
    test_mod2_path  = path + [i for i in pathlist if "test_mod2" in i][0]
# test_mod1_path = path + [i for i in pathlist if "output_test_mod1" in i][0]

output_path_dir = "output/predictions/predict_modality/"+path.split("/")[-2]+"/"
if not debug:
    os.mkdir(output_path_dir)
output_path = output_path_dir +path.split("/")[-2]+ ".output.h5ad"
par = {
    'input_train_mod1': train_mod1_path,
    'input_train_mod2': train_mod2_path,
    'input_test_mod1': test_mod1_path,
    'input_test_mod2' : test_mod2_path,
    'distance_method': 'minkowski',
    'output': output_path,
    'n_pcs': 50,
}


method_id = "python_starter_kit"

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])
input_test_mod2 = ad.read_h5ad(par['input_test_mod2'])

# TODO: implement own method

double_ae_loss_weight_list = [(0.5,0.5,1.0), (0.4,0.6,1.0), (0.3,0.7,1.0), (0.6,0.4,1.0), (0.7, 0.3, 1.0), (0.8,0.2,1.0), (0.9, 0.1, 1.0)]
# double_ae_loss_weight_list = [(0.7, 0.3, 1.0), (0.8,0.2,1.0), (0.9, 0.1, 1.0)]


parameters = {
    "search_double_ae_loss_weight":True,
    "double_ae_loss_weight_list":list(range(len(double_ae_loss_weight_list))),
    "double_ae_loss_weight":0,
    
}
search_space = register_search_space_by_parameters(parameters)
logging.info("\nsearch_space:\n"+prity_print_dict(search_space)+"\n\n")

unique_save_tag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) # 标识符号 不然不知道是那个文件产生的  

logging.info("\n\n\n ===========unique save tag {}============ \n\n\n".format(unique_save_tag))
def objective(trial):
    # hyperparameters 
    parameters_with_trail = get_parameters_by_trial_or_not(parameters,trial)
    num_epochs = 50  # train  double autoencoder 
    mlp_fit_epochs = 30 # train mlp 
    ae_learning_rate = 0.001 
    mlp_learning_rate= 0.001
    mod1ae_learning_rate=0.001
    test_ae_path="test_ae_model.pth"
    latent_space_dim= 100
    mlp_hidden_dim = 50
    latent_dim = 50
    batch_size = 320
    val_split_rate = 0.2
    num_mod1_epochs = 50 # test train
    batch = batch_size
    double_ae_loss_weight = double_ae_loss_weight_list[parameters_with_trail['double_ae_loss_weight']]
    # get description of ds  
    ishape,oshape= input_train_mod1.X.shape[1],input_train_mod2.X.shape[1]
    # model path 

    def get_score(x,ys):
        matrix = (x-ys).abs().pow(2).mean().sqrt().item()
        return matrix

    def total_train():
        logging.info("\n\n\n ============== train ============== \n\n\n")
        model_ae_path = '{}-{}ae.pth'.format(ishape,oshape)
        model_mlp_path = '{}-{}mlp.pth'.format(ishape,oshape)

        mod_obs = input_train_mod1.obs.batch.values.tolist() # get train input batch information 
        batch_dim = len(set(mod_obs)) # get length
        mod_obs_dict = {v:k for k,v in enumerate(set(mod_obs))} # map it into number
        logging.info("mod_obs_dict: "+str(mod_obs_dict))
        mod_obs = np.array([mod_obs_dict[i] for i in mod_obs]) 
        # test_obs = input_test_mod1.obs.batch.values.tolist()

        train_inputs = torch.from_numpy(np.array(input_train_mod1.X.toarray()))
        train_targets= torch.from_numpy(np.array(input_train_mod2.X.toarray()))


        sample_data_2_tnse_plot(train_inputs.numpy(),mod_obs,sample_rate=0.2,
                                title="source data sample 0.2 t-sne embedding",
                                savepath="log/souce_mod1_data_tsne")
        sample_data_2_tnse_plot(train_targets.numpy(),mod_obs,sample_rate=0.2,
                                title="mod1 source data sample 0.2 t-sne embedding",
                                savepath="log/souce_mod2_data_tsne")
        
        # split train val 
        idx = list(range(train_inputs.shape[0]))
        val_len = int(len(idx) * val_split_rate)
        val_idx = np.random.choice(idx,size=val_len,replace=False)
        train_idx = np.array([i for i in idx if i not in val_idx])
        
        
        train_obs = mod_obs
        train_ds = pairDataset(train_inputs[train_idx], train_targets[train_idx], obs=train_obs[train_idx])
        val_ds   = pairDataset(train_inputs[val_idx], train_targets[val_idx],obs=train_obs[val_idx])
        train_dl = DataLoader(train_ds, batch_size, shuffle=True,drop_last=False)
        val_dl = DataLoader(val_ds, batch_size, shuffle=False,drop_last=False)
        # get model and lossfn 
        model = AutoEncoder(ishape,oshape,batch_dim,latent_space_dim)
        loss_fn = double_autoencoder_loss
        logging.info('Start to build the model')
        opt = torch.optim.Adam(params=model.parameters(),lr=ae_learning_rate)
        model.cuda()

        def train(epoch):
            model.train()
            step = 0
            for q,y in train_dl:
                # Generate predictions
                q[0] = q[0].cuda()
                q[1] = q[1].cuda()
                y = F.one_hot(y,batch_dim).cuda()
                x = [q[0],q[1],y]
                pred = model(x)
                eloss = loss_fn(pred, x, weights=double_ae_loss_weight)
                loss = eloss 
                if step % 20 == 1:
                    logging.info("epoch {}; step: {}; loss {}: ".format(epoch,step,loss.item()))
                step += 1
                opt.zero_grad()
                loss.backward()
                opt.step()
                
        def validation():
            model.eval()
            step = 0
            total_loss = []
            logging.info("validation phrase ")
            for q,y in val_dl:
                # Generate predictions
                q[0] = q[0].cuda()
                q[1] = q[1].cuda()
                y = F.one_hot(y,batch_dim).cuda()
                x = [q[0],q[1],y]
                pred = model(x)
                eloss = loss_fn(pred, x, weights=double_ae_loss_weight)
                loss = eloss 
                total_loss.append(loss.item())
                step += 1
            mean_loss = sum(total_loss) / len(total_loss)
            logging.info("validation mean loss:  {}".format(mean_loss))
            return mean_loss
    
        def fit(epoches,early_stop=True):
            mean_loss = 99999999999999999999999
            for epoch in range(epoches):
                train(epoch)
                score = validation()
                if score < mean_loss:
                    mean_loss = score
                    torch.save(model.state_dict(),model_ae_path) # save cpu result 
            
        logging.info('Running Auto encoder prediction...')
        fit(num_epochs,False)
        model.load_state_dict(torch.load(model_ae_path))
        model.cpu()
        model.eval()
        torch.save(model.state_dict(),model_ae_path) # save cpu result 

        # step 2 train mlp  

        model.load_state_dict(torch.load(model_ae_path))
        model.cuda()
        model.eval()
        mlp = LatentMLP(latent_space_dim,mlp_hidden_dim)
        mlp.cuda()
        mlp_loss_fn = mlp_loss
        logging.info('Start to build the model')
        mlp_opt = torch.optim.Adam(params=mlp.parameters(),lr=mlp_learning_rate)


        
        def collect_ae_latent_representation():
            total_predict = []
            total_mod2_predict = []
            total_label = []
            for q,y in train_dl:
                q[0] = q[0].cuda()
                q[1] = q[1].cuda()
                total_label.append(y) # collect
                y = F.one_hot(y,batch_dim).cuda()
                x = [q[0],q[1],y] # construct model input 
                with torch.no_grad():
                    pred = model.get_encoder(x) # pred : [encode1,encoder2]
                total_predict.append(pred[0].cpu())
                total_mod2_predict.append(pred[1].cpu())
            total_predict = torch.cat(total_predict,dim=0)
            total_mod2_predict = torch.cat(total_mod2_predict,dim=0)
            total_label   = torch.cat(total_label, dim=0)
            return total_predict,total_mod2_predict,total_label
        
        
        def latent_representation_sample_tsne_plot():
            total_predict,total_mod2_predict,total_label = collect_ae_latent_representation()
            sample_data_2_tnse_plot(total_predict.numpy(),total_label.numpy(),
                                    sample_rate=0.2,
                                    title="ae embedding (sample rate 0.2) tsne embedding",
                                    savepath="./log/{}ae_embedding_mod1_tsne".format(unique_save_tag))
            sample_data_2_tnse_plot(total_mod2_predict.numpy(),total_label.numpy(),
                                    sample_rate=0.2,
                                    title="ae embedding (sample rate 0.2) tsne embedding",
                                    savepath="./log/{}ae_embedding_mod2_tsne".format(unique_save_tag))
        
        latent_representation_sample_tsne_plot()
        
        
        
        def mlp_train(epoch):
            mlp.train()
            step = 0
            for q,y in train_dl:
                # Generate predictions
                q[0] = q[0].cuda()
                q[1] = q[1].cuda()
                y = F.one_hot(y,batch_dim).cuda()
                x = [q[0],q[1],y] # construct model input 
                with torch.no_grad():
                    pred = model.get_encoder(x) # pred : [encode1,encoder2]
                mlp_pred = mlp(pred[0])
                eloss = mlp_loss_fn(mlp_pred,pred[1])
                loss = eloss 
                if step % 20 == 1:
                    logging.info(" mlp   epoch {}; step: {}; loss {}: ".format(epoch,step,loss.item()))
                step += 1
                mlp_opt.zero_grad()
                loss.backward()
                mlp_opt.step()

        def mlp_validation():
            mlp.eval()
            step = 0
            total_loss = []
            logging.info("mlp validation phase")
            for q,y in val_dl:
                # Generate predictions
                q[0] = q[0].cuda()
                q[1] = q[1].cuda()
                y = F.one_hot(y,batch_dim).cuda()
                x = [q[0],q[1],y] # construct model input 
                with torch.no_grad():
                    pred = model.get_encoder(x) # pred : [encode1,encoder2]
                mlp_pred = mlp(pred[0])
                eloss = mlp_loss_fn(mlp_pred,pred[1])
                loss = eloss 
                total_loss.append(loss.item())
            mean_loss = sum(total_loss) / len(total_loss)
            logging.info("mlp validation mean loss : {}".format(mean_loss))
            return mean_loss

        def mlp_fit(epochs):
            mean_loss_pre = 9999999999999999999999999999
            for epoch in range(epochs):
                mlp_train(epoch)
                score = mlp_validation()
                if score < mean_loss_pre:
                    torch.save(mlp.state_dict(),model_mlp_path)
                
        mlp_fit(mlp_fit_epochs)
        mlp.load_state_dict(torch.load(model_mlp_path))
        mlp.cpu()
        mlp.eval()
        torch.save(mlp.state_dict(),model_mlp_path) # save cpu result

    def total_test():
        logging.info("\n\n\n ============== test ============== \n\n\n")
        mod_obs = input_test_mod1.obs.batch.values.tolist()
        batch_dim = len(set(mod_obs))
        mod_obs_dict = {v:k for k,v in enumerate(set(mod_obs))}
        logging.info("test  mod batch dict   "+str(mod_obs_dict))
        mod_obs = np.array([mod_obs_dict[i] for i in mod_obs])
        # test_obs = input_test_mod1.obs.batch.values.tolist()
        model_ae_path = '{}-{}ae.pth'.format(ishape,oshape)
        model_mlp_path = '{}-{}mlp.pth'.format(ishape,oshape)
        # idx = range(input_train_mod1.X.shape[0])
        # val_len=int(len(idx) * 0.2)
        # val_idx=np.random.choice(idx,size=val_len,replace=False)
        # train_idx=[ i for i in idx if i not in val_idx]

        # test phase model apply  
        train_obs = mod_obs
        train_inputs = torch.from_numpy(np.array(input_test_mod1.X.toarray())) # 这里是为了方便 就没有改变量名  
        test_len = train_inputs.shape[0]
        train_targets= torch.from_numpy(np.array(input_test_mod2.X.toarray()))
        
        idx = list(range(train_inputs.shape[0]))
        val_len = int(len(idx) * val_split_rate)
        val_idx = np.random.choice(idx,size=val_len,replace=False)
        train_idx = np.array([i for i in idx if i not in val_idx])
        
        # train_ds = pairDataset(train_inputs, obs=train_obs)
        # train_dl = DataLoader(train_ds, batch_size, shuffle=True,drop_last=False) 
        train_ds = pairDataset(train_inputs[train_idx],  obs=train_obs[train_idx])
        val_ds   = pairDataset(train_inputs[val_idx], obs=train_obs[val_idx])
        train_dl = DataLoader(train_ds, batch_size, shuffle=True,drop_last=False)
        val_dl = DataLoader(val_ds, batch_size, shuffle=False,drop_last=False)
        mod1ae = Mod1AutoEncoderFinetune(ishape,batch_dim,latent_space_dim) # mod1 autoencoder  for mod1 2 mod1 
        # load parameters 
        ae_static_dict = torch.load(model_ae_path)
        parameter_modify(ae_static_dict)
        # set parameters not grad 
        Mod1AutoEncoderFinetuneParameterSetting(mod1ae,ae_static_dict)
        # update parameters which need grad  
        mod1ae_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, mod1ae.parameters()),lr=mod1ae_learning_rate)
        mod1ae_lossfn = rec_loss

        def total_test_train(epoch):
            # train mod1 2 mod1 to get batch effect result 
            mod1ae.train()
            step = 0
            for p,y in train_dl:
                p = p[0]
                y = F.one_hot(y,batch_dim)
                x = [p,y] # construct mod1 result 
                pred = mod1ae(x)
                loss = mod1ae_lossfn(pred,p)
                if step % 1 == 0:
                    logging.info("epoch {}; step: {}; loss {}: ".format(epoch,step,loss.item()))
                step += 1
                mod1ae_opt.zero_grad()
                loss.backward()
                mod1ae_opt.step()
        
        def total_test_validation():
            logging.info("test validation phrase")
            total_loss = []
            for p,y in val_dl:
                p = p[0]
                y = F.one_hot(y,batch_dim)
                x = [p,y] # construct mod1 result 
                pred = mod1ae(x)
                loss = mod1ae_lossfn(pred,p)
                total_loss.append(loss.item())
            mean_loss = sum(total_loss) / len(total_loss)
            logging.info("test validation mean loss: {}".format(mean_loss))
            return mean_loss
    
        def total_test_fit(num_mod1_epochs):
            mean_loss_pre = 999999999999999999999999999
            for epoch in range(num_mod1_epochs):
                total_test_train(epoch)
                score = total_test_validation()
                if score < mean_loss_pre:
                    mean_loss_pre = score
                    torch.save(mod1ae.state_dict(),test_ae_path)
        
        total_test_fit(num_mod1_epochs)
        
        mod1ae.load_state_dict(torch.load(test_ae_path))

        # get test mod2 obs batch effect encoder from mod1trainer

        # mlp model
        mlp = LatentMLP(latent_space_dim,mlp_hidden_dim)
        mlp.load_state_dict(torch.load(model_mlp_path))

        # predict model for mod2
        predict_model = Mod2Predict(oshape,batch_dim,latent_space_dim)
        # load parameter from ae and mod1 model
        Mod2PredictParameterSetting(predict_model,mod1ae,ae_static_dict)


        # set the training state
        mod1ae.eval()
        predict_model.eval()

        # using batch to get result 
        res= []
        for i in range(test_len // batch_size+1):
            test_inputs_ = train_inputs[i*batch:(i+1)*batch]
            if len(test_inputs_)<=0:
                break
            this_input_bs= torch.from_numpy(train_obs[i*batch:(i+1)*batch])
            x = [test_inputs_,F.one_hot(this_input_bs,batch_dim)]
            test_input_encoder = mod1ae.get_encoder(x)
            test_pred = predict_model([test_input_encoder,F.one_hot(this_input_bs,batch_dim)])
            res.append(test_pred.detach().cpu())
        total_predict = torch.cat(res,dim=0)
        score = get_score(total_predict,train_targets)
        return score 

    total_train()
    res = total_test()
    logging.info("res : {}".format(res))
    return res

def normal_run():
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(search_space),study_name=path.split("/")[-2])
    study.optimize(objective)

normal_run()



