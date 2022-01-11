# Dependencies:
# pip: scikit-learn, anndata, scanpy
#
# Python starter kit for the NeurIPS 2021 Single-Cell Competition.
# Parts with `TODO` are supposed to be changed by you.
#
# More documentation:
#
# https://viash.io/docs/creating_components/python/

# ./scripts/1_unit_test.sh
# ./scripts/2_generate_submission.sh
# ./scripts/3_evaluate_submission.sh

import logging
import anndata as ad
import gc
import sys
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pickle
logging.basicConfig(level=logging.INFO)

start = time.perf_counter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
# par = {
#     'input_train_mod1': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad',
#     'input_train_mod2': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad',
#     'input_test_mod1': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod1.h5ad',
#     'input_test_mod2': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.test_mod2.h5ad',
#     'distance_method': 'minkowski',
#     'output': 'output.h5ad',
#     'n_pcs': 50,
# }

par_RNA_DNA = {
    # RNA, 16394 * 13431
    'input_train_mod1': 'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad',
    # DNA, 16394 * 10000
    'input_train_mod2': 'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod2.h5ad',
    # RNA, 1000 * 13431
    'input_test_mod1': 'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad',
    # DNA, 1000 * 10000
    'input_test_mod2': 'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad',
    'distance_method': 'minkowski',
    'output': 'output.h5ad',
    'n_pcs': 50,
}

par_DNA_RNA = {
    # DNA, 16394 * 116490
    'input_train_mod1': 'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod1.h5ad',
    # RNA, 16394 * 13431
    'input_train_mod2': 'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_train_mod2.h5ad',
    # DNA, 1000 * 116490
    'input_test_mod1': 'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod1.h5ad',
    # RNA, 1000 * 13431
    'input_test_mod2': 'output/datasets/predict_modality/openproblems_bmmc_multiome_phase1_mod2/openproblems_bmmc_multiome_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
    'distance_method': 'minkowski',
    'output': 'output.h5ad',
    'n_pcs': 50,
}

par_RNA_Pro = {
    # RNA, 29077 * 13953
    'input_train_mod1': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod1.h5ad',
    # Protein, 29077 * 134
    'input_train_mod2': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_train_mod2.h5ad',
    # RNA, 1000 * 13953
    'input_test_mod1': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad',
    # Protein, 1000 * 134
    'input_test_mod2': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad',
    'distance_method': 'minkowski',
    'output': 'output.h5ad',
    'n_pcs': 50,
}

par_Pro_RNA = {
    # Protein, 29077 * 134
    'input_train_mod1': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod1.h5ad',
    # RNA, 29077 * 13953
    'input_train_mod2': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_train_mod2.h5ad',
    # Protein. 1000 * 134
    'input_test_mod1': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod1.h5ad',
    # RNA, 1000 * 13953
    'input_test_mod2': 'output/datasets/predict_modality/openproblems_bmmc_cite_phase1_mod2/openproblems_bmmc_cite_phase1_mod2.censor_dataset.output_test_mod2.h5ad',
    'distance_method': 'minkowski',
    'output': 'output.h5ad',
    'n_pcs': 50,

}

par_list = [par_RNA_DNA, par_DNA_RNA, par_RNA_Pro, par_Pro_RNA]
index = int(sys.argv[1])
# print('index:', type(int(sys.argv[1])))
par = par_list[index]

## VIASH END

# TODO: change this to the name of your method
# par = par_list[int(sys.argv[1])]
method_id = "WRENCH;)"

# logging.info('Reading `h5ad` files...')

# logging.info(par['input_train_mod1'])
# logging.info(par['input_train_mod2'])
# logging.info(par['input_test_mod1'])

input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

s2d4 = input_train_mod1[input_train_mod1.obs["batch"] == "s2d4", :]
input_train_mod1 = s2d4

s2d1 = input_train_mod2[input_train_mod2.obs["batch"] == "s2d4", :]
input_train_mod2 = s2d1

# s3d6 = RNA_data[RNA_data.obs["batch"] == "s3d6", :]
# s2d1 = RNA_data[RNA_data.obs["batch"] == "s2d1", :]
# s1d1 = RNA_data[RNA_data.obs["batch"] == "s1d1", :]

mod1 = input_train_mod1.var['feature_types'][0]
mod2 = input_train_mod2.var['feature_types'][0]

model_path = mod1 +'_to_' + mod2 + '.pth'


# print('model path:', model_path)

# input_train = ad.concat(
#     {"train": input_train_mod1, "test": input_test_mod1},
#     axis=0,
#     join="outer",
#     label="group",
#     fill_value=0,
#     index_unique="-"
# )
#######################################################################


X_train, X_val, Y_train, Y_val = train_test_split(input_train_mod1, input_train_mod2, test_size=0.2, random_state=42)

train_inputs = torch.from_numpy(np.array(X_train.X.toarray()))
train_targets = torch.from_numpy(np.array(Y_train.X.toarray()))
val_inputs = torch.from_numpy(np.array(X_val.X.toarray()))
val_targets = torch.from_numpy(np.array(Y_val.X.toarray()))
test_inputs = torch.from_numpy(np.array(input_test_mod1.X.toarray()))

train_inputs = train_inputs.float()
train_targets = train_targets.float()
val_inputs = val_inputs.float()
val_targets = val_targets.float()
test_inputs = test_inputs.float()

# print(train_inputs.shape, train_targets.shape)

if train_inputs.shape[1] == 600:
    model_path = 'sample_' + model_path

print('model path:', model_path)

test_targets = None
#########
input_test_mod2 = ad.read_h5ad(par['input_test_mod2'])
test_targets = torch.from_numpy(np.array(input_test_mod2.X.toarray()))
test_targets = test_targets.float()
#########
train_inputs = train_inputs.to(device)
train_targets = train_targets.to(device)
val_inputs = val_inputs.to(device)
val_targets = val_targets.to(device)
test_inputs = test_inputs.to(device)
test_targets = test_targets.to(device)
##############################
# num_epochs = int(sys.argv[1])
# learning_rate = float(sys.argv[2])

num_epochs = 500
learning_rate = 0.01
latent_dim = 50
loss_fn = F.mse_loss
batch_size = 1024*8
# model_path = 'auto_encoder_model.pth'

print('Epochs:', num_epochs, 'Learning rate:', learning_rate)
print('latent dim:', latent_dim)
print('batch size:', batch_size)

# Define data loader
train_ds = TensorDataset(train_inputs, train_targets)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

input_feature = train_inputs.shape[1]
output_feature = train_targets.shape[1]

class Autoencoder_model(nn.Module):
    def __init__(self):
        super(Autoencoder_model, self).__init__()
        # encoding
        self.encoder = nn.Sequential(
            nn.Linear(input_feature, input_feature//16),
            nn.Dropout(0.25),
            nn.ReLU(),

            nn.Linear(input_feature//16, input_feature//16),
            nn.ReLU(),

            nn.Linear(input_feature//16, latent_dim),
            nn.ReLU(),
        )
        # decoding
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_feature//16),
            nn.Dropout(0.25),
            nn.ReLU(),

            nn.Linear(input_feature//16, input_feature//16),
            nn.ReLU(),

            nn.Linear(input_feature//16, output_feature),
            # nn.ReLU()
        )

    def forward(self, x):
        encoding = self.encoder(x)
        output = self.decoder(encoding)
        # print('output type:', output.dtype)
        return output.float()



def fit(num_epochs, model, loss_fn):
    val_min_loss = float('inf')
    counter = 0
    # print('Model training...')
    for epoch in range(num_epochs):
        for x,y in train_dl:
            # Generate predictions
            model = model.train()
            pred = model(x)

            loss = loss_fn(pred, y)
            loss = loss.float()
            loss.backward()

            opt.step()
            opt.zero_grad()

        if epoch % 100 == 0:
            loss = loss.cpu().detach().numpy()
            model = model.eval()
            
            val_pred = model(val_inputs)
            val_loss = loss_fn(val_pred, val_targets)
            val_loss = torch.sqrt(val_loss)
            val_loss = val_loss.cpu().detach().numpy()
            
            train_pred = model(train_inputs)
            train_loss = loss_fn(train_pred, train_targets)
            train_loss = torch.sqrt(train_loss)
            train_loss = train_loss.cpu().detach().numpy()
            
            test_pred = model(test_inputs)
            test_loss = loss_fn(test_pred, test_targets)
            test_loss = torch.sqrt(test_loss)
            test_loss = test_loss.cpu().detach().numpy()

            print('Epoch ', epoch, 'Train_loss: ', train_loss, ' Validation_loss: ', val_loss, ' Test_loss: ', test_loss)
    # train_pred = train_pred.cpu().detach().numpy()
    # val_pred = val_pred.cpu().detach().numpy()
    # test_pred = test_pred.cpu().detach().numpy()
    
    return train_pred, val_pred, test_pred

model = Autoencoder_model()
model = model.to(device)
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

print('Training the model')
(train_pred, val_pred, test_pred) = fit(num_epochs, model, loss_fn)
print('train pred:')
# print(train_pred)
plt.matshow((train_pred-train_targets).cpu().detach().numpy())
train_error = (train_pred-train_targets).cpu().detach().numpy()
    
print('val pred:')
# print(val_pred)
plt.matshow((val_pred-val_targets).cpu().detach().numpy())
val_error = (val_pred-val_targets).cpu().detach().numpy()

print('test pred:')
# print(test_pred)
plt.matshow((test_pred-test_targets).cpu().detach().numpy())
test_error = (test_pred-test_targets).cpu().detach().numpy()

plt.show()

res = [train_error, val_error, test_error]

with open('error.pickle', 'wb') as handle:
    pickle.dump(res, handle)

train_error_square = np.square(train_error)
val_error_square = np.square(val_error)
test_error_square = np.square(test_error)

error_square = [train_error_square, val_error_square, test_error_square]
with open('error_square.pickle', 'wb') as handle:
    pickle.dump(error_square, handle)

train_error_square_gene = train_error_square.sum(axis = 0)
val_error_square_gene = val_error_square.sum(axis = 0)
test_error_square_gene = test_error_square.sum(axis = 0)

error_square_gene = [train_error_square_gene, val_error_square_gene, test_error_square_gene]

with open('error_square_gene.pickle', 'wb') as handle:
    pickle.dump(error_square_gene, handle)
    
# with open('error.pickle', 'rb') as handle:
#     error = pickle.load(handle)
    

# torch.save(model.state_dict(), model_path)

# # model = Autoencoder_model()
# model.load_state_dict(torch.load(model_path))
# model.eval()
# Y_pred = model(test_inputs)

# ###############
# # ## this is for self testing
# # input_test_mod2 = ad.read_h5ad(par['input_test_mod2'])
# # test_targets = torch.from_numpy(np.array(input_test_mod2.X.toarray()))
# # test_targets = test_targets.float()
# # test_loss = loss_fn(Y_pred, test_targets)
# # print('Testing loss:', test_loss.cpu().detach().numpy())
# ###############

# Y_pred = Y_pred.cpu().detach().numpy()

# Y_pred = csc_matrix(Y_pred)
# adata = ad.AnnData(
#     X=Y_pred,
#     obs=input_test_mod1.obs,
#     var=input_train_mod2.var,
#     uns={
#         'dataset_id': input_train_mod1.uns['dataset_id'],
#         'method_id': method_id,
#     },
# )

# logging.info('Storing annotated data...')
# adata.write_h5ad(par['output'], compression = "gzip")
# # adata.write_h5ad(par[model_path], compression = "gzip")
# end = time.perf_counter()
# print(f'Training finished in {end-start} seconds')
#######################################################################
# # TODO: implement own method

# # Do PCA on the input data
# print('Build the model')
# logging.info('Performing dimensionality reduction on modality 1 values...')
# embedder_mod1 = TruncatedSVD(n_components=50)
# mod1_pca = embedder_mod1.fit_transform(input_train.X)

# logging.info('Performing dimensionality reduction on modality 2 values...')
# embedder_mod2 = TruncatedSVD(n_components=50)
# mod2_pca = embedder_mod2.fit_transform(input_train_mod2.X)

# # split dimred back up
# X_train = mod1_pca[input_train.obs['group'] == 'train']
# X_test = mod1_pca[input_train.obs['group'] == 'test']
# y_train = mod2_pca

# # assert len(X_train) + len(X_test) == len(mod1_pca)

# # Get all responses of the training data set to fit the
# # KNN regressor later on.
# #
# # Make sure to use `toarray()` because the output might
# # be sparse and `KNeighborsRegressor` cannot handle it.

# logging.info('Running Linear regression...')

# reg = LinearRegression()

# # Train the model on the PCA reduced modality 1 and 2 data
# reg.fit(X_train, y_train)
# print('Prediction...')
# y_pred = reg.predict(X_test)

# # Project the predictions back to the modality 2 feature space
# y_pred = y_pred @ embedder_mod2.components_

# # Store as sparse matrix to be efficient. Note that this might require
# # different classifiers/embedders before-hand. Not every class is able
# # to support such data structures.
# y_pred = csc_matrix(y_pred)

# print(len(input_train_mod1))
# print(input_train_mod1.uns['dataset_id'])

# adata = ad.AnnData(
#     X=y_pred,
#     obs=input_test_mod1.obs,
#     var=input_train_mod2.var,
#     uns={
#         'dataset_id': input_train_mod1.uns['dataset_id'],
#         'method_id': method_id,
#     },
# )

# logging.info('Storing annotated data...')
# adata.write_h5ad(par['output'], compression = "gzip")
