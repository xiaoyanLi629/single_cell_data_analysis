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
import matlab.engine
import heapq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

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

par = par_list[3]

input_ = ad.read_h5ad(par['input_train_mod2'])
s1d1 = input_[input_.obs["batch"] == "s1d1", :]
RNA_s1d1 = s1d1
s1d1_array = RNA_s1d1.X.toarray()
s1d1_mean = np.mean(s1d1_array, axis=0)+10
s1d1_std = np.std(s1d1_array, axis=0)

rvals = 2*np.random.rand(s1d1_mean.shape[0],1)-1
elevation = np.arcsin(rvals)

azimuth = 2*np.pi*np.random.rand(s1d1_mean.shape[0],1)
radii = np.ones((s1d1_mean.shape[0], 1))
radii = s1d1_mean
radii = np.reshape(radii, (radii.shape[0], 1))
radii.shape

# [x,y,z] = sph2cart(azimuth,elevation,radii);

z = radii*np.sin(elevation)
x = radii*np.cos(elevation)*np.cos(azimuth)
y = radii*np.cos(elevation)*np.sin(azimuth)

# s = randi([100, 300], dim, 1);
# c = rand(dim, 3);

# %matplotlib inline
plt.rcParams["figure.figsize"] = [7.00, 8.0]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.axis('equal')
# ax.set_aspect('equal')
ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
ax.scatter3D(x, y, z, c=s1d1_std, s = s1d1_std);
x = x.tolist()
y = y.tolist()
z = z.tolist()
x = [ele[0] for ele in x]
y = [ele[0] for ele in y]
z = [ele[0] for ele in z]
for i in range(len(x)):
    distance = [(x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2 for j in range(len(x))]
    top_4 = (heapq.nsmallest(4, distance))
    nearest_index = [distance.index(top_4[1]), distance.index(top_4[2]), distance.index(top_4[3])]
    ax.plot([x[i], x[nearest_index[0]]], [y[i], y[nearest_index[0]]], [z[i], z[nearest_index[0]]], color='blue', linewidth=0.5)
    ax.plot([x[i], x[nearest_index[1]]], [y[i], y[nearest_index[1]]], [z[i], z[nearest_index[1]]], color='blue', linewidth=0.5)
    ax.plot([x[i], x[nearest_index[2]]], [y[i], y[nearest_index[2]]], [z[i], z[nearest_index[2]]], color='blue', linewidth=0.5)
plt.show()