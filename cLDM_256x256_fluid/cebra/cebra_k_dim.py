import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
Z_data = np.load('../data/ddim_latents_Re120_time_3000.npy')
labels = np.load('../data/norm_labels_Re120_time_3000.npy')
labels_raw = np.load('../data/raw_labels_Re120_time_3000.npy')
X_data = np.load('../data/X_data_Re120_time_3000.npy')


import sklearn.linear_model
from cebra import CEBRA


embeddings = dict()

save_dir = "saved_cebra_models"
os.makedirs(save_dir, exist_ok=True)

embeddings = {}

for k in [2, 3, 4, 8, 10, 14, 16, 32, 64, 128, 256, 512]:
    print(f"Fitting CEBRA for dim = {k}")

    cebra_model = CEBRA(
        model_architecture='offset10-model-mse',
        batch_size=512,
        learning_rate=5e-6,
        output_dimension=k,
        delta=20,
        temperature=1.12,
        max_iterations=5000,
        distance='euclidean',
        conditional='time',
        device='cuda_if_available',
        verbose=True,
        time_offsets=8
    )

    cebra_model.fit(Z_data, labels)

    # Save model separately
    model_path = os.path.join(save_dir, f"cebra_dim{k}.pt")
    cebra_model.save(model_path)

    # Save embedding in dictionary
    embeddings[f"dim{k}"] = cebra_model.transform(Z_data)

    print(f"Saved model: {model_path}")
    
print(embeddings.keys())    


os.makedirs("k-dim-cebra-data", exist_ok=True)
for key, value in embeddings.items():
   np.save(f"k-dim-cebra-data/{key}.npy", value)

dict_keys = ['dim2', 'dim3', 'dim4', 'dim8', 'dim10', 'dim14', 'dim16', 'dim32', 'dim64', 'dim128', 'dim256', 'dim512']
# Load back
cebra_k_dim = {key: np.load(f"k-dim-cebra-data/{key}.npy") for key in dict_keys}

import cebra
import sklearn.metrics

from sklearn.model_selection import train_test_split

number_list = list(range(3000))
train_block_size = 48
test_block_size = 12
num_blocks = 50
train_blocks = [number_list[i * (train_block_size+test_block_size)  : (i + 1) * train_block_size + test_block_size*i] 
                for i in range(num_blocks)]
test_blocks = [number_list[(i + 1) * train_block_size + test_block_size*i : (i+1)*(train_block_size + test_block_size)]
               for i in range(num_blocks)]  
train_idx = []
test_idx = []
for j in range(num_blocks):
    train_idx = train_idx + train_blocks[j]
    test_idx = test_idx + test_blocks[j]

print("Train:", len(train_idx), "Test:", len(test_idx))

# Hyperparameter selection for k
neighbors = [1, 3, 5, 8, 10, 20, 30, 50]

def rmse_pointwise(X, Y):  # X, Y: (T, dim)
    return np.sqrt(np.mean(np.sum((X - Y)**2, axis=1)))   

best_rmse, best_n = np.inf, None
rmse_values = []
for k in dict_keys:
    print(k)
    for n in neighbors:
        model = cebra.KNNDecoder(n_neighbors=n, metric="euclidean")
        model.fit(cebra_k_dim[k][train_idx], Z_data[train_idx])
        Y_pred = model.predict(cebra_k_dim[k][test_idx])
        rmse = rmse_pointwise(Y_pred, Z_data[test_idx])
        if rmse < best_rmse:
            best_rmse, best_n = rmse, n
    print("Best n_neighbors:", best_n, "Lowest RMSE:", best_rmse)
    rmse_values.append(best_rmse)