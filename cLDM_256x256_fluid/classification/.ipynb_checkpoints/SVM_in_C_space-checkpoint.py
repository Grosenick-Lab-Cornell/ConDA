import numpy as np
import time
start_time = time.time()

ddim_latents_ = np.load('Z_class.npy').reshape(88, 100, 4, 32, 32)

idx_lam = list(range(0,21)) + list(range(82,86))
idx_per = list(range(86,88)) + list(range(21,42))

# For classification: consider laminar and periodic flow with fixed r = 0.05, 
ddim_latents_lam  =  ddim_latents_[idx_lam]
ddim_latents_per  =  ddim_latents_[idx_per]

latent_data = np.concatenate((ddim_latents_lam, ddim_latents_per), axis=0)
labels = np.concatenate((np.ones(25*100, dtype ='int'), np.zeros(23*100, dtype = 'int'))) 
labels_reshaped = labels.reshape(48, 100)

# Select unseen Re for evaluation (4 laminar and 4 periodic)
import random
np.random.seed(42)
i_lam = random.sample(range(17,25), 4)
i_per = random.sample(range(25,33), 4)

Z_latents = latent_data
Z_unseen = np.concatenate((Z_latents[i_lam], Z_latents[i_per]), axis=0)

# Rest data 48-N_unseen Reynolds number
Z_latent_data = np.delete(Z_latents, i_lam + i_per, axis=0)
print(Z_latent_data.shape)

labels_unseen = np.concatenate((labels_reshaped[i_lam], labels_reshaped[i_per]), axis=0)
print(labels_unseen.shape)

labels_data = np.delete(labels_reshaped, i_lam + i_per, axis=0)
print(labels_data.shape)

# Shuffle the Reynolds number and set the seed for reproducibility
np.random.seed(42)

n_Re = len(labels_data) # no. of Re to train
idx = np.random.permutation(n_Re)
shuffled_latent_data = Z_latent_data[idx]
n_t = 100 # number of time points

n = n_Re*n_t

sel_latent = shuffled_latent_data.reshape(n, 4*32*32)
shuffled_labels = labels_data[idx].reshape(n)

# Shuffle the unseen subjects
n_Re_unseen = len(labels_unseen)
idx_unseen = np.random.permutation(n_Re_unseen)

n_ = n_Re_unseen*n_t

shuffled_Z_unseen = Z_unseen[idx_unseen].reshape(n_, 4*32*32)
shuffled_labels_unseen = labels_unseen[idx_unseen].reshape(n_)

Z_data = sel_latent
print(Z_data.shape)


### For SVM Classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

# Scale the data
scaler = StandardScaler()

X = scaler.fit_transform(Z_data)
X_unseen = scaler.transform(shuffled_Z_unseen)

y = shuffled_labels
y_unseen = shuffled_labels_unseen

# Define Re indices for cross-validation
Re_ids = np.repeat(np.arange(n_Re), n_t)  # Assign same ID to all time points of an Re

# Define SVM model with class_weight="balanced"
clf = SVC(kernel='rbf', probability=True)
#clf = SVC(kernel='linear', probability=True)

# Parameter grid for hyperparameter tuning
param_grid = {
   'C': [1e-2, 0.1, 1, 10, 1e2, 1e3],
   'gamma': [1e-3, 1e-2, 0.1, 1, 10]
}

# param_grid = {
#     'C': [1e-2,1e-1,1,10,1e2,1e3],
# }

# Use GroupKFold to ensure different Re are in separate folds
group_kfold = GroupKFold(n_splits=5)

# Grid Search with F1 scoring
grid_search = GridSearchCV(clf, param_grid, cv=group_kfold, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X, y, groups = Re_ids)

# Best parameters based on F1-score
print("Best parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

import pandas as pd
# Extract results
results = pd.DataFrame(grid_search.cv_results_)

## Predictions on unseen Re
print('Predictions for unseen Reynolds Number')
print("SVM best parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
best_model.fit(X, y)

y_pred = best_model.predict(X_unseen)

# Define a custom scorer for specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)  # True Negative Rate
    
# Define a custom scorer for specificity
def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)  # True Positive Rate

# Compute accuracy
accuracy = accuracy_score(y_unseen, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Compute the F1 score
f1 = f1_score(y_unseen, y_pred) 
print(f"F1 Score: {f1:.2f}")

spec = specificity_score(y_unseen, y_pred)
print(f"Specificity: {spec:.2f}")

sens = sensitivity_score(y_unseen, y_pred)
print(f"Sensitivity: {sens:.2f}")
    
y_pred_probs_u = best_model.predict_proba(X_unseen)[:, 1]
auc = roc_auc_score(y_unseen, y_pred_probs_u)
print(f"AUC: {auc:.2f}")

print('Unseen Re true labels:', y_unseen)
print('Unseen Re predicted labels:', y_pred)

end_time = time.time()

print("Start time:", start_time)
print("End time:", end_time)
print("Elapsed time:", end_time - start_time, "seconds")
