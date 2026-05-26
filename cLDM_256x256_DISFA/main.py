print("\n===================================================================================================")

import argparse
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import random
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
#from torch.nn.parallel import DistributedDataParallel
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import timeit
from PIL import Image

from opts import parse_opts
args = parse_opts()
wd = args.root_path
os.chdir(wd)

from utils import *
from models import *
from Train_CcLDM import *

#######################################################################################
'''                                   Settings                                      '''
#######################################################################################

# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = 8


# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# output folders
save_models_folder = wd + 'output/saved_models'
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = wd + 'output/saved_images'
os.makedirs(save_images_folder, exist_ok=True)
clean_images_folder = wd + 'output/clean_images/{}_clean'.format(args.LDM)
os.makedirs(clean_images_folder, exist_ok=True)
loss_folder = wd + 'output/loss'
os.makedirs(loss_folder, exist_ok=True)

#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# Load DISFA data of size 3X768x1024 
data_file = args.data_path

LeftVideo_array_train = np.load(data_file + 'train_LeftVideo_array.npy')
RightVideo_array_train = np.load(data_file + 'train_RightVideo_array.npy')

LeftVideo_array_test = np.load(data_file + 'test_LeftVideo_array.npy')
RightVideo_array_test = np.load(data_file + 'test_RightVideo_array.npy')

LeftLabels_train = np.load(data_file + 'train_LeftLabels.npy')
RightLabels_train = np.load(data_file + 'train_RightLabels.npy')

LeftLabels_test = np.load(data_file + 'test_LeftLabels.npy')
RightLabels_test = np.load(data_file + 'test_RightLabels.npy')

X_train  = np.concatenate((LeftVideo_array_train,RightVideo_array_train), axis=0)
print('Shape of training data frames', X_train.shape)
Y_train = np.concatenate((LeftLabels_train,RightLabels_train), axis=0)

X_test  = np.concatenate((LeftVideo_array_test,RightVideo_array_test), axis=0)
print('Shape of test data', X_test.shape)
Y_test = np.concatenate((LeftLabels_test,RightLabels_test), axis=0)

# Calcium imaging images
images_train = X_train.transpose(0,3,1,2)
images_val = X_test.transpose(0,3,1,2)

# Time point of data
labels_train_raw = Y_train
labels_val_raw = Y_test

q1 = 5

print("\n Range of unnormalized training labels: ({},{})".format(np.min(labels_train_raw), np.max(labels_train_raw))) 
print("\n Range of unnormalized test labels: ({},{})".format(np.min(labels_val_raw), np.max(labels_val_raw))) 

# Normalize the labels
labels_train = labels_train_raw / q1
print("\n Range of normalized training labels: ({},{})".format(np.min(labels_train), np.max(labels_train)))

labels_val = labels_val_raw / q1
print("\n Range of normalized test labels: ({},{})".format(np.min(labels_val), np.max(labels_val)))

## end if args.LDM

#######################################################################################
'''                                    cLDM training                                 '''
#######################################################################################

save_LDMimages_InTrain_folder = save_images_folder + '/{}_InTrain'.format(args.LDM)
os.makedirs(save_LDMimages_InTrain_folder, exist_ok=True)

start = timeit.default_timer()
print("\n Begin Training %s:" % args.LDM)

#----------------------------------------------
# Continuous cLDM
Filename_LDM = save_models_folder + '/ckpt_CcLDM_niters_{}_seed_{}.pth'.format(args.niters, args.seed)
print(Filename_LDM)

if not os.path.isfile(Filename_LDM):
    netLDM = cont_cond_unet_diffusion_model()
    netLDM = nn.DataParallel(netLDM)    
    # Start training
    netLDM = train_CcLDM(images_train, labels_train, images_val, labels_val, netLDM, save_images_folder=save_LDMimages_InTrain_folder, save_models_folder = save_models_folder)
    # store model
    torch.save({
        'netLDM_state_dict': netLDM.state_dict(),
    }, Filename_LDM)
        
else:
    print("Loading pre-trained continuous conditional latent diffusion model >>>")
    checkpoint = torch.load(Filename_LDM)
    netLDM = cont_cond_unet_diffusion_model().to(device)
    netLDM = nn.DataParallel(netLDM)
    netLDM.load_state_dict(checkpoint['netLDM_state_dict'])
    
stop = timeit.default_timer()
print("Diffusion model training finished; Time elapses: {}s".format(stop - start))

