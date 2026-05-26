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
#from Train_cLDM import *
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
clean_images_folder = wd + 'output/clean_images/CcLDM_clean'
os.makedirs(clean_images_folder, exist_ok=True)
loss_folder = wd + 'output/loss'
os.makedirs(loss_folder, exist_ok=True)

#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
# Load fluid data of size 256x256
data_filename = args.data_path + '/fluid_256x256.npy'
v_images_all = np.load(data_filename)
images_all = v_images_all.transpose((0,3,1,2))
print("\n flow dataset shape: {}x{}x{}x{}".format(images_all.shape[0], images_all.shape[1], images_all.shape[2], images_all.shape[3]))

#### Split data into train and test 80:20 ratio
number_list = list(range(3000))
train_block_size = 80
test_block_size = 20
num_blocks = 30
train_blocks = [number_list[i * (train_block_size+test_block_size)  : (i + 1) * train_block_size + test_block_size*i] 
                for i in range(num_blocks)]
test_blocks = [number_list[(i + 1) * train_block_size + test_block_size*i : (i+1)*(train_block_size + test_block_size)]
               for i in range(num_blocks)]  
indx_train = []
indx_val = []
for j in range(num_blocks):
    indx_train = indx_train + train_blocks[j]
    indx_val = indx_val + test_blocks[j]

labels_list = np.linspace(0.0005,1.5,3000)
labels_all = np.tile(labels_list, 82)

# data split
if args.data_split == "train":
    images_train = images_all.reshape(82,3000,3,256,256)[:,indx_train,:,:,:]
    images_train = images_train.reshape(82*2400,3,256,256)
    labels_train_raw = np.tile(labels_list[indx_train], 82)
    
    images_val = images_all.reshape(82,3000,3,256,256)[:,indx_val,:,:,:]
    images_val = images_val.reshape(82*600,3,256,256)
    labels_val_raw = np.tile(labels_list[indx_val], 82)
    
else:
    images_train = copy.deepcopy(images_all)
    labels_train_raw = copy.deepcopy(labels_all)
    images_val = copy.deepcopy(images_all)
    labels_val_raw = copy.deepcopy(labels_all)

labels_train = (labels_train_raw - np.min(labels_train_raw)) / (np.max(labels_train_raw)-np.min(labels_train_raw))
labels_val = (labels_val_raw - np.min(labels_val_raw)) / (np.max(labels_val_raw)-np.min(labels_val_raw))
      
#######################################################################################
'''                                    LDM training                                 '''
#######################################################################################
save_LDMimages_InTrain_folder = save_images_folder + '/CcLDM_InTrain'
os.makedirs(save_LDMimages_InTrain_folder, exist_ok=True)

start = timeit.default_timer()
print("\n Begin Training:" )

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

