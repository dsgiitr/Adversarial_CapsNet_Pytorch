#Experiment Objectives:
#1. Generate Attack from model j and test accuracies, L2 Distances and Reconstructions on model i (!=j)

import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from advertorch.attacks import LinfPGDAttack, GradientSignAttack, CarliniWagnerL2Attack, LinfBasicIterativeAttack
# import seaborn as sns

import sys
sys.path.insert(0,'/raid/sdas_ma/Adversarial_CapsNet_Pytorch/')
from model.net import *
from model.cnn_net import *
from utils.training import *
from data.data import *


base_path = '/raid/sdas_ma/Adversarial_CapsNet_Pytorch/'
model_path = "/raid/sdas_ma/Adversarial_CapsNet_Pytorch/weights/"#os.path.join(os.getcwd(), "weights")

Caps_args = {
    'DATASET_NAME':'mnist',
    'num_classes':10,
    
    'USE_CUDA': True if torch.cuda.is_available() else False,
    'BATCH_SIZE': 512,
    
    ##For Decoder
    'num_features':160,
    'LReLU_negative_slope':0.1,
    'input_height':28,
    'input_width':28,
    'input_channel':1,    
}

CNN_args = {
    'DATASET_NAME':'mnist',
    'num_classes':10,
    
    'USE_CUDA': True if torch.cuda.is_available() else False,
    'BATCH_SIZE': 256,    
    #For Decoder
    'num_features':160,
    'LReLU_negative_slope':0.1,
    'input_height':28,
    'input_width':28,
    'input_channel':1,
    'type':'plusCR',
}

class Caps_Config:
    def __init__(self, dataset='mnist'):
        # CNN (cnn)
        self.cnn_in_channels = 1
        self.cnn_out_channels = 12
        self.cnn_kernel_size = 15

        # Primary Capsule (pc)
        self.pc_num_capsules = 1
        self.pc_in_channels = 12
        self.pc_out_channels = 16
        self.pc_kernel_size = 8
        self.pc_num_routes = 7 * 7

        # Digit Capsule 1 (dc)
        self.dc_num_capsules = 49
        self.dc_num_routes = 7 * 7
        self.dc_in_channels = 16
        self.dc_out_channels = 16 #1
        
        # Digit Capsule 2 (dc)
        self.dc_2_num_capsules = 10
        self.dc_2_num_routes = 7 * 7
        self.dc_2_in_channels = 16 #1
        self.dc_2_out_channels = 16

        # Decoder
        self.input_width = 28
        self.input_height = 28
        
class CNN_Config:
    def __init__(self, dataset='mnist'):
        # CONV1
        self.conv1_in = 1
        self.conv1_out = 12
        self.conv1_kernel_size = 15

        # CONV2
        self.conv2_in = 12
        self.conv2_out = 16
        self.conv2_kernel_size = 8

        # FC1
        self.fc1_in = 7 * 7 * 16
        self.fc1_out = 784
        
        # FC1
        self.fc2_in = 784
        self.fc2_out = 160
        
torch.manual_seed(1)

################Loading Dataset####################
cmnistlist = ["brightness",  "canny_edges",  "dotted_line",  "fog" , "glass_blur" , "identity",  "impulse_noise",  "motion_blur",  "rotate",  "scale",  "shear" , "shot_noise" , "spatter",  "stripe",  "translate",  "zigzag"]
cmnistpath = os.path.join(base_path,"data/mnist_c/")

def test_n_l2(model, model_name):
    net.eval()
    Und_l2 = {key:torch.tensor([],dtype=torch.bool) for key in cmnistlist}
    l2_distances_all = {key:torch.tensor([],dtype=torch.int16) for key in cmnistlist}
    
    for corruption in cmnistlist:
        datapath = os.path.join(cmnistpath, corruption)
        unnorm_data = torch.tensor(np.load(os.path.join(datapath, "test_images.npy"))/255).permute((0,3,1,2))
        labels = torch.tensor(np.load(os.path.join(datapath, "test_labels.npy")))
        if(labels.min()==1):
            labels -= 1
        unnorm_data, labels = unnorm_data.cuda().type(torch.cuda.FloatTensor), labels.cuda()
        #normalizing
        data = (unnorm_data - 0.1307)/0.3081
        with torch.no_grad():
            output, reconstructions, max_length_indices = net(data)
            l2_distances = ((reconstructions.view(unnorm_data.size(0),-1)-unnorm_data.view(unnorm_data.size(0), -1))**2).sum(1).squeeze().detach()
            Und_l2[corruption] = torch.cat((Und_l2[corruption], (max_length_indices != labels).detach().cpu()))
            l2_distances_all[corruption] = torch.cat((l2_distances_all[corruption], l2_distances.detach().cpu()))
        Und_l2[corruption] = Und_l2[corruption].numpy()
        l2_distances_all[corruption] = l2_distances_all[corruption].numpy()
        print(model_name," : " ,corruption," : ",np.sum(Und_l2[corruption])/100," // ",np.sum(l2_distances_all[corruption][Und_l2[corruption]]<45)/100)
    np.save(os.path.join(base_path , "results", str("cmnist"+model_name+"Undl2.npy")), Und_l2)
    np.save(os.path.join(base_path , "results", str("cmnist"+model_name+"l2_dist.npy")), l2_distances_all)

def capsnet():
    config = Caps_Config()
    net = CapsNet(Caps_args, config)
    # capsule_net = torch.nn.DataParallel(capsule_net)
    if Caps_args['USE_CUDA']:
        net = net.cuda()
    net.load_state_dict(torch.load(os.path.join(model_path, 'CapsNet_mnist.pth'), map_location='cpu'))
    return net

def CNN(model_type):
    CNN_args['type'] = model_type
    config = CNN_Config()
    net = CNNnet(CNN_args, config)
    net.load_state_dict(torch.load(os.path.join(model_path, 'CNN'+model_type+'_mnist.pth'), map_location='cpu'))
    if CNN_args['USE_CUDA']:
        net = net.cuda()
    return net

def load_model(model_name):
    if(model_name=="capsnet"):
        net = capsnet()
        return net
    else:
        net = CNN(model_name)
        return net

model_name_list = ["capsnet", "plusCR", "plusR"]
for model_name in model_name_list:
    net = load_model(model_name)
    test_n_l2(net, model_name)



