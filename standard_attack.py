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

from advertorch.attacks.base import Attack, LabelMixin
from advertorch.utils import clamp
import matplotlib.pyplot as plt
from advertorch.attacks import LinfPGDAttack, GradientSignAttack, CarliniWagnerL2Attack, LinfBasicIterativeAttack



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

class Model_for_Adversary_Caps(nn.Module):
    def __init__(self, net):
        super(Model_for_Adversary_Caps, self).__init__()
        self.net = net
        
    def forward(self, x):
        output, recons, masked = self.net(x)
        classes = torch.sqrt((output ** 2).sum(2)).squeeze()
        return classes

class Model_for_Adversary_CNN(nn.Module):
    def __init__(self, net):
        super(Model_for_Adversary_CNN, self).__init__()
        self.net = net
        
    def forward(self, x):
        output, recons, masked = self.net(x)
        classes = output.sum(2)
        return classes
    
def WhiteBox_Attacks_Targeted(net, dataloader, adversary_dict, args):
    net.eval()
    n_batch = len(dataloader)
    Success_Rate = {key:0.0 for key in adversary_dict.keys()}
    Undetected_Rate = {key:0.0 for key in adversary_dict.keys()}
    Und_l2 = {key:torch.tensor([],dtype=torch.int16).cuda() for key in adversary_dict.keys()}
    for adversary in adversary_dict.keys():
        for batch_id, (data, labels) in enumerate(tqdm(dataloader)):
            if(args['USE_CUDA']):
                data, labels = data.cuda(), labels.cuda()
            target = torch.randint(0,10,size=(labels.size(0),), dtype=labels.dtype).cuda()
            while(torch.sum(target==labels)/target.size(0)>0.0001):
                target[target==labels] = torch.randint(0,10, size=(torch.sum(target==labels),), dtype=labels.dtype).cuda()
            adv_data = adversary_dict[adversary].perturb(data, target)
            with torch.no_grad():
                output, reconstructions, max_length_indices = net(adv_data)
                unnormalized_data = net.decoder.unnormalize(adv_data)
                l2_distances = ((reconstructions.view(adv_data.size(0),-1)-unnormalized_data.view(adv_data.size(0), -1))**2).sum(1).squeeze().detach()
                theta = np.percentile(l2_distances.cpu().numpy(), 95)
                if(adversary=='Clean'):
                    Und_l2[adversary] = torch.cat((Und_l2[adversary],l2_distances))
                else:
                    Und_l2[adversary] = torch.cat((Und_l2[adversary],l2_distances[max_length_indices == target]))
                Success_Rate[adversary]+=torch.sum(max_length_indices == target).item()
                Undetected_Rate[adversary]+=torch.sum(l2_distances[max_length_indices == target]<=theta).item()
        
        Und_l2[adversary] = Und_l2[adversary].cpu().numpy()
        Success_Rate[adversary]/=100
        Undetected_Rate[adversary]/=100
    return Success_Rate, Undetected_Rate, Und_l2

def WhiteBox_Attacks_Untargeted(net, dataloader, adversary_dict, args):
    net.eval()
    n_batch = len(dataloader)
    Success_Rate = {key:0.0 for key in adversary_dict.keys()}
    Undetected_Rate = {key:0.0 for key in adversary_dict.keys()}
    Und_l2 = {key:torch.tensor([],dtype=torch.int16).cuda() for key in adversary_dict.keys()}
    for adversary in adversary_dict.keys():
        for batch_id, (data, labels) in enumerate(tqdm(dataloader)):
            if(args['USE_CUDA']):
                data, labels = data.cuda(), labels.cuda()

            adv_data = adversary_dict[adversary].perturb(data)
            with torch.no_grad():
                output, reconstructions, max_length_indices = net(adv_data)
                unnormalized_data = net.decoder.unnormalize(adv_data)
                l2_distances = ((reconstructions.view(adv_data.size(0),-1)-unnormalized_data.view(adv_data.size(0), -1))**2).sum(1).squeeze().detach()
                theta = np.percentile(l2_distances.cpu().numpy(), 95)
                if(adversary=='Clean'):
                    Und_l2[adversary] = torch.cat((Und_l2[adversary],l2_distances))
                else:
                    Und_l2[adversary] = torch.cat((Und_l2[adversary],l2_distances[max_length_indices != labels]))
                Success_Rate[adversary]+=torch.sum(max_length_indices != labels).item()
                Undetected_Rate[adversary]+=torch.sum(l2_distances[max_length_indices!=labels]<=theta).item()
#                 print(Success_Rate[adversary])
#                 print(Undetected_Rate[adversary])
#                 print(theta)
        Und_l2[adversary] = Und_l2[adversary].cpu().numpy() 
        Success_Rate[adversary]/=100
        Undetected_Rate[adversary]/=100
    return Success_Rate, Undetected_Rate, Und_l2

class CleanAttack(Attack, LabelMixin):
    def __init__(self, clip_min=0., clip_max=1.):
        super(CleanAttack, self).__init__(None,None,clip_min, clip_max)

    def perturb(self, x, y=None):
        return x

def make_adversary_dict(model, model_name, targetted = False):
    if(model_name=="capsnet"):
        model_for_adversary = Model_for_Adversary_Caps(model)
    else:
        model_for_adversary = Model_for_Adversary_CNN(model)

    linf_eps = 0.3
    fgsm_step = 0.05
    bim_pgd_step = 0.01

    adversary_dict = {}
    adversary_dict['Clean'] = CleanAttack(clip_min=-0.4242, clip_max=2.8215)
    adversary_dict['PGD'] = LinfPGDAttack(
        model_for_adversary, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=(linf_eps/0.3081),
        nb_iter=100, eps_iter=(bim_pgd_step/0.3081), rand_init=True, clip_min=-0.4242, clip_max=2.8215,
        targeted=targetted)

    adversary_dict['FGSM'] = GradientSignAttack(model_for_adversary, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=(fgsm_step/0.3081), clip_min=-0.4242, clip_max=2.8215, targeted=targetted)
    adversary_dict['BIM'] = LinfBasicIterativeAttack(model_for_adversary, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=(linf_eps/0.3081), nb_iter=100, eps_iter=(bim_pgd_step/0.3081), clip_min=-0.4242, clip_max=2.8215, targeted=targetted)

    return adversary_dict

def capsnet():
    config = Caps_Config()
    net = CapsNet(Caps_args, config)
    # capsule_net = torch.nn.DataParallel(capsule_net)
    if Caps_args['USE_CUDA']:
        net = net.cuda()
    net.load_state_dict(torch.load(os.path.join(model_path, 'CapsNet_mnist.pth'), map_location='cpu'))
    return net, Caps_args

def CNN(model_type):
    CNN_args['type'] = model_type
    config = CNN_Config()
    net = CNNnet(CNN_args, config)
    net.load_state_dict(torch.load(os.path.join(model_path, 'CNN'+model_type+'_mnist.pth'), map_location='cpu'))
    if CNN_args['USE_CUDA']:
        net = net.cuda()
    return net, CNN_args

def load_model(model_name):
    if(model_name=="capsnet"):
        net, args = capsnet()
        return net, args
    else:
        net, args = CNN(model_name)
        return net, args


_, testloader = dataset(Caps_args)

model_name_list = ["plusCR", "plusR", "capsnet"]
types = [True, False]
funcs = [ WhiteBox_Attacks_Targeted, WhiteBox_Attacks_Untargeted]
for model_name in model_name_list:
    net, args = load_model(model_name)
    for i,func  in enumerate(funcs):
        adversary_dict = make_adversary_dict(net, model_name, targetted=types[i])
        _, _, Und_l2 = func(net, testloader, adversary_dict, args)
        for attack in adversary_dict.keys():
            print(Und_l2[attack].shape[0],": Success Rate for",model_name,attack)
            print(np.sum(Und_l2[attack]<45),": Undetected Rate for",model_name,attack)
        if (types[i] == True):
            targetted = "Targeted" 
        else:
            targetted = "Untargeted"
        if(model_name =="capsnet"):
            mod_name = "CapsNet"
        elif(model_name =="plusCR"):
            mod_name = "plusCR"
        else:
            mod_name = "plusR"
        np.save(os.path.join(base_path, "results",str("Und_L2_"+targetted+"_"+mod_name+".npy")),Und_l2)