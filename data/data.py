import torch
import numpy
import torch
import torchvision
import torchvision.transforms as transforms
import os

data_path = '/raid/sdas_ma/Adversarial_CapsNet_Pytorch/data/'

def dataset(args): 
    if(args['DATASET_NAME']=='mnist'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        trainset = torchvision.datasets.MNIST(root=os.path.join(data_path,'mnist'), train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['BATCH_SIZE'],
                                                  shuffle=True, num_workers=1)

        testset = torchvision.datasets.MNIST(root=os.path.join(data_path,'mnist'), train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args['BATCH_SIZE'],
                                                 shuffle=False, num_workers=1)
        return trainloader, testloader