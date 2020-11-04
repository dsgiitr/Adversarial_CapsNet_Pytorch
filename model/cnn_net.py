import torch
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = True if torch.cuda.is_available() else False

class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super(UnNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        output = tensor.mul(self.std).add(self.mean)
        return output

class cnnDecoder(nn.Module):
    def __init__(self, args):
        super(cnnDecoder, self).__init__()
        self.args=args
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(self.args['num_features'], 512),
            nn.LeakyReLU(negative_slope=self.args['LReLU_negative_slope'], inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=self.args['LReLU_negative_slope'], inplace=True),
            nn.Linear(1024, self.args['input_height'] * self.args['input_width'] * self.args['input_channel']),
            nn.Sigmoid()
        )
        self.mean = torch.tensor(0.1307)
        self.std = torch.tensor(0.3081)
        if(args['USE_CUDA']):
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        self.unnormalize = UnNormalize(self.mean, self.std)

    def forward(self, x, data):
        classes = x.sum(2)
        classes = F.softmax(classes.squeeze(), dim=1)
        _, max_length_indices = classes.max(dim=1)
        
        if(self.args['type']=='plusCR'):
            masked = torch.sparse.torch.eye(10)
            if USE_CUDA:
                masked = masked.cuda()
            masked = masked.index_select(dim=0, index=max_length_indices.squeeze().data)
            t = (x * masked[:, :, None]).view(x.size(0), -1)
            reconstructions = self.reconstraction_layers(t)
            return reconstructions, max_length_indices
        
        elif(self.args['type']=='plusR'):
            x = x.view(x.size(0), -1)
            reconstructions = self.reconstraction_layers(x)
            return reconstructions, max_length_indices

class CNNnet(nn.Module):
    def __init__(self, args, config=None):
        super(CNNnet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(in_channels=config.conv1_in, out_channels=config.conv1_out, kernel_size=config.conv1_kernel_size)
        self.conv2 = nn.Conv2d(in_channels=config.conv2_in, out_channels=config.conv2_out, kernel_size=config.conv2_kernel_size)
        self.fc1 = nn.Linear(in_features=config.fc1_in, out_features=config.fc1_out)
        self.fc2 = nn.Linear(in_features=config.fc2_in, out_features=config.fc2_out)
        self.decoder = cnnDecoder(self.args)
            
        self.mean = torch.tensor(0.1307)
        self.std = torch.tensor(0.3081)
        if(args['USE_CUDA']):
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        self.unnormalize = UnNormalize(self.mean, self.std)
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
            
    def forward(self, data):
        output = F.relu(self.conv2(F.relu(self.conv1(data))))
        output = output.view(data.size(0), -1)
        output = self.fc2(F.relu(self.fc1(output)))
        output = output.view(output.size(0), self.args['num_classes'], -1)
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.args['LAMBDA_class']*self.classification_loss(x, target) + self.args['LAMBDA_recon']*self.reconstruction_loss(data, reconstructions)

    def classification_loss(self, x, labels):
        x = x.sum(dim=2)
        loss = self.criterion(x, labels)
        return loss

    def reconstruction_loss(self, data, reconstructions):
        data_unnormalized = self.decoder.unnormalize(data)
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss
    