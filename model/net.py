import torch
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]        
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return self.squash(u)

    def squash(self, input_tensor, epsilon=1e-7):
        squared_norm = (input_tensor ** 2 + epsilon).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = torch.zeros(1, self.num_routes, self.num_capsules, 1)
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor, epsilon=1e-7):
        squared_norm = (input_tensor ** 2 + epsilon).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, args, input_width=28, input_height=28, input_channel=1):
        super(Decoder, self).__init__()
        self.args=args
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(args['num_features'], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.args['input_channel'], self.args['input_width'], self.args['input_height']),
            nn.Sigmoid()
        )
        self.mean = torch.tensor(0.1307)
        self.std = torch.tensor(0.3081)
        if(args['USE_CUDA']):
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        self.unnormalize = UnNormalize(self.mean, self.std)

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes.squeeze(), dim=1)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze().data))
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.args['input_channel'], self.args['input_width'], self.args['input_height'])
        return reconstructions, masked

class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super(UnNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        output = tensor.mul(self.std).add(self.mean)
        return output


class leaky_Decoder(nn.Module):
    def __init__(self, args):
        super(leaky_Decoder, self).__init__()
        self.args=args
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(args['num_features'], 512),
            nn.LeakyReLU(negative_slope=self.args['LReLU_negative_slope'], inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=self.args['LReLU_negative_slope'], inplace=True),
            nn.Linear(1024, self.args['input_channel']* self.args['input_width']* self.args['input_height']),
            nn.Sigmoid()
        )
        self.mean = torch.tensor(0.1307)
        self.std = torch.tensor(0.3081)
        if(args['USE_CUDA']):
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        self.unnormalize = UnNormalize(self.mean, self.std)
        

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes.squeeze(), dim=1)

        _, max_length_indices = classes.max(dim=1)
        masked = torch.sparse.torch.eye(10)
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze().data)
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.args['input_channel'], self.args['input_width'], self.args['input_height'])
        return reconstructions, max_length_indices
    
    def target_recon(self, x, target):
        masked = torch.sparse.torch.eye(10)
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=target.squeeze().data)
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.args['input_channel'], self.args['input_width'], self.args['input_height'])
        return reconstructions


class CapsNet(nn.Module):
    def __init__(self, args,config=None):
        super(CapsNet, self).__init__()
        self.args = args
        if config:
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels,
                                                config.pc_kernel_size, config.pc_num_routes)
            self.digit_capsules_1 = DigitCaps(config.dc_num_capsules, config.dc_num_routes, config.dc_in_channels,
                                            config.dc_out_channels)
            self.digit_capsules_2 = DigitCaps(config.dc_2_num_capsules, config.dc_2_num_routes, config.dc_2_in_channels,
                                            config.dc_2_out_channels)
            self.decoder = leaky_Decoder(self.args)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules_1 = DigitCaps()
            self.decoder = Decoder(self.args)
            
        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.primary_capsules(self.conv_layer(data))
        output = self.digit_capsules_1(output).squeeze(-1)
        output = self.digit_capsules_2(output)
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.args['LAMBDA_class']*self.classification_loss(x, target) + self.args['LAMBDA_recon']*self.reconstruction_loss(data, reconstructions)

    def classification_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)
        sparse_indices = torch.sparse.torch.eye(10)
        if(self.args['USE_CUDA']):
            sparse_indices = sparse_indices.cuda()
            labels = labels.cuda()
        labels = sparse_indices.index_select(dim=0, index=labels)
        
        
        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        data_unnormalized = self.decoder.unnormalize(data)
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss