import torch
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from livelossplot import PlotLosses
import matplotlib.pyplot as plt

def train_clean(net, optimizer, dataloader, args):
    liveloss = PlotLosses()
    if(args['USE_SCHEDULER']):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args['sched_milestones'], gamma=args['sched_gamma'])
        
    for e in range(args['N_EPOCHS']):
        logs={}
        for phase in ['train', 'val']:
            
            prefix = ''
            if(phase == 'train'):
                net.train()
            else:
                net.eval()
                prefix = 'val_'
            
            n_samples = len(dataloader[phase].dataset)
            n_batches = len(dataloader[phase])
            running_loss = 0.0
            running_acc = 0.0
            for batch_id, (data, target) in enumerate(tqdm(dataloader[phase])):
                
                if(args['USE_CUDA']):
                    data, target = data.cuda(), target.cuda()
                output, reconstructions, masked = net(data)
                loss = net.loss(data, output, target, reconstructions)
                if(phase == 'train'):
                    if(batch_id == n_batches-1):
                        img1 = data[0].reshape(28,28).detach().cpu().numpy()
                        img2 = reconstructions[0].reshape(28, 28).detach().cpu().numpy()
                        weight = net.decoder.reconstraction_layers[0].weight[0][:3]
                        grad = net.decoder.reconstraction_layers[0].weight.grad[0][:3].data
                        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                running_acc += torch.sum(masked == target).item()
                running_loss += loss.item()

                
            logs[prefix + 'loss'] = running_loss / float(n_samples)
            logs[prefix + 'accuracy'] = running_acc / float(n_samples)
            
            #Scheduler
            if(args['USE_SCHEDULER'] and phase=='train'):
                scheduler.step()
                for param in optimizer.param_groups:
                    print("LR for the epoch is:", param['lr'])
        
        liveloss.update(logs)
        liveloss.send()
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(img1)
        axarr[1].imshow(img2)
        plt.show()
        print("Weights of Reconstruction Layer:", weight)
        print("Grads of Reconstruction Layer:", grad)
    
        
        
    
def test_clean(net, test_loader, args):
    net.eval()
    test_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):
        if args['USE_CUDA']:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = net(data)
        loss = net.loss(data, output, target, reconstructions)

        test_loss += loss.item()
        correct += torch.sum(masked == target)

    tqdm.write(
        "Epoch: [{}/{}], test accuracy: {:.6f}, loss: {:.6f}".format(epoch, args['N_EPOCHS'], correct / len(test_loader.dataset),
                                                                  test_loss / len(test_loader)))

