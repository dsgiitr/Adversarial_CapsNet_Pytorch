import torch
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def train_CapsNet(model, optimizer, train_loader, epoch, args):
    capsule_net = model
    capsule_net.train()
    n_batch = len(train_loader)
    total_loss = 0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        if(args['type'] != 'plusC' and args['type'] != 'plusCR'):
            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        if(args['USE_CUDA']):
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()
        correct = torch.sum(masked == target, 1)
        train_loss = loss.item()
        total_loss += train_loss
        if batch_id % 100 == 0:
            tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, loss: {:.6f}".format(
                epoch,
                args['N_EPOCHS'],
                batch_id + 1,
                n_batch,
                correct / float(args['BATCH_SIZE']),
                train_loss / float(args['BATCH_SIZE'])
                ))
    tqdm.write('Epoch: [{}/{}], train loss: {:.6f}'.format(epoch,args['N_EPOCHS'],total_loss / len(train_loader.dataset)))
    
def test_CapsNet(capsule_net, test_loader, epoch, args):
    capsule_net.eval()
    test_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):
        if(args['type'] != 'plusC' and args['type'] != 'plusCR'):
            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        if args['USE_CUDA']:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.item()
        correct += torch.sum(masked == target)

    tqdm.write(
        "Epoch: [{}/{}], test accuracy: {:.6f}, loss: {:.6f}".format(epoch, args['N_EPOCHS'], correct / len(test_loader.dataset),
                                                                  test_loss / len(test_loader)))

