import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import datasets, transforms

import tensorflow as tf

from model import Model
from utils import progress_bar
from args import args
import visdom 


args = args()
torch.manual_seed(args.seed)
if args.instance is None:
    args.instance = 'data_{}_optim_{}_lr_{}_ld_{}_wd_{}_batch-size_{}_seed_{}'.format(args.dataset,args.optim, args.lr,args.lr_decay_interval, args.weight_decay, args.batch_size, args.seed)

if args.visdom:
    vis = visdom.Visdom(env=args.instance)

print(args.instance)

best_acc = 0.0
def train(epoch, net, loader, optim, loss_function):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    print('Epoch {}'.format(epoch))
    for i,(inputs, targets) in enumerate(loader):
        inputs_d, targets_d = inputs.to(args.device), targets.to(args.device)
        outputs = net(inputs_d)
        loss = loss_function(outputs, targets_d)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets_d).sum().item()
        progress_bar(i, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(i+1), 100.*correct/total, correct, total))
    train_loss /= len(loader)
    acc = correct / total * 100
    return train_loss, acc

def test(epoch, net, loader, loss_function):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = loss_function(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    test_loss /= len(loader)
    acc = 100.*correct/total
    tag = 'epoch_{}_loss_{:.4f}_acc_{:.2f}'.format(epoch, test_loss, acc)
    if acc > best_acc:
        best_acc = acc
        save_checkpoint(net, acc, epoch,tag='best_'+tag )
    if epoch % args.checkpoint_interval == 0:
        save_checkpoint(net, acc, epoch,tag=tag)

    return test_loss, acc
 
def save_checkpoint(net, acc, epoch, tag=None):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('checkpoint/{}'.format(args.instance)):
        os.mkdir('checkpoint/{}'.format(args.instance))
    if tag is None:
        filename = 'test' 
    else:
        filename = tag 
    if 'best' in tag:
        os.system('rm ./checkpoint/{}/best_*.t7'.format(args.instance))
    torch.save(state, './checkpoint/{}/{}.t7'.format(args.instance, filename))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        trainset = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(root='~/data/cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        data_size=(3,32,32)
    elif args.dataset == 'MNIST':
        trainloader = torch.utils.data.DataLoader(
            datasets.MNIST('~/data/mnist', train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=args.batch_size, shuffle=True, num_workers=2)


        testloader = torch.utils.data.DataLoader(
            datasets.MNIST('~/data/mnist', train=False, 
                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=args.batch_size, shuffle=True, num_workers=2)
        data_size = (1,28,28)
    else:
        raise NotImplementedError


    net = Model(in_size=data_size).to(args.device)

    epoch=args.epoch

    loss_function = nn.CrossEntropyLoss()
    if args.optim == 'sgd':
        optim = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    if vis:
        loss_win, acc_win = None, None
    for i in range(epoch):
        adjust_learning_rate(optim, i)
        train_loss, train_acc = train(i, net, trainloader, optim, loss_function)
        test_loss, test_acc = test(i, net, testloader, loss_function)
        if vis:
            loss_win, acc_win = visdom_plot(i, train_loss, test_loss, train_acc, test_acc, loss_win,acc_win )


def visdom_plot(epoch, train_loss,test_loss, train_acc, test_acc,  loss_win, acc_win):
    x = np.column_stack((np.arange(epoch, epoch+1), np.arange(epoch, epoch+1)))
    y = np.array([[train_loss, test_loss]])
    z = np.array([[train_acc, test_acc]])
    if loss_win is None:
        loss_win = vis.line(X=x,Y=y,opts=dict(title=args.instance, legend=['train','test']))
        acc_win = vis.line(X=x,Y=z,opts=dict(title=args.instance, legend=['train','test']))
    else:
        loss_win = vis.line(X=x,Y=y, update='append', win=loss_win,opts=dict(title=args.instance, legend=['train','test']))
        acc_win = vis.line(X=x,Y=z, update='append', win=acc_win,opts=dict(title=args.instance, legend=['train','test']))
    return loss_win, acc_win

main()
