import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

from model import Model
from utils import progress_bar
from args import args


args = args()
torch.manual_seed(args.seed)

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
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        save_checkpoint(net, acc, epoch)
 
def save_checkpoint(net, acc, epoch):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/{}.t7'.format(args.instance))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
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


    net = Model(in_size=data_size).to(args.device)

    epoch=args.epoch

    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for i in range(epoch):
        adjust_learning_rate(optim, i)
        train(i, net, trainloader, optim, loss_function)
        test(i, net, testloader, loss_function)

main()
