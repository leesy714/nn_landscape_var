import argparse

def args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--instance', default=None, type=str, help='instance name')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')

    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='epoch')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')

    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--seed', default=2018, type=int, help='seed')
    
    args = parser.parse_args()
    return args

