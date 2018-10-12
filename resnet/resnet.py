'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.nn import  Parameter

import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            #self.shortcut = nn.Sequential(
            #    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            #    nn.BatchNorm2d(self.expansion*planes)
            #)
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

        self.weight_vector_size = 0

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out = F.relu(self.conv1(x))
        #out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def get_weight_vector(self):
        conv1_w = self.conv1.weight.data.cpu().numpy().flatten()
        conv2_w = self.conv2.weight.data.cpu().numpy().flatten()

        w = np.concatenate((conv1_w, conv2_w ))
        if isinstance(self.shortcut, nn.Conv2d):
            shortcut_w = self.shortcut.weight.data.cpu().numpy().flatten()
            w = np.concatenate((w, shortcut_w))

        return w

    def get_weight_vector_size(self):
        if self.weight_vector_size == 0:
            self.weight_vector_size = self.get_weight_vector().shape[0]
        return self.weight_vector_size

    def set_weight_vector(self, vector):
        n = self.conv1.weight.size(0)  * self.conv1.weight.size(1) * self.conv1.weight.size(2) * self.conv1.weight.size(3) 
        n2 = self.conv2.weight.size(0)  * self.conv2.weight.size(1) * self.conv2.weight.size(2) * self.conv2.weight.size(3)
        w1 = vector[ : n].reshape(self.conv1.weight.size())
        w2 = vector[n : n + n2].reshape(self.conv2.weight.size())
        self.conv1.weight = Parameter(torch.Tensor(w1).to('cuda'))
        self.conv2.weight = Parameter(torch.Tensor(w2).to('cuda'))

        if isinstance(self.shortcut, nn.Conv2d):
            w3 = vector[n + n2 : ].reshape(self.shortcut.weight.size())
            self.shortcut.weight = Parameter(torch.Tensor(w3).to('cuda'))

    def filterwisely_normalize(self, vector):
        n = self.conv1.weight.size(0)  * self.conv1.weight.size(1) * self.conv1.weight.size(2) * self.conv1.weight.size(3) 
        n2 = self.conv2.weight.size(0)  * self.conv2.weight.size(1) * self.conv2.weight.size(2) * self.conv2.weight.size(3)

        w1 = vector[:n].reshape(self.conv1.weight.size())
        w2 = vector[n : n + n2].reshape(self.conv2.weight.size())

        w1_norm = conv_filter_normalize(self.conv1, w1)
        w2_norm = conv_filter_normalize(self.conv2, w2)
        w = np.concatenate((w1_norm, w2_norm))
        if isinstance(self.shortcut, nn.Conv2d):
            w3 = vector[n + n2 : ].reshape(self.shortcut.weight.size())
            w3_norm = conv_filter_normalize(self.shortcut, w3)
            w = np.concatenate((w, w3_norm))
        return w

class BasicBlockNoShort(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.weight_vector_size = 0

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out = F.relu(self.conv1(x))
        #out = self.conv2(out)
        out = F.relu(out)
        return out

    def get_weight_vector(self):
        conv1_w = self.conv1.weight.data.cpu().numpy().flatten()
        conv2_w = self.conv2.weight.data.cpu().numpy().flatten()

        w = np.concatenate((conv1_w, conv2_w ))
        return w

    def get_weight_vector_size(self):
        if self.weight_vector_size == 0:
            self.weight_vector_size = self.get_weight_vector().shape[0]
        return self.weight_vector_size

    def set_weight_vector(self, vector):
        n = self.conv1.weight.size(0)  * self.conv1.weight.size(1) * self.conv1.weight.size(2) * self.conv1.weight.size(3) 
        n2 = self.conv2.weight.size(0)  * self.conv2.weight.size(1) * self.conv2.weight.size(2) * self.conv2.weight.size(3)
        w1 = vector[ : n].reshape(self.conv1.weight.size())
        w2 = vector[n : n + n2].reshape(self.conv2.weight.size())
        self.conv1.weight = Parameter(torch.Tensor(w1).to('cuda'))
        self.conv2.weight = Parameter(torch.Tensor(w2).to('cuda'))


    def filterwisely_normalize(self, vector):
        n = self.conv1.weight.size(0)  * self.conv1.weight.size(1) * self.conv1.weight.size(2) * self.conv1.weight.size(3) 
        n2 = self.conv2.weight.size(0)  * self.conv2.weight.size(1) * self.conv2.weight.size(2) * self.conv2.weight.size(3)

        w1 = vector[:n].reshape(self.conv1.weight.size())
        w2 = vector[n : n + n2].reshape(self.conv2.weight.size())

        w1_norm = conv_filter_normalize(self.conv1, w1)
        w2_norm = conv_filter_normalize(self.conv2, w2)
        w = np.concatenate((w1_norm, w2_norm))
        return w


        




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def get_weight_vector(self):
        conv1_w = self.conv1.weight.data.cpu().numpy().flatten()
        wv = [conv1_w]
        for i, layer in self.layer1.named_children():
            if isinstance(layer, BasicBlock):
                wv.append(layer.get_weight_vector())
        for i, layer in self.layer2.named_children():
            if isinstance(layer, BasicBlock):
                wv.append(layer.get_weight_vector())
        for i, layer in self.layer3.named_children():
            if isinstance(layer, BasicBlock):
                wv.append(layer.get_weight_vector())
        for i, layer in self.layer4.named_children():
            if isinstance(layer, BasicBlock):
                wv.append(layer.get_weight_vector())
        linear_w = self.linear.weight.data.cpu().numpy().flatten()
        linear_b = self.linear.bias.data.cpu().numpy().flatten()
        wv.append(linear_w)
        wv.append(linear_b)
        w = np.concatenate(wv)
        return w

    def set_weight_vector(self, vector):
        n = self.conv1.weight.size(0)  * self.conv1.weight.size(1) * self.conv1.weight.size(2) * self.conv1.weight.size(3) 
        w1 = vector[:n].reshape(self.conv1.weight.size())
        self.conv1.weight = Parameter(torch.Tensor(w1).to('cuda'))
        for i, layer in self.layer1.named_children():
            if isinstance(layer, BasicBlock):
                length = layer.get_weight_vector_size()
                w = vector[n : n + length]
                layer.set_weight_vector(w)
                n += length
        for i, layer in self.layer2.named_children():
            if isinstance(layer, BasicBlock):
                length = layer.get_weight_vector_size()
                w = vector[n : n + length]
                layer.set_weight_vector(w)
                n += length
        for i, layer in self.layer3.named_children():
            if isinstance(layer, BasicBlock):
                length = layer.get_weight_vector_size()
                w = vector[n : n + length]
                layer.set_weight_vector(w)
                n += length
        for i, layer in self.layer4.named_children():
            if isinstance(layer, BasicBlock):
                length = layer.get_weight_vector_size()
                w = vector[n : n + length]
                layer.set_weight_vector(w)
                n += length

        length = self.linear.weight.size(0)  * self.linear.weight.size(1) 
        length2 = self.linear.bias.size(0) 
        w = vector[n : n+length].reshape(self.linear.weight.size())
        n+= length
        w2 = vector[n : n+ length2].reshape(self.linear.bias.size())
        n+= length2
        self.linear.weight = Parameter(torch.Tensor(w).to('cuda'))
        self.linear.bias = Parameter(torch.Tensor(w2).to('cuda'))

    def filterwisely_normalize(self,vector):
        wv = []
        n = self.conv1.weight.size(0)  * self.conv1.weight.size(1) * self.conv1.weight.size(2) * self.conv1.weight.size(3) 
        w1 = vector[:n].reshape(self.conv1.weight.size())
        w1_norm = conv_filter_normalize(self.conv1, w1)
        wv.append(w1_norm)

        for i, layer in self.layer1.named_children():
            if isinstance(layer, BasicBlock):
                length = layer.get_weight_vector_size()
                w = vector[n : n + length]
                n += length

                w_norm = layer.filterwisely_normalize(w)
                wv.append(w_norm)
        for i, layer in self.layer2.named_children():
            if isinstance(layer, BasicBlock):
                length = layer.get_weight_vector_size()
                w = vector[n : n + length]
                n += length

                w_norm = layer.filterwisely_normalize(w)
                wv.append(w_norm)
        for i, layer in self.layer3.named_children():
            if isinstance(layer, BasicBlock):
                length = layer.get_weight_vector_size()
                w = vector[n : n + length]
                n += length

                w_norm = layer.filterwisely_normalize(w)
                wv.append(w_norm)
        for i, layer in self.layer4.named_children():
            if isinstance(layer, BasicBlock):
                length = layer.get_weight_vector_size()
                w = vector[n : n + length]
                n += length

                w_norm = layer.filterwisely_normalize(w)
                wv.append(w_norm)

        length = self.linear.weight.size(0)  * self.linear.weight.size(1) 
        length2 = self.linear.bias.size(0) 
        w = vector[n : n+length].reshape(self.linear.weight.size())
        n+= length
        w2 = vector[n : n+ length2].reshape(self.linear.bias.size())
        n+= length2
        wv.append(w.flatten())
        wv.append(w2.flatten())
        w = np.concatenate(wv)
        return w

 









        
        

def conv_filter_normalize(layer, vector):
    vec_norm = np.linalg.norm(vector, axis=(2,3)).reshape(vector.shape[0], vector.shape[1], 1, 1)
    conv = layer.weight.data.cpu().numpy()
    conv_norm = np.linalg.norm(conv, axis=(2,3)).reshape(conv.shape[0],conv.shape[1], 1,1)
    vec = vector / vec_norm
    vec = vec * conv_norm
    return vec.flatten()

    











def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNetNoShort18():
    return ResNet(BasicBlockNoShort, [2,2,2,2])


def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])
'''
def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
'''

def test():
    net = ResNet18()
    zero_vector = np.zeros(net.get_weight_vector().shape)
    v = np.random.normal(zero_vector)

    
    print(net.get_weight_vector().shape)
    print(net.filterwisely_normalize(v).shape)



test()

