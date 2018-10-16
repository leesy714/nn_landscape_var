import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.nn import  Parameter
import numpy as np


class Model(nn.Module):
    def  __init__(self, in_size=(3,32,32), out=10,strides=(2,2)):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_size[0], 32, 3, stride=strides[0], bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=strides[1], bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, bias=False)
        size = (in_size[1] // strides[0] - 1) // strides[1] - 2
        self.dense = nn.Linear(64 * size * size, 512, bias=False)
        self.out = nn.Linear(512, out)
        self.weight_vector_size=0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.dense(x))
        out = self.out(x)
        return out

    def get_weight_vector(self):
        conv1_w = self.conv1.weight.data.cpu().numpy().flatten()
        conv2_w = self.conv2.weight.data.cpu().numpy().flatten()
        conv3_w = self.conv3.weight.data.cpu().numpy().flatten()
        dense_w = self.dense.weight.data.cpu().numpy().flatten()


        out_w = self.out.weight.data.cpu().numpy().flatten()
        out_b = self.out.bias.data.cpu().numpy().flatten()

        w = [conv1_w, conv2_w, conv3_w, dense_w, out_w, out_b]
        w= np.concatenate(w)
        return w

    def get_weight_vector_size(self):
        if self.weight_vector_size == 0:
            self.weight_vector_size = self.get_weight_vector().shape[0]
        return self.weight_vector_size

    def set_weight_vector(self, vector, device='cuda'):
        n1 = self.conv1.weight.size(0)  * self.conv1.weight.size(1) * self.conv1.weight.size(2) * self.conv1.weight.size(3) 
        n2 = self.conv2.weight.size(0)  * self.conv2.weight.size(1) * self.conv2.weight.size(2) * self.conv2.weight.size(3)
        n3 = self.conv3.weight.size(0)  * self.conv3.weight.size(1) * self.conv3.weight.size(2) * self.conv3.weight.size(3)
        n4 = self.dense.weight.size(0) * self.dense.weight.size(1)

        n5_w = self.out.weight.size(0) * self.out.weight.size(1)
        n5_b = self.out.bias.size(0)

        w1 = vector[ : n1].reshape(self.conv1.weight.size())
        w2 = vector[n1 : n1+n2].reshape(self.conv2.weight.size())
        w3 = vector[n1+n2 : n1+n2+n3].reshape(self.conv3.weight.size())
        w4 = vector[n1+n2+n3:n1+n2+n3+n4].reshape(self.dense.weight.size())

        w5 = vector[n1+n2+n3+n4 : n1+n2+n3+n4+n5_w].reshape(self.out.weight.size())
        b5 = vector[n1+n2+n3+n4+n5_w : ].reshape(self.out.bias.size())


        self.conv1.weight = Parameter(torch.Tensor(w1).to(device))
        self.conv2.weight = Parameter(torch.Tensor(w2).to(device))
        self.conv3.weight = Parameter(torch.Tensor(w3).to(device))
        self.dense.weight = Parameter(torch.Tensor(w4).to(device))

        self.out.weight = Parameter(torch.Tensor(w5).to(device))
        self.out.bias = Parameter(torch.Tensor(b5).to(device))

    def filterwisely_normalize(self, vector):
        n1 = self.conv1.weight.size(0)  * self.conv1.weight.size(1) * self.conv1.weight.size(2) * self.conv1.weight.size(3) 
        n2 = self.conv2.weight.size(0)  * self.conv2.weight.size(1) * self.conv2.weight.size(2) * self.conv2.weight.size(3)
        n3 = self.conv3.weight.size(0)  * self.conv3.weight.size(1) * self.conv3.weight.size(2) * self.conv3.weight.size(3)
        w1 = vector[ : n1].reshape(self.conv1.weight.size())
        w2 = vector[n1 : n1+n2].reshape(self.conv2.weight.size())
        w3 = vector[n1+n2 : n1+n2+n3].reshape(self.conv3.weight.size())

        w1_norm = conv_filter_normalize(self.conv1, w1)
        w2_norm = conv_filter_normalize(self.conv2, w2)
        w3_norm = conv_filter_normalize(self.conv3, w3)
        w = np.concatenate([w1_norm, w2_norm, w3_norm])
        w_rest = vector[n1+n2+n3:]
        return np.concatenate([w,w_rest])

 


def conv_filter_normalize(layer, vector):
    vec_norm = np.linalg.norm(vector, axis=(2,3)).reshape(vector.shape[0], vector.shape[1], 1, 1)
    conv = layer.weight.data.cpu().numpy()
    conv_norm = np.linalg.norm(conv, axis=(2,3)).reshape(conv.shape[0],conv.shape[1], 1,1)
    vec = vector / vec_norm
    vec = vec * conv_norm
    return vec.flatten()

       



def CifarTest():
    import torchvision
    from torchvision import datasets, transforms

    net = Model()
    print(net.get_weight_vector().shape)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))






        

if __name__ == '__main__':
    CifarTest() 
