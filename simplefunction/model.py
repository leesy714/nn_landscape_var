import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import collections

class Linear(nn.Module):
    def __init__(self, in_size ):
        super(Linear, self).__init__()

        self.fc1 = nn.Linear(in_size, 1)

    def forward(self,x):
        x = self.fc1(x)
        return x

    def get_weight_vector(self):
        v1 = self.fc1.weight.cpu().data.numpy().flatten()
        v2 = self.fc1.bias.cpu().data.numpy().flatten()
        return np.concatenate([v1,v2])

    def set_weight_vector(self, vector, device='cpu'):
        n1 = self.fc1.weight.size(0) * self.fc1.weight.size(1)
        n2 = self.fc1.bias.size(0)
        w1 = vector[ : n1 ].reshape(self.fc1.weight.size())
        w2 = vector[n1 : n1 + n2].reshape(self.fc1.bias.size())

        self.fc1.weight = Parameter(torch.Tensor(w1)).to(device)
        self.fc1.bias = Parameter(torch.Tensor(w2)).to(device)


    def filterwisely_normalize(self, vector):
        n1 = self.fc1.weight.size(0) * self.fc1.weight.size(1)
        n2 = self.fc1.bias.size(0)
        w1 = vector[ : n1]
        w2 = vector[n1 : n1 + n2]
        
        
        
        w1_norm = np.linalg.norm(w1)
        layer_norm_1 = np.linalg.norm(self.fc1.weight.data.cpu().numpy())
        w1_norm = w1 / w1_norm * layer_norm_1
        
        
        #w2_norm = np.linalg.norm(w2)
        #layer_norm_2 = np.linalg.norm(self.fc1.bias.data.cpu().numpy())
        #w2_norm = w2 / w2_norm * layer_norm_2
        w2_norm = w2
        
        return np.concatenate([w1_norm, w2_norm])
    
class MLP(nn.Module):
    def __init__(self, in_size, hidden_size=10,activation=F.relu):
        super(MLP, self).__init__()

        if isinstance(hidden_size, collections.Iterable):
            in_ = in_size
            self.fc = []
            for hidden in hidden_size:
                self.fc.append(nn.Linear(in_, hidden))
                in_ = hidden
            self.fc2 = nn.Linear(in_, 1)
        else:
            raise NotImplementedError

        self.activation = activation

    def forward(self,x):
        if isinstance(self.fc, list):
            for module in self.fc:
                x = self.activation(module(x))
            x = self.fc2(x)
        else:
            raise NotImplementedError
        return x

    def get_weight_vector(self):
        if isinstance(self.fc, list):
            v = []
            for module in self.fc:
                v.append(module.weight.cpu().data.numpy().flatten())
            v1 = np.concatenate(v)
        else:
            raise NotImplementedError
            
        v2 = self.fc2.weight.cpu().data.numpy().flatten()
        v3 = self.fc2.bias.cpu().data.numpy().flatten()
        return np.concatenate([v1,v2,v3])

    def set_weight_vector(self, vector, device='cpu'):
        
        if isinstance(self.fc, list):
            n1 = 0
            for module in self.fc:
                n = module.weight.size(0) * module.weight.size(1)
                w = vector[n1 : n1 + n].reshape(module.weight.size())
                module.weight = Parameter(torch.Tensor(w)).to(device)
                n1 += n
        n2 = self.fc2.weight.size(0) * self.fc2.weight.size(1)
        n3 = self.fc2.bias.size(0)
        
        w2 = vector[n1 : n1 + n2].reshape(self.fc2.weight.size())
        w3 = vector[n1 + n2 : n1 + n2 + n3 ].reshape(self.fc2.bias.size())

        self.fc2.weight = Parameter(torch.Tensor(w2)).to(device)
        self.fc2.bias = Parameter(torch.Tensor(w3)).to(device)



    def filterwisely_normalize(self, vector):
        if isinstance(self.fc, list):
            n1 = 0
            w1=[]
            for module in self.fc:
                n = module.weight.size(0) * module.weight.size(1)
                w = vector[n1 : n1 + n]
                w_norm = np.linalg.norm(w)
                layer_norm = np.linalg.norm(module.weight.data.cpu().numpy())
                w_norm = w / w_norm * layer_norm
                w1.append(w_norm)

        w1_norm = np.concatenate(w1)
        
        n2 = self.fc2.weight.size(0) * self.fc2.weight.size(1)
        n3 = self.fc2.bias.size(0)
        w2 = vector[n1 : n1 + n2]
        w3 = vector[n1 + n2 : n1 + n2 + n3]
        
        w2_norm = np.linalg.norm(w2)
        layer_norm_2 = np.linalg.norm(self.fc2.weight.data.cpu().numpy())
        w2_norm = w2 / w2_norm * layer_norm_2
        
        
        w3_norm = np.linalg.norm(w3)
        layer_norm_3 = np.linalg.norm(self.fc2.bias.data.cpu().numpy())
        w3_norm = w3 / w3_norm * layer_norm_3
        #w3_norm = w3 
        
        
        return np.concatenate([w1_norm, w2_norm, w3_norm])
        

def test():

    import itertools
    net = MLP(2,5)
    test_func =(lambda x,y : x*x + y*y)
    
    x,y = np.linspace(-1,1,11), np.linspace(-1,1,11)
    
    X = [(x,y) for (x,y) in itertools.product(x,y)]
    Y = [test_func(*x) for x in X]
    print(Y)


if __name__ == '__main__':
    test()
