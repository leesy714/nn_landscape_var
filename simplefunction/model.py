import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size=10):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_weight_vector(self):
        v1 = self.fc1.weight.cpu().data.numpy().flatten()
        v2 = self.fc2.weight.cpu().data.numpy().flatten()
        v3 = self.fc2.bias.cpu().data.numpy().flatten()
        return np.concatenate([v1,v2,v3])

    def set_weight_vector(self, vector, device='cpu'):
        n1 = self.fc1.weight.size(0) * self.fc1.weight.size(1)
        n2 = self.fc2.weight.size(0) * self.fc2.weight.size(1)
        n3 = self.fc2.bias.size(0)
        w1 = vector[ : n1].reshape(self.fc1.weight.size())
        w2 = vector[n1 : n1 + n2].reshape(self.fc2.weight.size())
        w3 = vector[n1 + n2 : n1 + n2 + n3].reshape(self.fc2.bias.size())

        self.fc1.weight = Parameter(torch.Tensor(w1)).to(device)
        self.fc2.weight = Parameter(torch.Tensor(w2)).to(device)
        self.fc2.bias = Parameter(torch.Tensor(w3)).to(device)




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
