import torch
import torch.nn as nn
import torch.nn.functional as F

from  torch.nn import  Parameter
from .distributions import Categorical, DiagGaussian
from .utils import init, init_normc_
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def get_weight_vector(self):
        w = self.base.get_weight_vector()
        w2 = self.dist.get_weight_vector()
        return np.concatenate((w,w2))


    def set_weight_vector(self, vector, device):
        n = self.base.set_weight_vector(vector, device)

        w_rest = vector[n:]
        
        self.dist.set_weight_vector(w_rest, device)

    def filterwisely_normalize(self, vector):
        w1 = self.base.filterwisely_normalize(vector)
        n = len(w1)
        w_rest = vector[n:]

        n5 = self.dist.linear.weight.size(0) * self.dist.linear.weight.size(1)
        n6 = self.dist.linear.bias.size(0)

        w5 = vector[n : n+n5]
        
        w5_norm = np.linalg.norm(w5)
        layer_norm_5 = np.linalg.norm(self.dist.linear.weight.data.cpu().numpy())
        w5_norm = w5 / w5_norm * layer_norm_5
        
        b5 = vector[n+n5 : ]
        b5_norm = np.linalg.norm(b5)
        layer_norm_5 = np.linalg.norm(self.dist.linear.bias.data.cpu().numpy())
        b5_norm = b5 / b5_norm * layer_norm_5
        w2 = np.concatenate([w5_norm, b5_norm])
        return np.concatenate([w1,w2])

 








class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)


        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, bias=False)
        self.dense = nn.Linear(64 * 7 * 7, hidden_size, bias=False)
        self.critic_linear = nn.Linear(hidden_size, 1)


        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = F.relu(self.conv1(inputs / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

    def get_weight_vector(self):
        conv1_w = self.conv1.weight.data.cpu().numpy().flatten()
        conv2_w = self.conv2.weight.data.cpu().numpy().flatten()
        conv3_w = self.conv3.weight.data.cpu().numpy().flatten()

        dense_w = self.dense.weight.data.cpu().numpy().flatten()
        critic_w = self.critic_linear.weight.data.cpu().numpy().flatten()
        critic_b = self.critic_linear.bias.data.cpu().numpy().flatten()


        w = [conv1_w, conv2_w, conv3_w, dense_w, critic_w, critic_b]
        w= np.concatenate(w)
        return w

    def set_weight_vector(self,vector,device):
        n1 = self.conv1.weight.size(0)  * self.conv1.weight.size(1) * self.conv1.weight.size(2) * self.conv1.weight.size(3) 
        n2 = self.conv2.weight.size(0)  * self.conv2.weight.size(1) * self.conv2.weight.size(2) * self.conv2.weight.size(3)
        n3 = self.conv3.weight.size(0)  * self.conv3.weight.size(1) * self.conv3.weight.size(2) * self.conv3.weight.size(3)
        n4 = self.dense.weight.size(0) * self.dense.weight.size(1)

        w1 = vector[ : n1].reshape(self.conv1.weight.size())
        w2 = vector[n1 : n1+n2].reshape(self.conv2.weight.size())
        w3 = vector[n1+n2 : n1+n2+n3].reshape(self.conv3.weight.size())
        w4 = vector[n1+n2+n3:n1+n2+n3+n4].reshape(self.dense.weight.size())


        n5 = self.critic_linear.weight.size(0) * self.critic_linear.weight.size(1)
        n6 = self.critic_linear.bias.size(0)

        w5 = vector[n1+n2+n3+n4 : n1+n2+n3+n4+n5].reshape(self.critic_linear.weight.size())
        w6 = vector[n1+n2+n3+n4+n5 : n1+n2+n3+n4+n5+n6 ].reshape(self.critic_linear.bias.size())


        self.conv1.weight = Parameter(torch.Tensor(w1).to(device))
        self.conv2.weight = Parameter(torch.Tensor(w2).to(device))
        self.conv3.weight = Parameter(torch.Tensor(w3).to(device))
        self.dense.weight = Parameter(torch.Tensor(w4).to(device))

        self.critic_linear.weight = Parameter(torch.Tensor(w5).to(device))
        self.critic_linear.bias = Parameter(torch.Tensor(w6).to(device))

        return n1+n2+n3+n4+n5+n6

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

        
        n4 = self.dense.weight.size(0) * self.dense.weight.size(1)

        n5_w = self.critic_linear.weight.size(0) * self.critic_linear.weight.size(1)
        n5_b = self.critic_linear.bias.size(0)

        w4 = vector[n1+n2+n3 : n1+n2+n3+n4]
        w4_norm = np.linalg.norm(w4)
        layer_norm_4 = np.linalg.norm(self.dense.weight.data.cpu().numpy())
        w4_norm = w4 / w4_norm * layer_norm_4
        
        w5 = vector[n1+n2+n3+n4 : n1+n2+n3+n4+n5_w]
        w5_norm = np.linalg.norm(w5)
        layer_norm_5 = np.linalg.norm(self.critic_linear.weight.data.cpu().numpy())
        w5_norm = w5 / w5_norm * layer_norm_5
        
        b5 = vector[n1+n2+n3+n4+n5_w : n1+n2+n3+n4+n5_w+n5_b]
        b5_norm = np.linalg.norm(b5)
        layer_norm_5 = np.linalg.norm(self.critic_linear.bias.data.cpu().numpy())
        b5_norm = b5 / b5_norm * layer_norm_5
        
        
        w_rest = np.concatenate([w4_norm, w5_norm, b5_norm])
        
        return np.concatenate([w,w_rest])


def conv_filter_normalize(layer, vector):
    vec_norm = np.linalg.norm(vector, axis=(2,3)).reshape(vector.shape[0], vector.shape[1], 1, 1)
    conv = layer.weight.data.cpu().numpy()
    conv_norm = np.linalg.norm(conv, axis=(2,3)).reshape(conv.shape[0],conv.shape[1], 1,1)
    vec = vector / vec_norm
    vec = vec * conv_norm
    return vec.flatten()

       


        


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

def test():
    from envs import make_vec_envs
    envs = make_vec_envs('PongNoFrameskip-v4', 2018, 2, 0.99, './gym/', True, 'cuda:0', False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': False})
    print(actor_critic.get_weight_vector().shape)
    print(sum(p.numel() for p in actor_critic.parameters() if p.requires_grad))

    zero = np.zeros(actor_critic.get_weight_vector().shape)
    actor_critic.set_weight_vector(zero, device='cuda:0')

if __name__ == '__main__':
    test()
 
