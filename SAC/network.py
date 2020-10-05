import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class CriticNetwork(nn.Module):

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chckpt_dir='checkpoints'):

        super().__init__()
        self.lr = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chckpt_dir = chckpt_dir
        self.chckpt_file = os.path.join(self.chckpt_dir, self.name + '_sac')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = F.relu(self.fc1(T.cat([state, action], dim=1)))
        action_value = F.relu(self.fc2(action_value))
        q1 = self.q1(action_value)
        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chckpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chckpt_file, map_location='cuda:0'))


class ActorNetwork(nn.Module):

    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 max_action, n_actions, name, chckpt_dir='checkpoints'):

        super().__init__()
        self.lr = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = max_action
        self.n_actions = n_actions
        self.name = name
        self.chckpt_dir = chckpt_dir
        self.chckpt_file = os.path.join(self.chckpt_dir, self.name + '_sac')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, n_actions)
        self.sigma = nn.Linear(self.fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chckpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chckpt_file, map_location='cuda:0'))


class ValueNetwork(nn.Module):

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 name, chckpt_dir='checkpoints'):

        super().__init__()
        self.lr = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.chckpt_dir = chckpt_dir
        self.chckpt_file = os.path.join(self.chckpt_dir, self.name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chckpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chckpt_file, map_location='cuda:0'))
