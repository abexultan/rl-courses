import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        self.chckpt_file = os.path.join(self.chckpt_dir, self.name + '_td3')

        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions,
                             self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = F.relu(self.fc1(T.cat([state, action], dim=1)))
        q1_action_value = F.relu(self.fc2(q1_action_value))
        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoints(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.chckpt_file)

    def load_checkpoints(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chckpt_file))


class ActorNetwork(nn.Module):

    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chckpt_dir='checkpoints'):

        super().__init__()
        self.lr = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chckpt_dir = chckpt_dir
        self.chckpt_file = os.path.join(self.chckpt_dir, self.name + '_td3')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        if T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = T.tanh(self.mu(prob))

        return mu

    def save_checkpoints(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.chckpt_file)

    def load_checkpoints(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chckpt_file))
