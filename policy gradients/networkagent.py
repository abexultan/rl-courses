import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):

    def __init__(self, lr, input_dims, n_actions, fc1_dims=128):
        super().__init__()
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc1_dims)
        self.fc3 = nn.Linear(self.fc1_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0') if T.cuda.is_available()\
            else T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class PolicyGradientAgent():

    def __init__(self, lr, input_dims, n_actions=4, gamma=0.99):

        self.lr = lr
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.policy.device)
        probabilities = F.softmax(self.policy(state))
        action_probabilities = T.distributions.Categorical(probabilities)
        action = action_probabilities.sample()
        log_probabilities = action_probabilities.log_prob(action)
        self.action_memory.append(log_probabilities)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.zero_grad()

        G = np.zeros_like(self.reward_memory)

        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        G = T.tensor(G).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
