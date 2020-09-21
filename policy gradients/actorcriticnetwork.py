import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorCriticNetwork(nn.Module):

    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.lr = lr
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr)
        if T.cuda.is_available():
            self.device = T.device('cuda:0')
        else:
            self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)


class Agent():

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions,
                 gamma=0.99):
        self.lr = lr
        self.gamma = gamma
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions,
                                               self.fc1_dims, self.fc2_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).\
                to(self.actor_critic.device)
        probability, _ = self.actor_critic(state)
        probability = F.softmax(probability, dim=1)
        action_probs = T.distributions.Categorical(probability)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor([reward], dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic(state)
        _, critic_value_ = self.actor_critic(state_)

        delta = reward + self.gamma * critic_value_ * (1-int(done))
        - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()
