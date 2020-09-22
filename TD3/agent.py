import numpy as np
import torch as T
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from network import ActorNetwork, CriticNetwork


class Agent():

    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=0.99, update_actor_interval=2, warmup=1000,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=100, noise=0.1):

        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims,
                                   n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions, name='actor')

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions, name='critic_1')

        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions, name='critic_2')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions,
                                         name='target_actor')

        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions,
                                             name='target_critic_1')

        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions,
                                             name='target_critic_2')

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise,
                                           size=(self.n_actions,))).\
                                               to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                                 dtype=T.float).to(self.actor.device)

        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transitions(state, action,
                                      reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, state_, done =\
            self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        state_ = T.tensor(state_, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)

        target_actions = self.target_actor(state_)
        target_actions = target_actions +\
            T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action[0],
                                 self.max_action[0])

        q1_ = self.target_critic_1(state_, target_actions)
        q2_ = self.target_critic_2(state_, target_actions)

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1(state, self.actor(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() \
                + (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() \
                + (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoints()
        self.critic_1.save_checkpoints()
        self.critic_2.save_checkpoints()
        self.target_actor.save_checkpoints()
        self.target_critic_1.save_checkpoints()
        self.target_critic_2.save_checkpoints()

    def load_models(self):
        self.actor.load_checkpoints()
        self.critic_1.load_checkpoints()
        self.critic_2.load_checkpoints()
        self.target_actor.load_checkpoints()
        self.target_critic_1.load_checkpoints()
        self.target_critic_2.load_checkpoints()
