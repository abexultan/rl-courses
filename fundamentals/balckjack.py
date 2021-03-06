import numpy as np
import gym 

class Agent():

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.V = {}
        self.sum_space = [i for i in range(4, 22)]
        self.dealer_show_card_space = [i for i in range(1, 11)]
        self.ace_space = [False, True]
        self.action_space = [0, 1] #stick or hit 
        self.state_space = []
        self.returns = {}
        self.states_visited = {} #first visit or not
        self.memory = []
        
        self.init_vals()

    def init_vals(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    self.V[(total, card, ace)] = 0
                    self.returns[(total, card, ace)] = []
                    self.states_visited[(total, card, ace)] = 0
                    self.state_space.append((total, card, ace))

    def policy(self, state):
        total, _, _ = state
        action = 0 if total >= 20 else 1
        return action

    def update_V(self):
        for idt, (state, _) in enumerate(self.memory):
            G = 0
            if self.states_visited[state] == 0:
                self.states_visited[state] += 1
                discount = 1
                for _, (_, reward) in enumerate(self.memory[idt:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[state].append(G)

        for state, _ in self.memory:
            self.V[state] = np.mean(self.returns[state])

        for state in self.state_space:
            self.states_visited[state] = 0
        
        self.memory = []
    



if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    agent = Agent()
    n_episodes = 500000

    for i in range(n_episodes):
        if i % 5000 == 0:
            print('starting episode ', i)
        observation = env.reset()
        done = False
        while not done:
            action = agent.policy(observation)
            observation_, reward, done, info = env.step(action)
            agent.memory.append((observation, reward))
            observation = observation_
        agent.update_V()
    print(agent.V[(21, 3, True)])
    print(agent.V[(4, 1, False)])