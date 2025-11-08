import numpy as np
import random
from collections import defaultdict

class MonteCarloAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(len(env.action_space)))
        self.returns = defaultdict(list)
        self.visit_count = defaultdict(lambda: np.zeros(len(env.action_space)))
        self.name = "Monte Carlo"
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            state_actions = self.Q[state]
            return self.env.action_space[np.argmax(state_actions)]
    
    def update(self, episode):
        states, actions, rewards = zip(*episode)
        G = 0
        
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            if (state, action) not in [(s, a) for s, a, _ in episode[:t]]:
                action_idx = self.env.action_space.index(action)
                self.returns[(state, action)].append(G)
                self.Q[state][action_idx] = np.mean(self.returns[(state, action)])
                self.visit_count[state][action_idx] += 1