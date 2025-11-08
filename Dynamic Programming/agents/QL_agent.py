import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(len(env.action_space)))
        self.name = "Q-Learning"
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            return self.env.action_space[np.argmax(self.Q[state])]
    
    def update(self, state, action, reward, next_state, done):
        action_idx = self.env.action_space.index(action)
        current_q = self.Q[state][action_idx]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        self.Q[state][action_idx] = current_q + self.alpha * (target - current_q)