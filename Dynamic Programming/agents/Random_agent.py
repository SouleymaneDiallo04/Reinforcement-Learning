import random

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.name = "Random"
    
    def act(self, state=None):
        return random.choice(self.env.action_space)
    
    def update(self, state, action, reward, next_state, done):
        pass  # Pas d'apprentissage