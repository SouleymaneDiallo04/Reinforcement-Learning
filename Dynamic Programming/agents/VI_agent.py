import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.values = np.zeros(env.grid_size)
        self.policy = {}
        self.name = "Value Iteration"
        self._learn()
    
    def _learn(self):
        iteration = 0
        while True:
            delta = 0
            new_values = np.copy(self.values)
            
            for i in range(4):
                for j in range(4):
                    state = (i, j)
                    if state == self.env.terminal_state:
                        continue
                    
                    action_values = []
                    for action in self.env.action_space:
                        next_state = self._get_next_state(state, action)
                        reward = self.env.rewards[next_state]
                        value = reward + self.gamma * self.values[next_state]
                        action_values.append(value)
                    
                    if action_values:
                        new_values[state] = max(action_values)
                    
                    delta = max(delta, abs(new_values[state] - self.values[state]))
            
            self.values = new_values
            iteration += 1
            
            if delta < self.theta:
                break
        
        self._extract_policy()
        print(f"Value Iteration terminée en {iteration} itérations")
    
    def _get_next_state(self, state, action):
        move = self.env.actions[action]
        return (
            max(0, min(3, state[0] + move[0])),
            max(0, min(3, state[1] + move[1]))
        )
    
    def _extract_policy(self):
        for i in range(4):
            for j in range(4):
                state = (i, j)
                if state == self.env.terminal_state:
                    self.policy[state] = None
                    continue
                
                best_action = None
                best_value = -float('inf')
                
                for action in self.env.action_space:
                    next_state = self._get_next_state(state, action)
                    reward = self.env.rewards[next_state]
                    value = reward + self.gamma * self.values[next_state]
                    
                    if value > best_value:
                        best_value = value
                        best_action = action
                
                self.policy[state] = best_action
    
    def act(self, state):
        return self.policy[state]
    
    def update(self, state, action, reward, next_state, done):
        pass