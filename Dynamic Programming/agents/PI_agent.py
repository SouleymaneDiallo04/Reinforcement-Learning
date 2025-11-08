import numpy as np
import random

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.values = np.zeros(env.grid_size)
        self.policy = {state: random.choice(env.action_space) 
                      for state in env.get_all_states() 
                      if state != env.terminal_state}
        self.policy[env.terminal_state] = None
        self.name = "Policy Iteration"
        self._learn()
    
    def _learn(self):
        iteration = 0
        policy_stable = False
        
        while not policy_stable:
            self._policy_evaluation()
            policy_stable = self._policy_improvement()
            iteration += 1
        
        print(f"Policy Iteration terminée en {iteration} itérations")
    
    def _policy_evaluation(self):
        while True:
            delta = 0
            for i in range(4):
                for j in range(4):
                    state = (i, j)
                    if state == self.env.terminal_state:
                        continue
                    
                    action = self.policy[state]
                    next_state = self._get_next_state(state, action)
                    reward = self.env.rewards[next_state]
                    
                    new_value = reward + self.gamma * self.values[next_state]
                    delta = max(delta, abs(new_value - self.values[state]))
                    self.values[state] = new_value
            
            if delta < 1e-6:
                break
    
    def _policy_improvement(self):
        policy_stable = True
        
        for i in range(4):
            for j in range(4):
                state = (i, j)
                if state == self.env.terminal_state:
                    continue
                
                old_action = self.policy[state]
                
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
                
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable
    
    def _get_next_state(self, state, action):
        move = self.env.actions[action]
        return (
            max(0, min(3, state[0] + move[0])),
            max(0, min(3, state[1] + move[1]))
        )
    
    def act(self, state):
        return self.policy[state]
    
    def update(self, state, action, reward, next_state, done):
        pass