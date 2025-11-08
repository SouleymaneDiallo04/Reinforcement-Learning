import numpy as np
import matplotlib.pyplot as plt

def train_agent(env, agent, episodes=1000, learning=True):
    print(f"=== ENTRAÎNEMENT {agent.name} ===")
    
    rewards_history = []
    steps_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_data = []
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if learning:
                if hasattr(agent, 'update') and not hasattr(agent.update, '__code__'):
                    # Pour Monte Carlo
                    episode_data.append((state, action, reward))
                else:
                    # Pour les autres agents
                    agent.update(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                if learning and hasattr(agent, 'update') and hasattr(agent.update, '__code__'):
                    if agent.update.__code__.co_argcount > 2:  # Pour Monte Carlo
                        agent.update(episode_data)
                
                rewards_history.append(total_reward)
                steps_history.append(steps)
                
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(rewards_history[-100:])
                    avg_steps = np.mean(steps_history[-100:])
                    print(f"Épisode {episode+1}: Reward moyen = {avg_reward:.2f}, "
                          f"Steps moyen = {avg_steps:.1f}")
                break
    
    return rewards_history, steps_history

def evaluate_agent(env, agent, episodes=100):
    rewards_history = []
    steps_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                rewards_history.append(total_reward)
                steps_history.append(steps)
                break
    
    return rewards_history, steps_history

def plot_results(results):
    names = list(results.keys())
    rewards = [results[name]['mean_reward'] for name in names]
    steps = [results[name]['mean_steps'] for name in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bars1 = ax1.bar(names, rewards, yerr=[results[name]['std_reward'] for name in names], 
                   capsize=5, alpha=0.7, color=['red', 'blue', 'green', 'orange', 'purple'])
    ax1.set_ylabel('Reward Moyen')
    ax1.set_title('Comparaison des Rewards par Agent')
    ax1.tick_params(axis='x', rotation=45)
    
    bars2 = ax2.bar(names, steps, yerr=[results[name]['std_steps'] for name in names], 
                   capsize=5, alpha=0.7, color=['red', 'blue', 'green', 'orange', 'purple'])
    ax2.set_ylabel('Steps Moyens')
    ax2.set_title('Comparaison des Steps par Agent')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()