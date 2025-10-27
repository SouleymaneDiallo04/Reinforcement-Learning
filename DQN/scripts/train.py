import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environments.gridworld import GridWorld
from agents.dqn_agent import DQNAgent
from utils.visualization import Visualizer
from utils.config import ExperimentConfig
import numpy as np

def train_experiment(use_replay=True, episodes=500, model_path="models/dqn_model.pth"):
    """Train a single DQN agent with or without experience replay"""
    
    env = GridWorld()
    agent = DQNAgent(
        state_size=4, 
        action_size=4, 
        use_replay=use_replay,
        model_path=model_path
    )
    
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            if agent.use_replay:
                agent.remember(state, action, reward, next_state, done)
                agent.replay(agent.batch_size)
            else:
                agent.learn(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done or steps > 100:
                break
        
        # Update epsilon after each episode
        agent.update_epsilon()
        scores.append(total_reward)
        
        if episode % 100 == 0:
            method = "avec Replay" if use_replay else "sans Replay"
            print(f"{method} - Episode {episode}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # Save final model
    agent.save_checkpoint()
    return scores, agent

def main():
    print("🚀 Starting GridWorld DQN Training...")
    
    # Train both versions
    print("\n" + "="*50)
    print("Training DQN WITHOUT Experience Replay")
    print("="*50)
    scores_simple, agent_simple = train_experiment(
        use_replay=False,
        **ExperimentConfig.get_simple_config()
    )
    
    print("\n" + "="*50)
    print("Training DQN WITH Experience Replay")
    print("="*50)
    scores_replay, agent_replay = train_experiment(
        use_replay=True,
        **ExperimentConfig.get_replay_config()
    )
    
    # Visualize results
    visualizer = Visualizer()
    visualizer.plot_training_results(
        scores_simple, 
        scores_replay,
        save_path="results/training_comparison.png"
    )
    
    print("\n✅ Training completed!")
    print("📊 Results saved in 'results/' directory")

if __name__ == "__main__":
    main()