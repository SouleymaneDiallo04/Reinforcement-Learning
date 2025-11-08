import sys
import os

# Ajouter le chemin des agents
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from grid_env import GridEnv
from agents import RandomAgent, ValueIterationAgent, PolicyIterationAgent, MonteCarloAgent, QLearningAgent
from utils import train_agent, evaluate_agent, plot_results

def compare_agents(episodes=1000):
    """Compare les performances de tous les agents"""
    agents = [
        RandomAgent,
        ValueIterationAgent, 
        PolicyIterationAgent,
        MonteCarloAgent,
        QLearningAgent
    ]
    
    results = {}
    
    for AgentClass in agents:
        env = GridEnv(deterministic=False)
        agent = AgentClass(env)
        
        print(f"\n{'='*50}")
        print(f"TEST DE {agent.name}")
        print(f"{'='*50}")
        
        if agent.name in ["Value Iteration", "Policy Iteration"]:
            # Agents de planification - évaluation seulement
            rewards, steps = evaluate_agent(env, agent, episodes=100)
        else:
            # Agents d'apprentissage - entraînement nécessaire
            rewards, steps = train_agent(env, agent, episodes=episodes, learning=True)
        
        results[agent.name] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps),
            'std_steps': np.std(steps)
        }
        
        print(f"Résultats {agent.name}:")
        print(f"  Reward moyen: {results[agent.name]['mean_reward']:.3f} ± {results[agent.name]['std_reward']:.3f}")
        print(f"  Steps moyen: {results[agent.name]['mean_steps']:.1f} ± {results[agent.name]['std_steps']:.1f}")
    
    return results

def main():
    print("=== FRAMEWORK RL AVEC MULTIPLES AGENTS ===")
    print("1. Comparer tous les agents")
    print("2. Tester un agent spécifique")
    print("3. Visualiser un agent optimal")
    
    choix = input("Choisissez le mode (1, 2 ou 3): ").strip()
    
    if choix == "1":
        results = compare_agents(episodes=500)
        plot_results(results)
        
    elif choix == "2":
        agent_type = input("Choisissez l'agent (random, vi, pi, mc, q): ").strip().lower()
        
        env = GridEnv(deterministic=False)
        
        agents_map = {
            "random": RandomAgent,
            "vi": ValueIterationAgent,
            "pi": PolicyIterationAgent, 
            "mc": MonteCarloAgent,
            "q": QLearningAgent
        }
        
        if agent_type in agents_map:
            agent = agents_map[agent_type](env)
        else:
            print("Agent non reconnu, utilisation de Random")
            agent = RandomAgent(env)
        
        rewards, steps = train_agent(env, agent, episodes=1000, learning=True)
        
        print(f"\nRésultats finaux {agent.name}:")
        print(f"Reward moyen: {np.mean(rewards):.3f}")
        print(f"Steps moyen: {np.mean(steps):.1f}")
        
    elif choix == "3":
        env = GridEnv(deterministic=True)
        agent = ValueIterationAgent(env)
        
        print("Visualisation de l'agent optimal...")
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render(delay=0.5)
            print(f"Step {steps}: {state} -> {next_state}, Action: {action}, Reward: {reward}")
            
            state = next_state
            
            if done:
                print(f"Terminé! Reward total: {total_reward}, Steps: {steps}")
                break
                
        env.render(delay=3)
        
    else:
        print("Mode par défaut: comparaison rapide")
        results = compare_agents(episodes=100)
        plot_results(results)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    main()