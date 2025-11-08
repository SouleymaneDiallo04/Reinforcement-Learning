
Ce dépôt regroupe deux projets complémentaires en apprentissage par renforcement, allant des méthodes classiques aux approches deep learning modernes.

## 📁 Structure du Dépôt

### 1. 🎯 **RL Framework** - Algorithmes Classiques
*Dossier : `rl_framework/`*

Un framework complet implémentant les algorithmes fondamentaux du RL sur un environnement GridWorld statique.

```
rl_framework/
├── 📄 grid_env.py              # Environnement de grille 4x4 standard
├── 📄 main.py                  # Programme principal de comparaison
├── 📄 utils.py                 # Visualisation et métriques
│
└── 📂 agents/                  # Collection d'agents RL
    ├── 📄 random_agent.py              # Agent aléatoire (baseline)
    ├── 📄 value_iteration_agent.py     # Value Iteration (DP)
    ├── 📄 policy_iteration_agent.py    # Policy Iteration (DP)
    ├── 📄 monte_carlo_agent.py         # Monte Carlo
    └── 📄 q_learning_agent.py          # Q-Learning (TD)
```

### 2. 🚀 **Deep Q-Network** - Approche Deep Learning  
*Dossier : `gridworld_dqn/`*

Implémentation de DQN et variantes pour un environnement GridWorld dynamique avec objectifs changeants.

```
gridworld_dqn/
├── agents/
│   ├── base_agent.py              # Classe abstraite pour les agents
│   └── dqn_agent.py               # Implémentation DQN avec/sans Experience Replay
├── environments/
│   └── gridworld.py               # Environnement à objectifs dynamiques
├── models/
│   └── networks.py                # Architectures de réseaux neuronaux
├── scripts/
│   ├── train.py                   # Script d'entraînement
│   └── test.py                    # Script de test
└── utils/
    ├── config.py                  # Configuration
    └── visualization.py           # Visualisation
```

## 🤖 Algorithmes Implémentés

### 🔹 Méthodes Classiques (rl_framework)

| Agent | Algorithme | Type | Description |
|-------|------------|------|-------------|
| 🔀 Random Agent | Aléatoire | Baseline | Mouvements uniformément aléatoires |
| 🧭 Policy Iteration | Programmation Dynamique | On-policy | Évaluation/amélioration itérative |
| 📊 Value Iteration | Programmation Dynamique | Model-based | Convergence vers la fonction valeur optimale |
| 🎲 Monte Carlo | First-Visit MC | Model-free | Apprentissage par épisodes complets |
| ⚡ Q-Learning | TD Learning | Off-policy | Mise à jour incrémentale des Q-values |

### 🔹 Approches Deep Learning (gridworld_dqn)

| Algorithme | Description | Expérience Replay |
|------------|-------------|-------------------|
| **Deep Q-Network (DQN)** | Approximation des Q-values par réseau neuronal | ✅ Supporté |
| **Dueling DQN** | Séparation valeur/avantage | ✅ Supporté |

## 🎮 Environnements

### GridWorld Statique (rl_framework)
- **Taille** : 4x4 grille fixe
- **États** : Position de l'agent
- **Objectif** : Position fixe à atteindre
- **Actions** : Haut, Bas, Gauche, Droite

### GridWorld Dynamique (gridworld_dqn)  
- **Taille** : 5x5 grille personnalisable
- **États** : [agent_x, agent_y, goal_x, goal_y]
- **Objectif** : Position changeante à chaque épisode
- **Récompenses** : +10 (succès), -0.1 (par pas)

## 🚀 Utilisation Rapide

### Pour les algorithmes classiques :
```bash
cd rl_framework
python main.py
```

### Pour DQN :
```bash
cd gridworld_dqn
python scripts/train.py
python scripts/test.py
```

## 📊 Objectifs Pédagogiques

Ce portfolio couvre l'évolution des méthodes RL :
- **Programmation Dynamique** : Méthodes model-based (Value/Policy Iteration)
- **Monte Carlo** : Méthodes model-free sans bootstrap
- **Apprentissage TD** : Combinaison bootstrap/sampling (Q-Learning)
- **Deep RL** : Approximation de fonctions avec réseaux neuronaux (DQN)

## 📦 Dépendances

### RL Framework :
- Python 3.7+
- NumPy
- Matplotlib

### DQN Project :
- Python 3.7+
- PyTorch
- NumPy
- Matplotlib

---

*Les deux projets permettent une comparaison systématique des performances entre différentes approches de l'apprentissage par renforcement.*
