## Structure du projet : 

Ce projet fournit un cadre pour l'implémentation et la comparaison de différents algorithmes d'apprentissage par renforcement dans un environnement de grille. Il inclut plusieurs agents RL, allant des méthodes de programmation dynamique (Value Iteration, Policy Iteration) aux méthodes de Monte Carlo et Q-Learning.

```

rl_framework/
│
├── 📄 grid_env.py                          # Environnement de grille 4x4
├── 📄 main.py                              # Programme principal
├── 📄 utils.py                             # Fonctions utilitaires
│
└── 📂 agents/                  # Dossier des agents RL
    ├── 📄 __init__.py
    ├── 📄 random_agent.py              # Agent aléatoire (baseline)
    ├── 📄 value_iteration_agent.py     # Value Iteration
    ├── 📄 policy_iteration_agent.py    # Policy Iteration
    ├── 📄 monte_carlo_agent.py         # Monte Carlo
    └── 📄 q_learning_agent.py          # Q-Learning

```
    Méthode 1 : Menu interactif (Recommandée)

python main.py

=== FRAMEWORK RL AVEC MULTIPLES AGENTS ===
1. Comparer tous les agents
2. Tester un agent spécifique  
3. Visualiser un agent optimal

```
   Méthode 2 : Ligne de commande directe
```
### Comparaison complète (500 épisodes)
python main.py

### Test spécifique d'un agent
python -c " 
from main import compare_agents 
compare_agents(episodes=200)
"
