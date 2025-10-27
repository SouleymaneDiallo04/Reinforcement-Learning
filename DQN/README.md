#### **Deep Q-Learning pour GridWorld avec Objectifs Changeants**



##### **Aperçu du Projet**



Ce projet implémente le Deep Q-Network (DQN) et ses variantes pour résoudre un environnement GridWorld où la position de l'objectif change à chaque épisode. L'agent doit apprendre une politique générale pour naviguer de n'importe quel point de départ à n'importe quelle position cible. Le projet compare deux stratégies d'entraînement : DQN avec et sans Experience Replay.



###### **Structure du Projet**



```

gridworld\_dqn/

├── agents/

│   ├── base\_agent.py              # Classe de base abstraite pour les agents

│   └── dqn\_agent.py               # Implémentation de l'agent DQN

├── environments/

│   └── gridworld.py               # Environnement GridWorld avec objectifs changeants

├── models/

│   └── networks.py                # Architectures de réseaux de neurones (DQN, Dueling DQN)

├── utils/

│   ├── config.py                  # Classes de configuration pour l'entraînement

│   └── visualization.py           # Outils de tracé et de visualisation

├── scripts/

│   ├── train.py                   # Script d'entraînement pour les deux agents

│   └── test.py                    # Script de test pour les agents entraînés

├── README.md

└── requirements.txt               # Dépendances Python

```



###### **Fonctionnalités Principales**



***Environnement à Objectifs Dynamiques***

\- **GridWorld** : Taille de grille personnalisable (5x5 par défaut)

\- **Objectifs Changeants** : La position cible change à chaque épisode

\- **Représentation d'État** : \[agent\_x, agent\_y, goal\_x, goal\_y]

\- **Récompenses** : Objectif (+10), Pénalité de pas (-0.1)



***Algorithme Deep Q-Learning***

\- Approximation des Q-values par réseau de neurones

\- Deux modes d'entraînement : avec et sans Experience Replay

\- Exploration epsilon-greedy avec décroissance

\- Entraînement par mini-lots depuis la mémoire de replay (si utilisée)



***Architecture du Réseau de Neurones***

\- \*\*Couche d'Entrée\*\* : 4 neurones (positions de l'agent et de l'objectif)

\- \*\*Couches Cachées\*\* : 2 couches fully connected avec activation ReLU (64 neurones chacune)

\- \*\*Couche de Sortie\*\* : 4 neurones (Q-values pour chaque action : haut, bas, gauche, droite)

\- \*\*Fonction de Coût\*\* : Mean Squared Error (MSE)

\- \*\*Optimiseur\*\* : Adam



###### **Dépendances**



\- Python 3.7+

\- PyTorch

\- NumPy

\- Matplotlib



###### **Utilisation**



\### Entraînement des Agents



Pour entraîner les deux agents DQN (avec et sans experience replay) :

```bash

python scripts/train.py

```



\### Test des Agents



Pour tester les agents pré-entraînés :

```bash

python scripts/test.py

```



***Paramètres Clés***



\- **Nombre d'épisodes** : 500 (pour chaque agent)

\- **Taux d'apprentissage (α)** : 0.001

\- **Facteur d'actualisation (γ)** : 0.95

\- **Exploration initiale (ε)** : 1.0

\- **Décroissance d'epsilon** : 0.995

\- **Epsilon minimum** : 0.01

\- **Taille des couches cachées** : 64 neurones

\- **Taille du batch** : 32 (pour l'experience replay)

\- **Taille de la mémoire de replay** : 100 expériences



###### **Composants de l'Algorithme**



***Représentation d'État***

L'état est représenté comme un vecteur de quatre entiers : \[agent\_x, agent\_y, goal\_x, goal\_y]. Chaque coordonnée est normalisée par la taille de la grille.



***Réseau de Neurones***

```python

class DQN(nn.Module):

&nbsp;   def \_\_init\_\_(self, input\_size, output\_size, hidden\_size=64):

&nbsp;       self.network = nn.Sequential(

&nbsp;           nn.Linear(input\_size, hidden\_size),

&nbsp;           nn.ReLU(),

&nbsp;           nn.Linear(hidden\_size, hidden\_size),

&nbsp;           nn.ReLU(),

&nbsp;           nn.Linear(hidden\_size, output\_size)

&nbsp;       )

```



***Boucle d'Entraînement***



1\. **Collecte d'Expérience** : Stocke (état, action, récompense, état\_suivant, terminé) dans la mémoire de replay (si utilisée)

2\. ***Échantillonnage par Mini-lots*** : Échantillonne aléatoirement des expériences de la mémoire (si utilisée)

3\. ***Prédiction des Q-values*** : Utilise le Q-network pour les Q-values courantes

4\. ***Calcul de la Cible*** : Utilise le même réseau pour les Q-values suivantes (avec gradient clipping)

5\. ***Optimisation de la Loss*** : Met à jour les poids du Q-network via backpropagation



Pour l'agent sans experience replay, l'apprentissage se fait immédiatement après chaque étape.



###### **Fichiers de Sortie**



***Visualisations Générées***



***1. Graphiques d'Entraînement :***

&nbsp;  - `results/training\_comparison.png` : Courbes d'entraînement pour les deux agents (scores bruts et moyennes mobiles)



***2. Points de Contrôle des Modèles :***

&nbsp;  - `models/dqn\_simple.pth` : DQN sans experience replay

&nbsp;  - `models/dqn\_replay.pth` : DQN avec experience replay



###### **Métriques Clés Suivies**



\- ***Récompenses par Épisode*** : Récompense cumulative par épisode

\- ***Taux d'Exploration*** : Valeur epsilon courante

\- ***Taux de Succès*** : Pourcentage d'épisodes atteignant l'objectif



###### **Mécanismes du Deep Q-Learning**



***Experience Replay*** (quand utilisée)

\- Brise les corrélations temporelles entre les expériences

\- Permet la réutilisation d'expériences passées

\- Améliore l'efficacité des échantillons

\- Fournit un entraînement plus stable



***Stratégie Epsilon-Greedy***

\- Équilibre exploration et exploitation

\- Commence avec une exploration complète (ε=1.0)

\- Décroît jusqu'à une exploration minimale (ε=0.01)

\- Assure une couverture adéquate de l'espace d'états



###### **Comparaison des Stratégies d'Entraînement**



***Sans Experience Replay***

\- ***Avantages*** : Plus simple, apprentissage initial plus rapide, moins d'utilisation mémoire

\- ***Désavantages*** : Moins stable, peut oublier les expériences précédentes



***Avec Experience Replay***

\- ***Avantages*** : Plus stable, meilleure efficacité des échantillons, robuste au bruit

\- ***Désavantages*** : Apprentissage initial plus lent, plus d'utilisation mémoire



###### **Analyse des Performances**



***Courbe d'Apprentissage Attendue***

\- ***Phase Initiale*** : Exploration aléatoire, récompenses négatives (due à la pénalité de pas)

\- ***Phase d'Apprentissage*** : Amélioration graduelle des récompenses et du taux de succès

\- ***Phase de Convergence*** : Politique stable avec un taux de succès élevé



***Nos Résultats***

Dans nos expériences, les deux agents ont atteint des performances finales similaires (scores autour de 9.7-10.0) après 500 épisodes. Cependant, l'agent avec experience replay a montré un apprentissage plus stable et une meilleure récupération après des baisses de performance.



###### **Personnalisation**



***Modification de l'Architecture du Réseau***

Modifiez `models/networks.py` pour changer l'architecture du réseau de neurones.



***Ajustement des Paramètres d'Entraînement***

Modifiez `utils/config.py` pour ajuster les hyperparamètres.



***Modifications de l'Environnement***

Changez la taille de la grille, les récompenses ou la dynamique dans `environments/gridworld.py`.





###### **Conclusion**



Ce projet démontre l'efficacité du Deep Q-Learning dans des environnements dynamiques avec des objectifs changeants. Les deux stratégies d'entraînement (avec et sans experience replay) peuvent atteindre des performances élevées, mais l'experience replay fournit un apprentissage plus stable. La structure modulaire du code permet une expérimentation et une extension faciles.







