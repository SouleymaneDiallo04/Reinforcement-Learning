# GridWorld Environment - Gymnasium

Un environnement GridWorld modulaire et hautement configurable base sur Gymnasium, concu pour des experiences d'apprentissage par renforcement flexibles et extensibles.

## Apercu

Ce module implemente un environnement GridWorld modulaire utilisant Gymnasium, specialement concu pour les experiences d'apprentissage par renforcement. Il permet des configurations flexibles de taille de grille, d'etats terminaux (objectifs), d'obstacles et de systemes de recompenses.

## Fonctionnalites Principales

### Grille Configurable
- **Grilles rectangulaires** : Dimensions personnalisables (grid_width × grid_height)
- **Grilles carrees** : Option simplifiee via grid_size
- **Espace d'observation** : MultiDiscrete pour les positions (x, y)

### Systeme d'Actions
- 0 : Deplacement vers le haut
- 1 : Deplacement vers le bas
- 2 : Deplacement vers la droite
- 3 : Deplacement vers la gauche

### Etats Terminaux (Objectifs)
- **Positions personnalisables** via terminal_states ou goal_positions
- **Generation aleatoire** avec num_goals
- **Configuration par defaut** : coins opposes de la grille

### Systeme d'Obstacles
- **Obstacles statiques** : Positions fixes
- **Obstacles mobiles** : Deplacement aleatoire a chaque pas (obstacles_move=True)
- **Generation automatique** via num_obstacles

### Systeme de Recompenses
- goal_reward : Recompense pour atteindre un objectif
- step_penalty : Penalite pour chaque deplacement
- wall_penalty : Penalite pour collision avec les bords
- obstacle_penalty : Penalite distincte pour les obstacles

### Visualisation Avancee
- **Rendu graphique** optionnel avec matplotlib
- **Expressions faciales** de l'agent selon l'etat
- **Couleurs distinctives** pour chaque type de cellule
- **Affichage en temps reel** de la position et du compteur de pas

## Initialisation

```python
GW(
    grid_width=4,
    grid_height=4,
    grid_size=None,
    terminal_states=None,  # Par defaut: [(0,0), (width-1,height-1)]
    initial_state=None,    # Par defaut: aleatoire
    max_steps=40,
    show_agent=True,
    show_goals=True,
    goal_reward=10,
    step_penalty=-1,
    wall_penalty=-2,
    obstacle_penalty=-5,
    num_goals=None,
    goal_positions=None,
    num_obstacles=0,
    obstacle_positions=None,
    obstacles_move=False
)
```

## Parametres

| Parametre | Type | Description |
|-----------|------|-------------|
| grid_width, grid_height | int | Dimensions de la grille rectangulaire |
| grid_size | int | Definit une grille carree (ecrase width/height) |
| terminal_states | list[tuple] | Liste fixe des positions terminales |
| initial_state | tuple | Position de depart de l'agent |
| max_steps | int | Nombre maximum de pas par episode |
| show_agent | bool | Afficher l'agent dans le rendu |
| show_goals | bool | Afficher les labels des objectifs |
| goal_reward | float | Recompense pour atteindre un objectif |
| step_penalty | float | Penalite pour chaque deplacement |
| wall_penalty | float | Penalite pour collision avec les bords |
| obstacle_penalty | float | Penalite pour tentative de deplacement vers un obstacle |
| num_goals | int | Nombre d'objectifs generes aleatoirement |
| goal_positions | list[tuple] | Positions personnalisees des objectifs |
| num_obstacles | int | Nombre d'obstacles generes aleatoirement |
| obstacle_positions | list[tuple] | Positions fixes des obstacles |
| obstacles_move | bool | Si True, les obstacles se deplacent aleatoirement |

## Methodes

### reset(seed=None, options=None)
Reinitialise l'environnement a un etat initial :
- Definit aleatoirement ou manuellement la position de depart de l'agent
- Genere les objectifs et obstacles selon la configuration
- Retourne l'observation initiale et un dictionnaire d'information vide

**Retourne :**
- state : position initiale [x, y]
- info : dictionnaire vide

### step(action)
Execute une etape dans l'environnement.

**Arguments :**
- action : entier dans {0, 1, 2, 3}

**Retourne :**
- next_state : position mise a jour [x, y]
- reward : valeur scalaire selon le resultat du deplacement
- terminated : True si l'agent a atteint un objectif
- truncated : True si le nombre maximum de pas est depasse
- info : dictionnaire vide

**Comportement :**
- Les deplacements sont contraints par les limites de la grille
- Les penalites sont appliquees pour les obstacles ou murs
- Les obstacles se deplacent aleatoirement si obstacles_move=True


## Exemple d'Utilisation

```python
import gymnasium as gym
from grid_env import GW

# Initialisation de l'environnement
env = GW(grid_width=5, grid_height=5, num_goals=1, num_obstacles=3)

# Reinitialisation
state, info = env.reset()

done = False
while not done:
    # Echantillonnage d'une action aleatoire
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Rendu de l'environnement
    env.render()
    done = terminated or truncated

# Fermeture de l'environnement
env.close()
```

## Cas d'Usage

### Algorithmes de Programmation Dynamique
Environnement ideal pour tester Value Iteration et Policy Iteration lorsque le modele est connu.

### Methodes Model-Free
Parfait pour Q-Learning et autres methodes sans modele lorsque les transitions sont inconnues.

### Extensions Deep RL
Adapte pour les extensions de RL profond avec etats de grande dimension ou caracteristiques complexes.

## Notes

- Cet environnement est entierement compatible avec Gymnasium
- Concu pour demontrer les concepts progressifs du Reinforcement Learning
- Permet une transition fluide entre les algorithmes classiques et modernes
- Ideal pour la comparaison systematique des performances des algorithmes

---

*Environnement developpe pour l'experimentation et l'education en Reinforcement Learning.*