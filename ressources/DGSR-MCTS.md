# 🧠 Méthode – Deep Generative Symbolic Regression with MCTS

## 🎯 Objectif

Combiner un **modèle génératif neuronal** avec une **recherche arborescente (MCTS)** pour explorer efficacement l’espace des expressions mathématiques.

---

## 🧩 Représentation des expressions

* Les équations sont représentées comme des **arbres syntaxiques** (operators + variables + constantes)
* Chaque nœud correspond à une opération (ex : +, sin, ×)

On cherche une fonction :
[
f_\theta(x) \approx y
]

---

## 🤖 1. Modèle génératif neuronal

### Pré-entraînement

* Un modèle (type Transformer) est entraîné à générer des expressions valides
* Il apprend une **distribution sur les équations plausibles** :

[
p_\theta(e)
]

### Rôle

* Proposer des **mutations intelligentes** d’expressions existantes
* Générer des candidats conditionnés :

[
p_\theta(e' \mid e)
]

---

## 🌳 2. Monte-Carlo Tree Search (MCTS)

### Principe

* Exploration d’un arbre où :

  * chaque nœud = une expression
  * chaque branche = une mutation

### Sélection (UCB)

[
a^* = \arg\max_a \left( Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}} \right)
]

* (Q(s,a)) : valeur moyenne
* (N(s)) : nombre de visites du nœud
* (N(s,a)) : nombre de visites de l’action
* (c) : coefficient d’exploration

### Étapes

1. **Sélection** (UCB)
2. **Expansion** : génération via (p_\theta(e'|e))
3. **Simulation / évaluation**
4. **Backpropagation**

---

## 🔁 3. Mutations neuronales contextuelles

* Le modèle génère des mutations **conditionnées sur l’expression actuelle**

[
e' \sim p_\theta(\cdot \mid e)
]

* Modifications locales : remplacement de sous-arbres

---

## 📈 4. Apprentissage en ligne

* Mise à jour du modèle pour maximiser les bonnes mutations

Objectif implicite :

[
\max_\theta \mathbb{E}*{e' \sim p*\theta(\cdot|e)} [R(e')]
]

* (R(e')) : récompense basée sur la qualité de l’équation

---

## ⚖️ Fonction objectif

### Erreur de prédiction (MSE)

[
\mathcal{L}*{\text{fit}} = \frac{1}{n} \sum*{i=1}^n (f(x_i) - y_i)^2
]

### Pénalisation de complexité

[
\mathcal{L}_{\text{complex}} = \lambda \cdot \text{size}(e)
]

### Objectif total

[
\mathcal{L} = \mathcal{L}*{\text{fit}} + \mathcal{L}*{\text{complex}}
]

👉 Objectif : compromis **précision / simplicité**

---

## 🔄 Pipeline global

1. Initialiser avec des expressions simples
2. Boucle MCTS :

   * sélection via UCB
   * génération : (p_\theta(e'|e))
   * évaluation via (\mathcal{L})
   * mise à jour des valeurs (Q)
3. Mise à jour du modèle
4. Répéter jusqu’à convergence

---

## 💡 Idée clé

> Le modèle neuronal apprend une distribution sur les bonnes transformations, tandis que MCTS optimise explicitement une fonction de coût.

---

## 🧠 Résumé en une ligne

👉 Un **modèle probabiliste guide l’exploration**, et une **recherche UCB optimise la découverte d’équations**.
