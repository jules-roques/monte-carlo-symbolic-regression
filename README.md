# Monte Carlo Symbolic Regression

This repository implements Symbolic Regression using Monte Carlo Tree Search techniques, specifically UCT and its AlphaZero-inspired variant PUCT (Predictor + UCT).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .  # Install the project in editable mode
```

## Configuration de l'environnement (Hugging Face)

Le dataset `SRSD-easy` est hébergé sur Hugging Face. Pour éviter un message d'avertissement de limite de requêtes (*Rate Limit*), vous pouvez vous authentifier via un token.

Copiez le fichier de modèle `.env.example` et renommez-le en `.env` :

```bash
cp .env.example .env
```

Ouvrez le fichier `.env` nouvellement créé et insérez votre token Hugging Face associé à la variable `HF_TOKEN`. Ce fichier précise aussi que le dossier de cache de Hugging Face de vos datasets sera stocké dans un répertoire `./data` pour éviter de tout retélécharger à chaque fois, ce dernier étant bien sûr ignoré par `.gitignore`.

## Workflow Complet : De l'Entraînement à la Visualisation

### 1. Entraîner le Predictor (PUCT)
Pour utiliser PUCT de manière optimale, il faut entraîner le réseau de neurones (`PredictorNN`) qui servira de guide (Value et Prior Policy) pendant la recherche.

```bash
python src/train.py --epochs 30
```
*Le script extraira des trajectoires MCTS et générera des poids enregistrés dans le répertoire `checkpoints/`.*

### 2. Lancer les Benchmarks
Vous pouvez évaluer les différents algorithmes via les scripts de validation et de tests. 
Par défaut, pour que l'évaluation reste rapide sur un seul CPU, le script ne teste que les **5 premières équations**. Vous pouvez modifier cette limite avec l'argument `--num-equations` (ou mettre `--num-equations 0` pour évaluer le dataset complet).

**Tester UCT (Baseline) :**
```bash
python scripts/run_validation.py --config configs/uct_default.json --num-equations 5
```

**Tester PUCT (Deep Learning) :**
```bash
# Utilisera automatiquement le dernier checkpoint grâce à puct_nn.json
python scripts/run_validation.py --config configs/puct_nn.json --num-equations 5
```

*Note: Vous pouvez aussi cibler des équations spécifiques via `--equations feynman-i.12.1,feynman-i.12.4`.*
À chaque exécution, les résultats détaillés sont automatiquement enregistrés sous format JSON dans le dossier **`logs/`**.

### 3. Dashboard et Visualisation
Une fois vos runs terminés pour UCT et PUCT, vous pouvez générer une comparaison sous forme de graphique Box-plot :

```bash
python scripts/visualize_benchmark.py
```
L'image `benchmark_r2.png` sera sauvegardée dans le dossier **`results/`**, illustrant les performances $R^2$ de chaque algorithme évalué !

## Lancer les Tests Unitaires

```bash
pytest tests/
```

## Cas typique
### NMCS
```bash
source .venv/bin/activate && python scripts/run_validation.py --config configs/nmcs_default.json --max-true-atoms 6 && python scripts/visualize_benchmark.py
```
### UCT
```bash
source .venv/bin/activate && python scripts/run_validation.py --config configs/uct_default.json --max-true-atoms 6 && python scripts/visualize_benchmark.py
```
### PUCT
Training de base
```bash
source .venv/bin/activate && python src/train.py --epochs 30
```
Training supplémentaire pour affiner la distribution de proba sur les atomes simples
```bash
source .venv/bin/activate && python src/train_2.py --epochs 30
```

```bash
source .venv/bin/activate && python scripts/run_validation.py --config configs/puct_nn.json --max-true-atoms 6 && python scripts/visualize_benchmark.py
```

### DGSR_MCTS
Training
```bash
source .venv/bin/activate && python src/train_dgsr.py --epochs 30
```

```bash
source .venv/bin/activate && python scripts/run_validation.py --config configs/dgsr_mcts.json --max-true-atoms 6 && python scripts/visualize_benchmark.py
```

### + edit les fichiers de conf pour les hyperparametres des algos