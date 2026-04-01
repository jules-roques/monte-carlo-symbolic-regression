# Monte Carlo Symbolic Regression

This repository implements Symbolic Regression using Monte Carlo Tree Search (MCTS) techniques, specifically UCT and its AlphaZero-inspired variant PUCT (Predictor + UCT). We test these algorithms on the [SRSD-easy benchmark](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy) .

## Installation with uv

We recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable package management. To install it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then, to install all the dependencies and the project in editable mode:
```bash
uv sync
```



## Environment Configuration

### Hugging Face Authentication

The datasets are hosted on Hugging Face. To avoid rate-limiting warnings, you can create a Token on [Hugging Face](https://huggingface.co/settings/tokens) and authenticate via the Hugging Face [CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login
```


## Full Workflow: From Training to Visualization

### 1. Train the Predictor (PUCT)

For PUCT to perform optimally, you need to train the `PredictorNN` that guides the search (Value and Prior Policy).

```bash
uv run scripts/training/train.py --epochs 30
```
*The script extracts MCTS trajectories and saves model weights in the `checkpoints/` directory.*

### 2. Run Benchmarks

Evaluate algorithms by discovering equations and then testing them.

**Discover Equations (Search phase):**
Run the algorithm on the training and validation sets:
```bash
uv run scripts/find.py --config configs/uct_default.json
```
*This command saves the best discovered expressions as pickles in `artifacts/pickles/uct/`.*

**Evaluate Results (Testing phase):**
Confront the discovered expressions with the test set to compute R² and Normalized Edit Distance (NED):
```bash
uv run scripts/test.py --model_name uct
```


### 3. Dashboard and Visualization

Generate visual and structured reports once you have multiple models' results:

**Generate Plots:**
Create heatmaps, barplots, and boxplots for all models evaluated in a specific difficulty:
```bash
uv run scripts/plot.py
```
*Outputs are saved in `results/figures/`.*

**Generate Markdown Table:**
Create a detailed table of discovered equations in a Markdown file:
```bash
uv run scripts/markdown.py
```
*Results are saved in `results/equations.md`.*



## Typical Use Cases

### Nested Monte Carlo Search (NMCS)
```bash
uv run scripts/search.py --config configs/nmcs_default.json && uv run scripts/test.py --model_name nmcs
```

### Upper Confidence Bound for Trees (UCT)
```bash
uv run scripts/search.py --config configs/uct_default.json && uv run scripts/test.py --model_name uct
```

### Predictor + UCT (PUCT)
**Training:**
```bash
uv run scripts/training/train.py --epochs 30
```
**Search and Test:**
```bash
uv run scripts/search.py --config configs/puct_nn.json && uv run scripts/test.py --model_name puct
```

### Deep Generative Symbolic Regression (DGSR)
**Training:**
```bash
uv run scripts/training/train_dgsr_mcts.py --epochs 30
```
**Search and Test:**
```bash
uv run scripts/search.py --config configs/DGSR.json && uv run scripts/test.py --model_name dgsr
```

## Configuration
Algorithm hyperparameters (like `max_atoms` or `num_iterations`) can be tuned by editing the JSON files in the `configs/` directory.