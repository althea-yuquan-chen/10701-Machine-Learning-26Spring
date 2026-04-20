# 10701-Machine-Learning — Spring 2026

Homework assignments for **CMU 10-701: Introduction to Machine Learning**, Spring 2026.

---

## Repository Structure

```
HW1-kNN-DecisionTree/     # k-Nearest Neighbors & Decision Trees
HW2-LinearRegression/     # Linear Regression & Time Series
HW3-NN/                   # Neural Networks from scratch (PyTorch)
HW4/                      # Recurrent Neural Networks (Yelp sentiment)
```

---

## Assignments

### HW1 — k-Nearest Neighbors & Decision Trees
Implements kNN classification and a decision tree with pruning from scratch.

- **Key files:**
  - `decision_tree.py` — Decision tree implementation
  - `decision_tree.ipynb` — Experiments and analysis notebook
  - `depth_metrics.png` — Train/val accuracy vs. tree depth plot
  - `trained_tree.txt` / `pruned_tree.txt` — Serialized tree outputs
- **Data:** `data/` — Heart disease and education datasets (train/val splits in TSV format), plus a small toy dataset

---

### HW2 — Linear Regression & Time Series
Implements linear regression and applies it to time series forecasting.

- **Key files:**
  - `time_series.py` — Linear regression implementation
  - `practice.ipynb` — Experiments notebook
- **Data:** `temperature.csv` — Temperature time series data

---

### HW3 — Neural Networks (from scratch)
Implements core neural network components manually using PyTorch primitives, trained on FashionMNIST.

- **Key files:**
  - `nn_implementation_code/custom_functions.py` — Activation functions, loss functions
  - `nn_implementation_code/custom_modules.py` — Custom layer/module implementations
  - `nn_implementation_code/base_experiment.py` — Training and evaluation loop
  - `nn_implementation_code/weights.pt` — Saved model weights
  - `check/` — Intermediate gradient/activation check files (`a.txt`, `b.txt`, `z.txt`, `updated_params.pt`)
- **Data:** FashionMNIST — auto-downloaded by PyTorch on first run (not committed to the repo)

---

### HW4 — Recurrent Neural Networks
Implements an RNN for sentiment analysis on the Yelp reviews dataset.

- **Key files:**
  - `Programming/rnn.py` — RNN implementation
  - `Programming/reference_data.py` / `reference_data_q4.py` — Data loading utilities
  - `Programming/environment.yml` — Conda environment spec
  - `Programming/requirements.txt` — Python dependencies
- **Data:** Yelp reviews dataset — preprocessed splits (`yelp_train_sampled.pkl`, `yelp_test_sampled.pkl`) are not committed to the repo; obtain them separately.

---

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd 10701-Machine-Learning-26Spring

# Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

For **HW4**, a Conda environment is provided:

```bash
cd HW4/Programming
conda env create -f environment.yml
conda activate <env-name>
# or with pip:
pip install -r requirements.txt
```

---

## Course

[10-701 Introduction to Machine Learning](https://www.cs.cmu.edu/~10701/) — School of Computer Science, Carnegie Mellon University.

---

## Author

Yuquan (Althea) Chen
