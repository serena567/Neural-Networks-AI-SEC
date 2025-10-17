
"""
demo_custom_numbers.py
----------------------
A tweakable script to run the NumPy DNN with YOUR numbers.

Change the variables in the "CONFIG" section below:
- SEED: random seed (reproducibility)
- N_SAMPLES: dataset size
- N_FEATURES: number of input features
- N_CLASSES: number of classes
- HIDDEN_UNITS: width of the hidden layer
- EPOCHS: training epochs
- LR: learning rate
- N_SHOW: how many predictions to print

This uses CrossEntropyLoss on logits (no final Softmax layer for training).
"""

import numpy as np
from deep_nn_from_scratch import Dense, ReLU, LinearAct, NeuralNetwork, CrossEntropyLoss

# ====================
# CONFIG — CHANGE HERE
# ====================
SEED         = 7       # e.g., 7, 42, 123...
N_SAMPLES    = 300     # total samples
N_FEATURES   = 6       # input feature dimension
N_CLASSES    = 4       # number of classes (labels 0..N_CLASSES-1)
HIDDEN_UNITS = 32      # hidden layer width
EPOCHS       = 300     # training epochs
LR           = 0.03    # learning rate
N_SHOW       = 8       # how many rows to print for preview

# ===============
# DATA GENERATION
# ===============
np.random.seed(SEED)
X = np.random.randn(N_SAMPLES, N_FEATURES)
y = np.random.randint(0, N_CLASSES, size=N_SAMPLES)

# ===================
# MODEL CONSTRUCTION
# ===================
layers = [
    Dense(N_FEATURES, HIDDEN_UNITS, activation=ReLU()),       # He init auto
    Dense(HIDDEN_UNITS, N_CLASSES, activation=LinearAct()),   # logits
]
net = NeuralNetwork(layers, loss=CrossEntropyLoss())

# =======
# TRAINING
# =======
losses = net.fit(X, y, epochs=EPOCHS, lr=LR, verbose=True)

# =======
# OUTPUTS
# =======
probs = net.predict_proba(X[:N_SHOW])
preds = net.predict(X[:N_SHOW])

print("\n=== CONFIG ===")
print(f"SEED={SEED}, N_SAMPLES={N_SAMPLES}, N_FEATURES={N_FEATURES}, "
      f"N_CLASSES={N_CLASSES}, HIDDEN_UNITS={HIDDEN_UNITS}, "
      f"EPOCHS={EPOCHS}, LR={LR}, N_SHOW={N_SHOW}")

print("\nFirst rows of probabilities (sum to 1 across classes):")
print(np.round(probs, 3))
print("\nPredicted class indices for those rows:")
print(preds)
