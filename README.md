# Monte Carlo Dropout for Uncertainty Quantification (JAX)

This repository implements **Monte Carlo (MC) Dropout** for uncertainty estimation in neural networks using **JAX**, following the ideas of *Gal & Ghahramani (2016)*.

The model is trained on the **Fashion-MNIST** dataset and demonstrates how keeping dropout active at test time allows us to estimate **predictive uncertainty** via multiple stochastic forward passes.

---

## Key Features

- Fully-connected neural network implemented from scratch in JAX  
- Dropout applied during both training and inference  
- Monte Carlo sampling to estimate:
  - predictive mean
  - predictive variance
  - predictive entropy
  - mutual information (epistemic uncertainty)
- Vectorised inference using `jax.vmap`
- No probabilistic programming libraries required  

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/joela03/bayesian-droput-jax
cd bayesian-droput
```

### 2. Create a virtual environment
#### Windows (PowerShell)

```bash
pip install virtualenv
virtualenv venv
venv\Scripts\activate
```

#### macOS / Linux
```bash
pip install virtualenv
virtualenv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the code
```bash
python main.py
```
This will:
- Load and preprocess the Fashion-MNIST dataset
- Train a neural network with dropout
- Evaluate standard predictions (dropout disabled)
- Perform MC Dropout inference (dropout enabled)
- Print uncertainty metrics for selected test samples

## JAX vs PyTorch / TensorFlow

This project is implemented in **JAX**, which differs from PyTorch and TensorFlow in several important ways.

The key features of JAX are:
- **Accelerated Linear Algebra (XLA)**: Performs aggressive linear algebra optimisations to efficiently execute numerical computations on CPUs, GPUs, and TPUs.
- **Just-in-time (JIT) compilation**: Compiles Python functions to XLA-optimised machine code for faster execution.
- **Automatic differentiation**: Computes exact gradients using function transformations such as `jax.grad`, enabling efficient and numerically stable backpropagation without manually coding gradients.

### Why JAX for Monte Carlo Dropout?

Monte Carlo Dropout requires performing many stochastic forward passes through the same network. JAX is particularly well-suited for this because:

- `vmap` enables efficient batching of multiple stochastic forward passes
- Explicit random number generation makes uncertainty estimates reproducible
- `jit` compilation allows Monte Carlo inference to run efficiently
- Functional programming encourages clear separation between model, randomness, and parameters

## Notes on Reproducibility

- Dropout randomness is controlled using explicit PRNG keys
- Results may vary slightly due to stochastic optimisation
- Increasing the number of MC samples improves uncertainty estimates

## References
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation
- Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting