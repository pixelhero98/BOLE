import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.samplers import TPESampler
from scipy.stats import norm

# Example 1D black-box function
def black_box(x: float) -> float:
    return np.sin(3 * x) + 0.1 * np.random.randn()

# Schedule for lambda per iteration (Eq. 30)
def schedule_lambda(t: int, max_iter: int) -> float:
    frac = min(t / max_iter * np.pi, np.pi)
    return (1 - np.cos(frac)) / 2

# Probability of Improvement acquisition
def probability_improvement(mu: np.ndarray, sigma: np.ndarray, mu_sample_opt: float, xi: float = 0.01) -> np.ndarray:
    with np.errstate(divide='ignore'):
        Z = (mu - mu_sample_opt - xi) / sigma
        pi = norm.cdf(Z)
        pi[sigma == 0.0] = 0.0
    return pi

# Expected Improvement acquisition
def expected_improvement(mu: np.ndarray, sigma: np.ndarray, mu_sample_opt: float, xi: float = 0.01) -> np.ndarray:
    with np.errstate(divide='ignore'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

# Simple Bayesian Neural Network via MC Dropout
class BNN(nn.Module):
    def __init__(self, layers=[1, 50, 50, 1], dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.fcs = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs)-1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=True)
        return x

# Predict mu and sigma from BNN via MC samples
def bnn_predict(model, X, mc_iters=50):
    model.train()
    preds = []
    x_tensor = torch.tensor(X, dtype=torch.float32)
    for _ in range(mc_iters):
        preds.append(model(x_tensor).detach().numpy())
    preds = np.stack(preds, axis=0)
    mu = preds.mean(axis=0).reshape(-1)
    sigma = preds.std(axis=0).reshape(-1)
    return mu, sigma

if __name__ == '__main__':
    bounds = (0.0, 2.0)
    n_init, budget = 5, 30

    # Generate initial samples
    X = np.random.uniform(bounds[0], bounds[1], size=(n_init, 1))
    Y = np.array([black_box(x[0]) for x in X]).reshape(-1, 1)

    # Initialize and train BNN surrogate
    bnn = BNN()
    optimizer = torch.optim.Adam(bnn.parameters(), lr=1e-2)
    for epoch in range(2000):
        optimizer.zero_grad()
        preds = bnn(torch.tensor(X, dtype=torch.float32)).reshape(-1, 1)
        loss = F.mse_loss(preds, torch.tensor(Y, dtype=torch.float32))
        loss.backward()
        optimizer.step()

    # Evaluate acquisition (EI and PI) over a grid
    X_grid = np.linspace(bounds[0], bounds[1], 500).reshape(-1, 1)
    mu_bnn, sigma_bnn = bnn_predict(bnn, X_grid)
    mu_star = np.max(Y)
    ei_vals = expected_improvement(mu_bnn, sigma_bnn, mu_star)
    pi_vals = probability_improvement(mu_bnn, sigma_bnn, mu_star)

    x_ei = X_grid[np.argmax(ei_vals)][0]
    x_pi = X_grid[np.argmax(pi_vals)][0]

    print(f"BNN + EI: x={x_ei:.4f}, EI={np.max(ei_vals):.4f}")
    print(f"BNN + PI: x={x_pi:.4f}, PI={np.max(pi_vals):.4f}")
