import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
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

# Propose next x by maximizing acquisition over random candidates
def propose_acquisition(acq_func, gp, X_sample, Y_sample, bounds, xi=0.01, n_cand=5000):
    X_cand = np.random.uniform(bounds[0], bounds[1], size=(n_cand, 1))
    mu, sigma = gp.predict(X_cand, return_std=True)
    mu_sample_opt = np.max(Y_sample)
    vals = acq_func(mu, sigma, mu_sample_opt, xi)
    return X_cand[np.argmax(vals)]

# Simple Bayesian Neural Network via MC Dropout
class BNN(nn.Module):
    def __init__(self, layers=[1, 50, 50, 1], dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.fcs = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fcs.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
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

    # GP surrogate
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
    gp.fit(X, Y)

    # Optuna TPE study optimizing Probability of Improvement on GP
    def gp_objective(trial):
        x = trial.suggest_uniform('x', bounds[0], bounds[1])
        lam = schedule_lambda(trial.number + 1, budget)
        mu, sigma = gp.predict(np.array([[x]]), return_std=True)
        pi = probability_improvement(mu, sigma, np.max(Y), xi=0.01)[0]
        return pi

    study_pi = optuna.create_study(direction='maximize', sampler=TPESampler())
    study_pi.optimize(gp_objective, n_trials=budget)
    print(f"Best PI via TPE: x={study_pi.best_params['x']:.4f}, PI={study_pi.best_value:.4f}")

    # Bayesian Neural Net surrogate + EI/PI
    bnn = BNN()
    optimizer = torch.optim.Adam(bnn.parameters(), lr=1e-2)
    # Train BNN on initial data
    for epoch in range(2000):
        optimizer.zero_grad()
        preds = bnn(torch.tensor(X, dtype=torch.float32)).reshape(-1, 1)
        loss = F.mse_loss(preds, torch.tensor(Y, dtype=torch.float32))
        loss.backward()
        optimizer.step()

    # Evaluate acquisition over grid
    X_grid = np.linspace(bounds[0], bounds[1], 500).reshape(-1, 1)
    mu_bnn, sigma_bnn = bnn_predict(bnn, X_grid)
    mu_star = np.max(Y)
    ei_vals = expected_improvement(mu_bnn, sigma_bnn, mu_star)
    pi_vals = probability_improvement(mu_bnn, sigma_bnn, mu_star)
    x_ei = X_grid[np.argmax(ei_vals)][0]
    x_pi = X_grid[np.argmax(pi_vals)][0]

    print(f"BNN + EI: x={x_ei:.4f}, EI={np.max(ei_vals):.4f}")
    print(f"BNN + PI: x={x_pi:.4f}, PI={np.max(pi_vals):.4f}")
