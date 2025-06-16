import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import optuna

# Example 1D black-box function
def black_box(x: float) -> float:
    return np.sin(3 * x) + 0.1 * np.random.randn()

# Schedule for lambda per iteration (Eq. 30)
def schedule_lambda(t: int, max_iter: int) -> float:
    frac = min(t / max_iter * np.pi, np.pi)
    return (1 - np.cos(frac)) / 2

# Construction of the acquisition objective J(x, lambda) = mu(x) - lambda * sigma^2(x)
def J(x: float, gp: GaussianProcessRegressor, lam: float) -> float:
    mu, sigma = gp.predict(np.array([[x]]), return_std=True)
    return float(mu[0] - lam * (sigma[0] ** 2))

if __name__ == '__main__':
    # Search space bounds
    bounds = (0.0, 2.0)
    # Initial random samples to fit the GP surrogate
    n_init = 5
    budget = 30

    X_init = np.random.uniform(bounds[0], bounds[1], size=(n_init, 1))
    Y_init = np.array([black_box(x[0]) for x in X_init]).reshape(-1, 1)

    # Fit Gaussian Process surrogate
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)
    gp.fit(X_init, Y_init)

    # Optuna objective: maximize J(x, lambda_t)
    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_uniform('x', bounds[0], bounds[1])
        lam = schedule_lambda(trial.number + 1, budget)
        return J(x, gp, lam)

    # Create and run the Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=budget)

    x_opt = study.best_params['x']
    j_opt = study.best_value
    f_opt = black_box(x_opt)

    print(f"Best acquisition J: {j_opt:.4f} at x = {x_opt:.4f}")
    print(f"True function value f(x): {f_opt:.4f}")
