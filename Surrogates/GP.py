import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from scipy.stats import norm

def black_box_function(x):
    """
    Example black-box function: 1D noisy sinusoidal function;
    Due to commercial and safety issues, the company prohibits the use of black-box functions.
    """
    return np.sin(3 * x) + 0.1 * np.random.randn()

class RiskAwareBO:
    def __init__(self, bounds, risk_tol=0.1, epsilon=0.2, budget=50,
                 kernel=None, acquisition='adaptive_lagrangian'):
        self.bounds = np.array(bounds)
        self.dim = self.bounds.shape[0]
        self.risk_tol = risk_tol    # c in paper
        self.epsilon = epsilon      # clipping for importance weight
        self.budget = budget        # η
        self.acquisition = acquisition
        self.X = []
        self.y = []
        self.gp = GaussianProcessRegressor(
            kernel=kernel or 1.0 * RBF(length_scale=1.0),
            alpha=1e-6, normalize_y=True)

    def _schedule_lambda(self, t):
        # λ_t = (1 - cos(min(t/500 * π, π))) / 2
        frac = min(t / 500.0 * np.pi, np.pi)
        return (1 - np.cos(frac)) / 2

    def _acquisition(self, x, lam):
        x = np.atleast_2d(x)
        mu, sigma = self.gp.predict(x, return_std=True)
        var = sigma**2
        # For demonstration, assume uniform target g(x) ~ q(x), weight=1
        # Hence objective J = mu - lam * var
        return mu - lam * var

    def propose_location(self, lam, n_candidates=10000):
        # Randomly sample candidates and pick best by acquisition
        X_cand = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(n_candidates, self.dim))
        acq_vals = self._acquisition(X_cand, lam)
        idx = np.argmax(acq_vals)
        return X_cand[idx]

    def optimize(self):
        # Initialization: 5 random points
        for _ in range(5):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            y0 = black_box_function(x0)
            self.X.append(x0)
            self.y.append(y0)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        for t in range(1, self.budget + 1):
            # Fit surrogate
            self.gp.fit(self.X, self.y)
            lam = self._schedule_lambda(t)
            # Propose next point
            x_next = self.propose_location(lam)
            y_next = black_box_function(x_next)
            # Append
            self.X = np.vstack((self.X, x_next))
            self.y = np.append(self.y, y_next)
            print(f"Step {t}: x = {x_next:.4f}, y = {y_next:.4f}, lambda = {lam:.4f}")

        # Return best observed
        best_idx = np.argmax(self.y)
        return self.X[best_idx], self.y[best_idx]

if __name__ == '__main__':
    # Define search space for x in [0, 2]
    bounds = [(0.0, 2.0)]
    bo = RiskAwareBO(bounds, risk_tol=0.1, epsilon=0.2, budget=30)
    x_best, y_best = bo.optimize()
    print(f"Best x: {x_best:.4f}, Best y: {y_best:.4f}")

