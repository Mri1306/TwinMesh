import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigvals


class HawkesMLE:
    """
    Multivariate Hawkes process with exponential kernel.
    MLE estimation via L-BFGS-B. Pure numpy — no tick required.
    """

    def __init__(
        self,
        n_nodes: int,
        decay: float = 0.3,
        max_iter: int = 500,
        tol: float = 1e-6,
        verbose: bool = False,
        stability_margin: float = 0.95,
    ):
        self.n_nodes         = n_nodes
        self.decay           = float(decay)
        self.max_iter        = max_iter
        self.tol             = tol
        self.verbose         = verbose
        self.stability_margin = stability_margin

        self.baseline  = None   
        self.adjacency = None   
        self._fitted   = False
        self._T        = None   

    def _compute_R(self, events: list) -> list:
        """
        Pre-compute R_ij(k) = sum_{t_l < t_k^i} exp(-beta*(t_k^i - t_l^j))
        for all (i, k, j) combinations. This is the recursive term.
        R_ij(k) = exp(-beta*(t_k - t_{k-1})) * (R_ij(k-1) + 1_{j fires at t_{k-1}})

        Returns list of (n_nodes x n_nodes x n_events_i) arrays.
        """
        n      = self.n_nodes
        beta   = self.decay
        R_list = []

        for i in range(n):
            t_i = np.array(events[i])
            n_i = len(t_i)
            R   = np.zeros((n, n_i))

            for k in range(1, n_i):
                dt = t_i[k] - t_i[k-1]
                decay_factor = np.exp(-beta * dt)

                for j in range(n):
                    t_j   = np.array(events[j])
                    mask  = t_j <= t_i[k-1]
                    if not mask.any():
                        R[j, k] = 0.0
                    else:
                        at_prev = np.sum(np.abs(t_j - t_i[k-1]) < 1e-10)
                        R[j, k] = decay_factor * (R[j, k-1] + at_prev)

            R_list.append(R)
        return R_list

    def _log_likelihood(self, params: np.ndarray, events: list, T: float) -> float:
        """
        Negative log-likelihood of multivariate Hawkes.
        params = [mu_0..mu_{n-1}, alpha_00, alpha_01, ..., alpha_{n-1,n-1}]
        """
        n    = self.n_nodes
        beta = self.decay

        mu    = params[:n]
        alpha = params[n:].reshape(n, n)

        if np.any(mu <= 0) or np.any(alpha < 0):
            return 1e10

        ll = 0.0

        for i in range(n):
            t_i = np.array(events[i])
            n_i = len(t_i)

            if n_i == 0:

                ll -= mu[i] * T
                continue

            log_intensity_sum = 0.0
            for k, tk in enumerate(t_i):
                lam_ik = mu[i]
                for j in range(n):
                    t_j       = np.array(events[j])
                    past_j    = t_j[t_j < tk]
                    if len(past_j):
                        lam_ik += alpha[i, j] * np.sum(
                            beta * np.exp(-beta * (tk - past_j))
                        )
                lam_ik = max(lam_ik, 1e-300)
                log_intensity_sum += np.log(lam_ik)

            compensator = mu[i] * T
            for j in range(n):
                t_j = np.array(events[j])
                if len(t_j):
                    compensator += (alpha[i, j] / beta) * np.sum(
                        1.0 - np.exp(-beta * (T - t_j))
                    )

            ll += log_intensity_sum - compensator

        return -ll   

    def _log_likelihood_fast(self, params: np.ndarray, events: list, T: float) -> float:
        """
        Vectorised log-likelihood using numpy broadcasting.
        Faster than the loop version for large event sets.
        """
        n    = self.n_nodes
        beta = self.decay

        mu    = np.maximum(params[:n], 1e-9)
        alpha = np.maximum(params[n:].reshape(n, n), 0.0)

        ll = 0.0

        for i in range(n):
            t_i = np.array(events[i], dtype=float)
            n_i = len(t_i)

            if n_i == 0:
                ll -= mu[i] * T
                continue

            lam = np.full(n_i, mu[i])
            for j in range(n):
                t_j = np.array(events[j], dtype=float)
                if len(t_j) == 0:
                    continue

                dt_matrix = t_i[:, None] - t_j[None, :]   
                valid     = dt_matrix > 0                   
                contrib   = np.where(valid, beta * np.exp(-beta * dt_matrix), 0.0)
                lam      += alpha[i, j] * contrib.sum(axis=1)

            lam = np.maximum(lam, 1e-300)
            ll += np.sum(np.log(lam))

            comp = mu[i] * T
            for j in range(n):
                t_j = np.array(events[j], dtype=float)
                if len(t_j):
                    comp += (alpha[i, j] / beta) * np.sum(
                        1.0 - np.exp(-beta * (T - t_j))
                    )
            ll -= comp

        return -ll

    def fit(self, events: list) -> "HawkesMLE":
        """
        Fit Hawkes parameters via MLE (L-BFGS-B).

        events: list of n_nodes lists, each containing sorted event timestamps.

        Strategy: multi-start with 3 decay values, keep best log-likelihood.
        This prevents convergence to alpha=0 (Poisson) which happens when
        the single initial point is too close to the zero boundary.

        Bounds: mu > 0, alpha in [0, 2/n] to control spectral radius.
        Stability: spectral radius of alpha clipped to stability_margin.
        """
        n    = self.n_nodes
        beta = self.decay

        all_times = [t for ev in events for t in ev]
        T = max(all_times) + 1e-3 if all_times else 1.0
        self._T = T

        best_ll    = float("-inf")
        best_mu    = None
        best_alpha = None

        mu_empirical = np.array([max(len(ev) / T, 1e-4) for ev in events])

        alpha_inits = [
            np.full((n, n), 0.01),         
            np.full((n, n), 0.3 / max(n, 1)), 
            np.eye(n) * 0.2,                  
            np.full((n, n), 0.1),             
        ]

        bounds = [(1e-8, None)] * n + [(0.0, 1.0)] * (n * n)

        for alpha_init in alpha_inits:
            x0 = np.concatenate([mu_empirical, alpha_init.flatten()])
            try:
                result = minimize(
                    fun     = self._log_likelihood_fast,
                    x0      = x0,
                    args    = (events, T),
                    method  = "L-BFGS-B",
                    bounds  = bounds,
                    options = {
                        "maxiter": self.max_iter,
                        "ftol":    self.tol,
                        "gtol":    1e-6,
                        "disp":    False,
                    },
                )
                mu_cand    = result.x[:n]
                alpha_cand = result.x[n:].reshape(n, n)
                ll_cand    = -result.fun

                if ll_cand > best_ll:
                    best_ll    = ll_cand
                    best_mu    = mu_cand.copy()
                    best_alpha = alpha_cand.copy()

            except Exception:
                continue

        if best_mu is None:
            best_mu    = mu_empirical
            best_alpha = np.zeros((n, n))

        mu    = best_mu
        alpha = best_alpha

        try:
            eigenvalues = np.abs(eigvals(alpha))
            rho         = float(eigenvalues.max().real)
            if rho >= 1.0:
                alpha = alpha * (self.stability_margin / rho)
        except Exception:
            pass

        self.baseline  = mu
        self.adjacency = alpha
        self._fitted   = True
        self._best_ll  = best_ll

        return self

    def intensity(self, t: float, events: list) -> np.ndarray:
        """
        Computes lambda_i(t) for all nodes at time t.
        Returns array of shape (n_nodes,).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        n    = self.n_nodes
        beta = self.decay
        lam  = self.baseline.copy()

        for i in range(n):
            for j in range(n):
                t_j      = np.array(events[j], dtype=float)
                past_j   = t_j[t_j < t]
                if len(past_j):
                    lam[i] += self.adjacency[i, j] * np.sum(
                        beta * np.exp(-beta * (t - past_j))
                    )
        return lam

    def intensity_series(
        self, timestamps: np.ndarray, events: list
    ) -> np.ndarray:
        """
        Computes lambda_i(t) for all nodes at each timestamp.
        Returns array of shape (len(timestamps), n_nodes).
        """
        return np.array([self.intensity(t, events) for t in timestamps])

    def log_likelihood_value(self, events: list) -> float:
        """Returns the fitted log-likelihood (positive = better)."""
        if not self._fitted:
            return float("-inf")
        # Use stored best LL from multi-start fit if available
        if hasattr(self, "_best_ll") and self._best_ll != float("-inf"):
            return self._best_ll
        params = np.concatenate([self.baseline, self.adjacency.flatten()])
        return -self._log_likelihood_fast(params, events, self._T)

    def spectral_radius(self) -> float:
        """Returns spectral radius of adjacency matrix. Must be < 1 for stability."""
        if not self._fitted:
            return float("nan")
        return float(np.abs(eigvals(self.adjacency)).max().real)

    def summary(self, port_names: list = None) -> str:
        """Human-readable summary of fitted parameters."""
        if not self._fitted:
            return "Not fitted."
        n     = self.n_nodes
        names = port_names or [f"Node_{i}" for i in range(n)]
        lines = [
            f"HawkesMLE (n={n}, decay={self.decay})",
            f"  Spectral radius: {self.spectral_radius():.4f} (stable: {self.spectral_radius() < 1})",
            f"  Baseline mu:",
        ]
        for i, name in enumerate(names):
            lines.append(f"    {name:<20} mu={self.baseline[i]:.6f}")
        lines.append(f"  Adjacency alpha (top entries):")

        alpha_flat = [
            (self.adjacency[i, j], names[i], names[j])
            for i in range(n) for j in range(n) if i != j
        ]
        alpha_flat.sort(reverse=True)
        for val, src, tgt in alpha_flat[:5]:
            lines.append(f"    {src:<20} -> {tgt:<20} alpha={val:.4f}")

        return "\n".join(lines)