"""Fontes de dados: DGP simulado e dataset real IHDP (embutido no econml)."""

import numpy as np


def simulate_data(n_samples: int = 800, seed: int = 42):
    """DGP simulado com efeito de tratamento contínuo, monotônico negativo.

    Retorna (Y, T, X, true_te_fn) onde true_te_fn(X) devolve o efeito real.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, 1))
    T = 5 + X[:, 0] + rng.normal(0, 1, size=n_samples)
    efeito_real = -2.0 - 0.5 * X[:, 0]
    Y = 100 + efeito_real * T + X[:, 0] * 5 + rng.normal(0, 2, size=n_samples)
    true_te = lambda Xt: -2.0 - 0.5 * np.atleast_1d(np.asarray(Xt)).ravel()  # noqa: E731
    return Y.ravel(), T.ravel(), X, true_te


def load_ihdp(surface: str = "B", seed: int = 42):
    """Dataset real IHDP (Hill, 2011) com efeito verdadeiro conhecido.

    Covariates reais (n~985, 47 features), tratamento binário, targets
    semi-sintéticos.  Surface 'A': efeito constante (=4).  Surface 'B':
    efeito heterogêneo.  Arquivo CSV embutido no econml — sem download.
    """
    from econml.data.dgps import ihdp_surface_A, ihdp_surface_B

    fn = ihdp_surface_A if surface.upper() == "A" else ihdp_surface_B
    Y, T, X, true_te = fn(random_state=seed)
    return Y.ravel(), T.ravel(), X, true_te.ravel()


def load_dataset(name: str, n_samples: int = 800, seed: int = 42):
    """Dispatch funcional por nome: 'sim' | 'ihdp-a' | 'ihdp-b'."""
    if name == "sim":
        Y, T, X, true_te = simulate_data(n_samples, seed)
        return {"name": name, "Y": Y, "T": T, "X": X, "true_te_fn": true_te}
    if name in ("ihdp-a", "ihdp-b"):
        Y, T, X, true_te = load_ihdp(surface=name[-1], seed=seed)
        return {"name": name, "Y": Y, "T": T, "X": X, "true_te": true_te}
    raise ValueError(f"dataset desconhecido: {name}")
