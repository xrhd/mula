"""Extração do score de confiança (incerteza do GP final) de um DML treinado."""

import warnings

import numpy as np


class GPScoreExtractor:
    """Isola acesso ao GP final do econml e o cálculo do CATE com incerteza.

    O econml treina `dml.model_final_` sobre cross_product([intercepto, X], T_res),
    ou seja, colunas [T_res, X*T_res].  Para obter o efeito de (T=1 vs T=0),
    avaliamos o GP em [1, x] e em [0, 0] — a diferença reproduz o mesmo
    `dml.effect()` do econml, mas com acesso ao desvio-padrão nativo do GP.
    """

    def __init__(self, dml):
        self.gp = dml.model_final_

    def _predict(self, feats: np.ndarray):
        # macs: BLAS "Accelerate" da Apple emite RuntimeWarnings espúrios em
        # matmul mesmo com valores finitos — suprimimos pontualmente aqui.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self.gp.predict(feats, return_std=True)

    def effect_and_std(self, X_test: np.ndarray):
        """CATE predito (linha média) e incerteza agregada por ponto."""
        X = np.atleast_2d(np.asarray(X_test, dtype=float))
        f1 = np.column_stack([np.ones(len(X)), X])
        f0 = np.zeros_like(f1)
        mu1, s1 = self._predict(f1)
        mu0, s0 = self._predict(f0)
        cate = np.ravel(mu1) - np.ravel(mu0)
        std = np.sqrt(np.ravel(s1) ** 2 + np.ravel(s0) ** 2)
        return cate, std

    def confidence_band(self, X_test: np.ndarray, level: float = 0.95):
        """Banda de confiança (lower, upper) para o nível informado."""
        cate, std = self.effect_and_std(X_test)
        z = 1.959963984540054 if abs(level - 0.95) < 1e-9 else _z_from_level(level)
        return cate - z * std, cate + z * std

    def coverage(self, X_eval: np.ndarray, true_te: np.ndarray, level: float = 0.95):
        """Avalia predições em X_eval contra o efeito real: RMSE e cobertura."""
        lower, upper = self.confidence_band(X_eval, level)
        cate, _ = self.effect_and_std(X_eval)
        true_te = np.ravel(true_te)
        rmse = float(np.sqrt(np.mean((cate - true_te) ** 2)))
        inside = np.mean((true_te >= lower) & (true_te <= upper))
        return {"rmse": rmse, "coverage": float(inside), "level": level}


def _z_from_level(level: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(0.5 + level / 2))
