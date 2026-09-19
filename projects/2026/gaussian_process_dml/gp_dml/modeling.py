"""Instanciação e treino dos modelos DML: GP final e LinearDML benchmark."""

import warnings

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from econml.dml import DML, LinearDML


def build_kernel(name: str = "default"):
    """Kernels com bounds restritos (evita explosão de hiperparâmetros no L-BFGS).

    - 'default': kernel testado no ihdp-b (cal R^2C 0.512→0.67).
    - 'sim': RBF de curto alcance (permite curvatura local do DGP 1-D).
    - 'smooth': RBF ultra-suave (length_scale>=100): GP quase constante;
      tunado no ihdp-a (cal -1.2 -> -0.1; sem heterogeneidade espúria).
    """
    presets = {
        "default": lambda: (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e1))
        ),
        "sim": lambda: (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(5e-2, 5))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1))
        ),
        "smooth": lambda: (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=100.0, length_scale_bounds=(100, 1e3))
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 1e2))
        ),
    }
    if name not in presets:
        raise ValueError(f"kernel desconhecido: {name}; opções: {list(presets)}")
    return presets[name]()


def build_gp(kernel_name: str = "default", seed: int = 42):
    return GaussianProcessRegressor(
        kernel=build_kernel(kernel_name),
        n_restarts_optimizer=3,
        normalize_y=True,
        random_state=seed,
    )


#: preset de kernel por dataset — 'auto' seleciona por nome do dataset
KERNEL_FOR_DATASET = {"sim": "sim", "ihdp-a": "smooth", "ihdp-b": "default"}


def build_models(discrete_treatment: bool, seed: int = 42):
    common = dict(random_state=seed, verbose=-1, force_col_wise=True)
    model_y = LGBMRegressor(n_estimators=30, max_depth=3, **common)
    model_t = (
        LGBMClassifier(n_estimators=30, max_depth=3, **common)
        if discrete_treatment
        else LGBMRegressor(n_estimators=30, max_depth=3, **common)
    )
    return model_y, model_t


def _fit(model, data: dict):
    with warnings.catch_warnings():
        # silencia ruído numérico benigno durante a otimização do GP / econml
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        model.fit(data["Y"], data["T"], X=data["X"])
    return model


def fit_dml(data: dict, seed: int = 42, kernel_name: str = "auto") -> DML:
    """Treina o DML com GP final; dataset sintético -> T contínuo, IHDP -> T binário."""
    discrete = np.unique(data["T"]).size <= 2
    model_y, model_t = build_models(discrete, seed)
    if kernel_name == "auto":
        kernel_name = KERNEL_FOR_DATASET.get(data["name"], "default")
    dml = DML(
        model_y=model_y,
        model_t=model_t,
        model_final=build_gp(kernel_name, seed),
        discrete_treatment=discrete,
        random_state=seed,
    )
    return _fit(dml, data)


def fit_linear_dml(data: dict, seed: int = 42) -> LinearDML:
    """Benchmark: LinearDML com os mesmos nuisances (o diferencial é o final model)."""
    discrete = np.unique(data["T"]).size <= 2
    model_y, model_t = build_models(discrete, seed)
    lm = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=discrete,
        random_state=seed,
    )
    return _fit(lm, data)
