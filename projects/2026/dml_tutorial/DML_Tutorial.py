# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Double Machine Learning + validação de CATE
#
# **Do confounding ao *ranking* honesto de quem tratar.**
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xrhd/mula/blob/main/projects/2026/dml_tutorial/DML_Tutorial.ipynb)
#
# Neste tutorial você vai:
# 1. Ver por que diferença de médias **não** é efeito causal (o problema do confounding);
# 2. A mecânica do **Double ML** (Chernozhukov et al., 2018), equação por equação;
# 3. Estimar efeitos heterogêneos por cliente (o CATE) com `LinearDML` do **EconML**;
# 4. Avaliar o modelo **sem ter o contrafactual**, com o teste de calibração (Dwivedi et al., 2020) e o coeficiente **Qini** (Radcliffe, 2007) via `econml.validate.DRTester`.
#
# > **Público:** assumimos familiaridade com ML supervisionado (regressão, validação
# > cruzada, boosting). Os conceitos de inferência causal são construídos do zero,
# > sempre com uma analogia com ML clássico ao lado.
#
# > **Narrativa:** somos um app de delivery. Distribuímos cupons de desconto (`T`) e
# > queremos saber o quanto cada cliente gasta a mais por causa do cupom (`Y`).
# > Mais importante: descobrir **para quem** o cupom funciona melhor.

# %% [markdown]
# ## Guia rápido de siglas
#
# | sigla | nome | o que é aqui |
# |---|---|---|
# | **DGP** | *Data Generating Process* | o processo que gera os dados; aqui, uma simulação com resposta conhecida |
# | **DAG** | *Directed Acyclic Graph* | diagrama de setas que declara quais variáveis causam quais |
# | **ATE** | *Average Treatment Effect* | efeito médio do tratamento na população toda |
# | **CATE** | *Conditional ATE* | efeito médio do tratamento para clientes com perfil `X = x` |
# | **GATE** | *Group ATE* | CATE de um grupo (ex.: um quartil), usado no teste de calibração |
# | **DML** | *Double Machine Learning* | o método que estima CATE sem viés de confounding, usando ML em duas etapas |
# | **DR** | *Doubly Robust* | estimador que combina dois modelos auxiliares e basta acertar um deles |
# | **BLP** | *Best Linear Predictor* | teste que regredir o efeito real (DR) na predição do modelo |
# | **Qini** | coeficiente de Qini | área que mede o ganho de priorizar clientes pelo CATE vs. aleatório |
# | **TOC / AUTOC** | *Targeted Operator Characteristic* | curva irmã da Qini, sem ponderar pelo volume tratado |
# | **IC** | intervalo de confiança | faixa de incerteza de uma estimativa |
# | **IV** | *Instrumental Variables* | técnica para quando há confounder não observado (citada no fim) |

# %% [markdown]
# ## 0. Setup

# %%
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "econml", "lightgbm", "plotly", "nbformat"],
        check=True,
    )

import warnings

import matplotlib.pyplot as plt

# %matplotlib inline

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split

from econml.dml import LinearDML
from econml.validate import DRTester

SEED = 42
# econml 0.17 emite RuntimeWarnings numéricos (divide-by-zero em matmul) benignos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ## 1. Por que correlação ≠ efeito causal
#
# Pense em cada cliente como uma linha do dataset com **dois labels possíveis**: o
# gasto se receber o cupom, `Y(1)`, e o gasto se não receber, `Y(0)`. O efeito do
# cupom para o cliente `i` seria
#
# $$
# \tau_i = Y_i(1) - Y_i(0)
# $$
#
# mas a vida real só revela um dos dois. O outro é o **contrafactual**, e nunca é
# observado. Em termos de ML, é um problema de regressão em que metade dos targets
# está faltando, e faltando de forma sistemática: quem recebeu o cupom foi escolhido
# a dedo pelo time de marketing, justamente os clientes mais engajados.
#
# Os dois objetos que queremos estimar:
#
# $$
# \mathrm{ATE} = E\big[Y(1) - Y(0)\big], \qquad
# \tau(x) = E\big[Y(1) - Y(0) \,\big|\, X = x\big]
# $$
#
# O **ATE** é o efeito médio na população. O **CATE** é o efeito médio dado o perfil
# `x`, e é ele que responde "para quem o cupom funciona melhor".

# %% [markdown]
# O engajamento (`X`) é um **confounder**: uma variável que causa ao mesmo tempo
# quem recebe o cupom e quanto a pessoa gasta. No DAG (o diagrama causal) abaixo,
# ela abre um caminho indireto `T ← X → Y` que contamina a comparação simples
# entre tratados e controles:

# %% [markdown]
# ```mermaid
# graph LR
#   X["X: engajamento<br/>(freq., idade da conta, ticket)"] --> T["T: recebeu cupom"]
#   X --> Y["Y: gasto"]
#   T --> Y
# ```
#
# > **Analogia com ML:** comparar as médias de gasto de tratados e controles é como
# > avaliar um modelo num conjunto de teste com **viés de seleção**: o grupo
# > tratado não é uma amostra aleatória da população, então a "métrica" mede
# > `efeito do cupom + diferença de engajamento`, tudo misturado.

# %% [markdown]
# ## 2. O DGP: dados simulados com ground truth
#
# > **Analogia com ML:** vamos fazer o equivalente causal do `make_regression` do
# > sklearn: simular o **DGP** (*Data Generating Process*, o processo que gera os
# > dados) com coeficientes conhecidos. Assim conhecemos o efeito verdadeiro `τ(x)`
# > e podemos medir diretamente se o método recupera a resposta certa.
#
# O processo, batendo linha a linha com o código abaixo:
#
# $$
# x_0,\, x_1 \sim \mathcal{N}(0, 1), \qquad x_2 = 0.6\,x_0 + 0.8\,u, \;\; u \sim \mathcal{N}(0, 1)
# $$
# $$
# T \sim \text{Bernoulli}\big(\sigma(0.8\,x_0 + 0.4\,x_2)\big) \quad\text{[propensão]}
# $$
# $$
# \tau(x) = 5 + 3\,x_0, \qquad
# g(x) = 30 + 4\,x_0 + 2\sin(2\,x_1) + x_2
# $$
# $$
# Y = g(X) + \tau(X)\,T + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 2^2)
# $$
#
# onde:
#
# - `x₀, x₁, x₂` são as features do cliente (frequência passada, idade da conta,
#   ticket médio), com `x₂` correlacionada a `x₀`;
# - a **propensão** `e(x) = σ(f(x))` é a probabilidade de receber cupom dado o
#   perfil. Note que é o mesmo modelo da **regressão logística**: aqui ela *gera*
#   o dado, e na seção 3 vamos *estimá-la* com um classificador;
# - `g(x)` é o gasto esperado sem cupom (o baseline);
# - `τ(x)` é o efeito do cupom, **heterogêneo**: cresce com a frequência `x₀`.
#
# Note que `X` entra em **três** lugares (propensão, `g(·)` e `τ(·)`), reproduzindo
# o DAG da seção 1.

# %%
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate_coupons(n=5000, seed=SEED):
    """Cupom (T binário) -> gasto (Y), com confounding via X e efeito heterogêneo."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, n)                       # frequência passada
    x1 = rng.normal(0, 1, n)                       # idade da conta
    x2 = 0.6 * x0 + 0.8 * rng.normal(0, 1, n)      # ticket médio (correl. com x0)
    X = np.column_stack([x0, x1, x2])

    e_x = sigmoid(0.8 * x0 + 0.4 * x2)             # quem recebe cupom (propensão)
    T = (rng.uniform(size=n) < e_x).astype(int)

    true_tau = 5.0 + 3.0 * x0                      # efeito real: heterogêneo em x0
    g_x = 30 + 4 * x0 + 2 * np.sin(2 * x1) + x2    # baseline de gasto
    Y = g_x + true_tau * T + rng.normal(0, 2, n)

    return X, T, Y, true_tau, e_x


X, T, Y, true_tau, e_x = simulate_coupons()
pd.DataFrame(X, columns=["frequencia", "idade_conta", "ticket_medio"]).describe().T.round(2)

# %%
fig = go.Figure()
fig.add_histogram(x=e_x[T == 1], name="tratados", opacity=0.7)
fig.add_histogram(x=e_x[T == 0], name="controles", opacity=0.7)
fig.update_layout(
    barmode="overlay", height=350,
    title="Propensão e(x): tratados e controles são populações diferentes",
    xaxis_title="probabilidade prevista de receber cupom", yaxis_title="n",
)
fig.show()

# %% [markdown]
# > **Analogia com ML:** a propensão é o score de um classificador que prevê quem
# > recebe cupom. Se esse classificador acertasse demais (AUC perto de 1), não
# > existiriam tratados e controles parecidos para comparar, e nada do que vem
# > adiante funcionaria. Essa condição se chama **overlap** (ou positividade).

# %%
# diferença de médias naïve vs efeito real médio
naive = Y[T == 1].mean() - Y[T == 0].mean()
true_ate = true_tau.mean()

fig = go.Figure(go.Bar(
    x=["naïve: E[Y|T=1]−E[Y|T=0]", "efeito real (ATE)"],
    y=[naive, true_ate], text=[f"{naive:.1f}", f"{true_ate:.1f}"], textposition="auto",
))
fig.update_layout(height=350, title="O viés do confounder em números", yaxis_title="efeito estimado")
fig.show()

print(f"naïve = {naive:.2f} | ATE real = {true_ate:.2f} | viés = {naive - true_ate:+.2f}")

# %% [markdown]
# A estimativa naïve **sobrestima** o cupom: parte do gasto extra dos tratados já
# existiria sem cupom nenhum (é o termo `4·x₀` do baseline). Precisamos isolar a
# seta `T → Y` do DAG. É exatamente isso que o DML faz.

# %% [markdown]
# ## 3. As equações do Double ML
#
# O **Double ML** ([Chernozhukov et al., 2018](https://arxiv.org/abs/1608.00060))
# resolve o problema com duas etapas de ML supervisionado que você já conhece, mais
# um ingrediente estatístico que impede os erros dessas etapas de contaminar a
# resposta. Vamos por partes.
#
# ### 3.1 O modelo parcialmente linear
#
# É o exemplo que abre o paper original ([Seção 1](https://arxiv.org/abs/1608.00060)):
#
# $$
# Y = \theta(X)\,T + g(X) + \varepsilon, \qquad T = m(X) + \eta
# $$
#
# com as hipóteses $E[\varepsilon \mid X, T] = 0$ e $E[\eta \mid X] = 0$: o ruído do
# gasto não carrega informação extra, e a parte do tratamento não explicada por `X`
# é aleatória (é ela que funciona como "experimento").
#
# - `θ(X)` é o CATE, o alvo que queremos aprender;
# - `g(X) = E[Y \mid X, T=0]` e `m(X) = E[T \mid X]` são as **funções de nuisance**
#   (em inglês, "incômodas"): precisamos estimá-las, mas não são o objetivo.
#
# > **Analogia com ML:** as nuisances são **modelos auxiliares de primeira etapa**,
# > como um encoder ou um pré-processamento: servem ao modelo final. Aqui, `m(X)` é
# > um classificador de propensão e `E[Y|X]` é um regressor padrão.
#
# ### 3.2 Residualização (o truque central)
#
# Estime os dois modelos supervisionados, $\hat\ell(X) = \hat E[Y|X]$ (regressão) e
# $\hat m(X) = \hat E[T|X]$ (classificação), e subtraia as predições dos valores
# observados:
#
# $$
# \tilde Y = Y - \hat\ell(X), \qquad \tilde T = T - \hat m(X)
# $$
#
# O resíduo `T̃` é a parte do tratamento **não explicada pelo confounder**, como se
# sobrasse apenas a variação experimental do cupom. Regredir `Ỹ` em `T̃` recupera
# `θ` sem o viés de `X`.
#
# > **Analogia com ML:** é a mesma ideia do **gradient boosting**, em que cada
# > estágio aprende sobre o resíduo do anterior. Em regressão linear isso é o
# > clássico teorema de Frisch–Waugh–Lovell: regredir `Y` em `T` controlando `X`
# > dá o mesmo coeficiente que regredir resíduo em resíduo. O DML é essa versão com
# > ML no lugar da regressão linear.
#
# Um detalhe: usamos `ℓ(X) = E[Y|X]` e não `g(X)`, porque na prática prevemos `Y`
# sem olhar `T`. As duas funções se relacionam por `ℓ = θ·m + g`.
#
# ### 3.3 Por que isso é *debiased*
#
# O estimador final resolve o **momento ortogonal de Neyman** (a condição formal é
# a Definição 2.1 do paper):
#
# $$
# \psi\big(W; \theta, \eta\big) = \big(\tilde Y - \theta(X)\,\tilde T\big)\,\tilde T,
# \qquad \frac{\partial}{\partial \eta}\,E[\psi] = 0
# $$
#
# A derivada zerada em relação a `η` (os parâmetros das nuisances) é a propriedade
# chave: perto do ótimo, o objetivo é **plano** na direção dos erros dos modelos
# auxiliares, então esses erros afetam `θ̂` apenas em **segunda ordem** (ao quadrado).
#
# > **Analogia com ML:** pense numa loss **plana na direção dos hiperparâmetros
# > auxiliares** perto do mínimo: errar um pouco o modelo auxiliar quase não move o
# > resultado, porque o termo de primeira ordem da expansão de Taylor é zero. A
# > condição técnica é que o produto dos erros das nuisances seja `o(n^{-1/2})`;
# > em palavras, os modelos auxiliares só precisam ser razoáveis, não perfeitos.
#
# ### 3.4 Cross-fitting
#
# Estimar as nuisances e o `θ` na **mesma** amostra deixa o overfitting da primeira
# etapa vazar para a segunda. A solução: dividir os dados em K folds, treinar as
# nuisances em K−1 folds, prever no fold restante, e repetir trocando o fold (é o
# argumento `cv=5` do EconML). Os algoritmos completos (DML1 e DML2) estão na
# Seção 3 do paper.
#
# > **Analogia com ML:** é exatamente o mecanismo das **predições out-of-fold do
# > stacking** (`cross_val_predict` do sklearn): cada resíduo é gerado por um modelo
# > que nunca viu aquela amostra.

# %% [markdown]
# ```mermaid
# flowchart LR
#   D[dados] --> K[K folds]
#   K --> MY["model_y<br/>E[Y|X]"]
#   K --> MT["model_t<br/>E[T|X]"]
#   MY --> R["resíduos<br/>Ỹ, T̃"]
#   MT --> R
#   R --> F["model_final<br/>regressão de Ỹ em T̃·X"]
#   F --> C["CATE τ̂(x)"]
# ```

# %% [markdown]
# ## 4. Fit com `LinearDML`
#
# O [`LinearDML`](https://www.pywhy.org/EconML/_autosummary/econml.dml.LinearDML.html)
# assume o CATE **linear nas features**: `τ(x) = β₀ + βᵀx`. A vantagem
# é a interpretabilidade: cada coeficiente diz quanto aquela feature muda o efeito
# do cupom. As nuisances continuam livres e não-lineares (aqui, LightGBM).
#
# Por dentro, o modelo final é uma **regressão linear com MSE** sobre features de
# interação entre o resíduo do tratamento e `X`:
#
# $$
# \min_{\beta}\; \sum_i \Big( \tilde Y_i - (\beta_0 + \beta^\top X_i)\,\tilde T_i \Big)^2
# $$
#
# > **Split honesto:** guardamos uma validação que o modelo de CATE nunca vê, como
# > em qualquer pipeline supervisionado. É nela que o `DRTester` vai avaliar tudo
# > nas seções 5 a 8.

# %%
X_train, X_val, T_train, T_val, Y_train, Y_val = train_test_split(
    X, T, Y, test_size=0.25, random_state=SEED, stratify=T,
)

lgbm_kwargs = dict(n_estimators=50, max_depth=3, random_state=SEED, verbose=-1)
est = LinearDML(
    model_y=LGBMRegressor(**lgbm_kwargs),
    model_t=LGBMClassifier(**lgbm_kwargs),
    discrete_treatment=True,
    cv=5,
    random_state=SEED,
)
est.fit(Y_train, T_train, X=X_train)
est.summary()

# %%
ate = est.ate(X_val)
ate_lo, ate_hi = est.ate_interval(X_val)
print(f"ATE estimado = {ate:.2f}  IC95% [{ate_lo:.2f}, {ate_hi:.2f}]")
print(f"ATE real     = {true_ate:.2f}   |   naïve era {naive:.2f}")

# %%
order = np.argsort(X_val[:, 0])
x_sorted = X_val[order]
tau_hat = est.effect(x_sorted)
lo, hi = est.effect_interval(x_sorted)
tau_true_sorted = 5.0 + 3.0 * x_sorted[:, 0]

fig = go.Figure()
fig.add_scatter(x=x_sorted[:, 0], y=tau_true_sorted, name="τ(x) real", line=dict(dash="dash"))
fig.add_scatter(x=x_sorted[:, 0], y=tau_hat, name="τ̂(x) LinearDML")
fig.add_scatter(
    x=np.concatenate([x_sorted[:, 0], x_sorted[::-1, 0]]),
    y=np.concatenate([hi, lo[::-1]]),
    fill="toself", fillcolor="rgba(0,114,178,0.2)", line=dict(color="rgba(255,255,255,0)"),
    name="IC 95%", showlegend=True,
)
fig.update_layout(height=400, title="CATE: estimado (com IC) vs real", xaxis_title="x₀ (frequência)", yaxis_title="efeito do cupom")
fig.show()

# %% [markdown]
# O DML removeu o viés de nível (o ATE estimado bate com o real) e capturou a
# **heterogeneidade**: o efeito cresce com `x₀`, como no DGP. O `IC 95%` (intervalo
# de confiança) quantifica a incerteza de cada coeficiente, no mesmo espírito do
# erro padrão de uma regressão.
#
# Agora a pergunta difícil: e se **não** soubéssemos o ground truth? No mundo real
# não existe `τ(x)` para comparar. Como medir a qualidade do `τ̂(x)`?

# %% [markdown]
# ## 5. Avaliando CATE sem contrafactual
#
# Em ML supervisionado, você avaliaria o modelo comparando predição com label de
# validação. Aqui o label não existe: nunca vemos o mesmo cliente com **e** sem
# cupom, então `τᵢ` é inobservável.
#
# A saída do [`DRTester`](https://www.pywhy.org/EconML/_autosummary/econml.validate.DRTester.html#econml-validate-drtester)
# é **fabricar um label**: o pseudo-outcome *doubly-robust*
#
# $$
# Y_i^{DR} = \hat\mu_1(X_i) - \hat\mu_0(X_i)
# + \frac{T_i\,(Y_i - \hat\mu_1(X_i))}{\hat e(X_i)}
# - \frac{(1 - T_i)\,(Y_i - \hat\mu_0(X_i))}{1 - \hat e(X_i)}
# \qquad\text{tal que}\qquad
# E[Y^{DR}\,|\,X] = \tau(X)
# $$
#
# onde `μ₁` e `μ₀` são regressores de `Y` treinados só com tratados e só com
# controles, e `e` é o classificador de propensão. O primeiro termo é a diferença
# das predições; os dois últimos corrigem o erro de cada modelo, com peso inversamente
# proporcional à propensão.
#
# > **Analogia com ML:** o termo `T / e(X)` é **importance weighting**, o mesmo
# > truque de covariate shift: um cliente tratado apesar da propensão baixa vale
# > mais, porque representa uma região rara do espaço. O resultado é um label
# > sintético, ruidoso ponto a ponto, mas sem viés na média condicional. A partir
# > daí, avaliar `τ̂` vira um problema supervisionado comum.
#
# > O "doubly robust" funciona como um **ensemble com fallback**: basta `μ` **ou**
# > `e` estarem certos para o alvo ser correto. É isso que `tester.fit_nuisance(...)`
# > constrói, com cross-fitting interno (`cv=5`).

# %%
tester = DRTester(
    model_regression=LGBMRegressor(**lgbm_kwargs),
    model_propensity=LGBMClassifier(**lgbm_kwargs),
    cate=est,
    cv=5,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tester.fit_nuisance(X_val, T_val, Y_val, X_train, T_train, Y_train)
    tester.get_cate_preds(X_val, X_train)
    res = tester.evaluate_all(Xval=X_val, Xtrain=X_train, n_bootstrap=500)

res.summary()

# %% [markdown]
# Leitura rápida do sumário (todas as métricas usam `Y^DR` como label):
#
# - `blp_est` / `blp_pval`: o teste **BLP** (*Best Linear Predictor*, de
#   [Chernozhukov et al., 2022](https://arxiv.org/abs/1712.04802)) regredir o
#   label DR na predição do modelo. Se o CATE captura heterogeneidade real, a
#   inclinação fica perto de 1 e significativa. É o análogo causal da *slope* de
#   calibração de um regressor;
# - `cal_r_squared`: calibração dos subgrupos (seção 6);
# - `qini_est` / `qini_pval`: ganho de priorização sobre targeting aleatório (seção 7);
# - `autoc_est`: a variante TOC, mesma ideia do Qini sem ponderar pelo volume tratado.
#
# Vamos abrir as duas métricas principais.

# %% [markdown]
# ## 6. Teste de calibração ([Dwivedi et al., 2020](https://arxiv.org/abs/2008.10109))
#
# **Pergunta:** os subgrupos que o modelo diz terem efeitos diferentes *de fato*
# os têm?
#
# > **Analogia com ML:** é o **reliability diagram** da calibração de classificadores
# > (`sklearn.calibration_curve`): agrupamos as predições em quantis e comparamos,
# > em cada grupo, a média prevista com a média observada do label.
#
# 1. Divida a validação em grupos pelos quantis de `τ̂(x)` (por padrão, quartis);
# 2. Em cada grupo `k`, compare a média predita $\overline{\hat\tau}_k$ com o
#    **GATE** (*Group Average Treatment Effect*), a média do label DR no grupo;
# 3. Meça o erro de calibração e a variação entre grupos:
#
# $$
# \mathrm{Cal}_G = \sum_k \pi(k)\,\big|\, \mathrm{GATE}_k - \overline{\hat\tau}_k \,\big|,
# \qquad
# \mathrm{Cal}_O = \sum_k \pi(k)\,\big|\, \mathrm{GATE}_k - \mathrm{ATE} \,\big|
# $$
# $$
# \boxed{\ \mathcal{R}^2_C = 1 - \frac{\mathrm{Cal}_G}{\mathrm{Cal}_O}\ }
# $$
#
# onde `π(k)` é a fração da amostra no grupo `k`.
#
# **Leitura:** é o mesmo espírito do R² da regressão (`1 − erro do modelo / erro do
# baseline`), com duas trocas: o baseline é o ATE constante, e o erro é absoluto
# (estilo MAE), não quadrático. `R²_C → 1`: os grupos são tão diferentes quanto o
# modelo afirma, ou seja, heterogeneidade real. `R²_C ≤ 0`: os grupos explicam a
# variação **pior** que o ATE constante; a "heterogeneidade" era ruído.

# %%
tmt = tester.treatments[1]
ax = res.plot_cal(tmt=tmt)
ax.figure.set_size_inches(6, 4)
plt.show()

# %%
# A mesma leitura, em Plotly: pontos perto da reta y=x = bem calibrado
df_cal = res.cal.plot_data_dict[tmt]

fig = go.Figure()
lim = [df_cal[["gate", "g_cate"]].min().min(), df_cal[["gate", "g_cate"]].max().max()]
fig.add_scatter(x=lim, y=lim, mode="lines", name="calibração perfeita", line=dict(dash="dash"))
fig.add_scatter(
    x=df_cal["g_cate"], y=df_cal["gate"], mode="markers+lines", name="grupos (quartis)",
    error_y=dict(type="data", array=1.96 * df_cal["se_gate"]),
    text=[f"grupo {i}" for i in df_cal["ind"]],
)
fig.update_layout(
    height=400, title=f"Calibração: GATE (real, DR) vs τ̂ predito. R²_C = {res.cal.cal_r_squared[0]:.3f}",
    xaxis_title="τ̂ médio predito no grupo", yaxis_title="GATE (E[Y^DR | grupo])",
)
fig.show()

# %% [markdown]
# ## 7. Qini: o cupom vale mais para quem? (Radcliffe, 2007)
#
# **Pergunta:** se o orçamento só permite mandar cupom para os top-q% segundo `τ̂`,
# quanto de efeito extra ganhamos em relação a mandar para q% aleatórios?
#
# > **Analogia com ML:** é o **cumulative gains chart** (ou lift) de campanhas:
# > ordenar a base pelo score e medir o retorno acumulado em cada corte. A área sob
# > a curva tem o mesmo papel da AUC num ranking, e o targeting aleatório faz o
# > papel da diagonal.
#
# Ordene a validação por `τ̂` decrescente e, para cada quantil `q`, defina:
#
# $$
# \tau_{QINI}(q) = \mathrm{Cov}\big( Y^{DR},\ \mathbb{1}\{\,\hat\tau(Z) \ge \hat\mu(q)\,\} \big)
# = q\,\Big(\ E[Y^{DR} \mid \hat\tau \ge \hat\mu(q)] - E[Y^{DR}]\ \Big)
# $$
# $$
# \boxed{\ \mathrm{QINI} = \int_0^1 \tau_{QINI}(q)\,dq\ }
# $$
#
# - É o efeito médio no top-q **menos** o ATE, ponderado pelo volume `q`: o **ganho
#   sobre targeting aleatório**;
# - `qini_est` é a área sob essa curva; `qini_pval` vem de bootstrap;
# - O **TOC/AUTOC** é o mesmo objeto sem o peso `q`. Num paralelo com ranking, o TOC
#   mede a qualidade no topo (estilo precision@k), enquanto o Qini mede o ganho
#   total da política, que cresce com o volume tratado.

# %%
ax = res.plot_qini(tmt=tmt)
ax.figure.set_size_inches(6, 4)
plt.show()

# %%
# Qini na mão (~10 linhas) para mostrar que a métrica não é caixa preta
dr_val = tester.dr_val_[:, 0]          # pseudo-outcome DR na validação
cate_val = est.effect(X_val)           # score de priorização

def qini_curve(scores, dr):
    order = np.argsort(-scores)
    dr_sorted = dr[order]
    q = np.arange(1, len(dr) + 1) / len(dr)
    return q, np.cumsum(dr_sorted) / len(dr) - q * dr.mean()

q_manual, curve_manual = qini_curve(cate_val, dr_val)
q_random, curve_random = qini_curve(np.random.default_rng(SEED).normal(size=len(dr_val)), dr_val)

fig = go.Figure()
fig.add_scatter(x=q_manual, y=curve_manual, name="Qini do τ̂(x)")
fig.add_scatter(x=q_random, y=curve_random, name="targeting aleatório (baseline)", line=dict(dash="dash"))
fig.add_scatter(x=[0, 1], y=[0, 0], mode="lines", name="zero", line=dict(color="gray", width=1))
fig.update_layout(height=400, title="Curva Qini calculada à mão (mesma lógica do DRTester)",
                  xaxis_title="fração tratada (top-q por τ̂)", yaxis_title="ganho acumulado sobre aleatório")
fig.show()

# %%
ax = res.plot_toc(tmt=tmt)
ax.figure.set_size_inches(6, 4)
plt.show()

# %% [markdown]
# ## 8. Diagnóstico: e se o modelo estiver errado?
#
# Métrica que não reprova modelo ruim não serve para nada. Como sanity check,
# treinamos um CATE **deliberadamente quebrado**: o mesmo `LinearDML`, mas com as
# linhas de `X` embaralhadas, destruindo a relação entre features e efeito.
#
# > **Analogia com ML:** é o clássico **teste de permutação**: um modelo treinado
# > em dados embaralhados deve performar no nível do acaso. Se as métricas
# > aprovarem esse modelo, o problema está nelas.

# %%
rng = np.random.default_rng(SEED)
X_bad_train = X_train[rng.permutation(len(X_train))]
X_bad_val = X_val[rng.permutation(len(X_val))]

est_bad = LinearDML(
    model_y=LGBMRegressor(**lgbm_kwargs),
    model_t=LGBMClassifier(**lgbm_kwargs),
    discrete_treatment=True, cv=5, random_state=SEED,
)
est_bad.fit(Y_train, T_train, X=X_bad_train)

tester_bad = DRTester(
    model_regression=LGBMRegressor(**lgbm_kwargs),
    model_propensity=LGBMClassifier(**lgbm_kwargs),
    cate=est_bad, cv=5,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tester_bad.fit_nuisance(X_val, T_val, Y_val, X_train, T_train, Y_train)
    tester_bad.get_cate_preds(X_bad_val, X_bad_train)
    res_bad = tester_bad.evaluate_all(Xval=X_bad_val, Xtrain=X_bad_train, n_bootstrap=500)

# %%
cols = ["blp_est", "blp_pval", "qini_est", "qini_pval", "autoc_est", "autoc_pval", "cal_r_squared"]
compare = pd.concat(
    [
        res.summary().set_index("treatment").add_prefix("bom_"),
        res_bad.summary().set_index("treatment").add_prefix("ruim_"),
    ],
    axis=1,
)[["bom_" + c for c in cols] + ["ruim_" + c for c in cols]]
compare.round(3)

# %%
_, curve_bad = qini_curve(est_bad.effect(X_bad_val), dr_val)

fig = go.Figure()
fig.add_scatter(x=q_manual, y=curve_manual, name="modelo bom")
fig.add_scatter(x=q_manual, y=curve_bad, name="modelo ruim (X embaralhado)")
fig.add_scatter(x=[0, 1], y=[0, 0], mode="lines", name="zero", line=dict(color="gray", dash="dash"))
fig.update_layout(height=400, title="O teste reprova o modelo quebrado",
                  xaxis_title="fração tratada (top-q)", yaxis_title="ganho acumulado")
fig.show()

# %% [markdown]
# O padrão de um modelo que **não** captura heterogeneidade: `cal_r_squared ≤ 0`,
# `qini_pval` não significativo e curva Qini colada no zero. É a assinatura que nos
# protege de publicar "efeitos heterogêneos" que eram só ruído.
#
# ## Recap: checklist de validação de CATE
#
# | métrica | o que pergunta | modelo bom | modelo ruim |
# |---|---|---|---|
# | **BLP** (`blp_pval`) | `Y^DR` cresce com `τ̂`? | p < 0.05 | p alto |
# | **Calibração** (`cal_r_squared`) | GATEs batem com o predito? | → 1 | ≤ 0 |
# | **Qini** (`qini_est/pval`) | priorizar por τ̂ ganha do aleatório? | área > 0, p < 0.05 | área ≈ 0 |
# | **AUTOC** (`autoc_est/pval`) | há heterogeneidade no topo? | > 0, p < 0.05 | ≈ 0 |

# %% [markdown]
# ## 9. Baseline: e se não houvesse ortogonalização? (S-learner)
#
# Antes de concluir, fica a pergunta: precisava mesmo de todo o aparato do DML? A
# alternativa mais simples é a família dos **meta-learners** ([Künzel et al.,
# 2019](https://arxiv.org/abs/1706.03461)):
#
# - **S-learner**: um único modelo supervisionado treinado em `[X, T]`; o CATE é a
#   diferença entre prever com `T=1` e com `T=0`;
# - **T-learner**: dois modelos separados, um treinado só nos tratados e outro só
#   nos controles; o CATE é a diferença das predições;
# - **X-learner**: extensão do T-learner que combina os efeitos imputados dos dois
#   grupos.
#
# > **Analogia com ML:** o S-learner trata o problema como feature engineering:
# > joga `T` como mais uma feature e confia que o modelo a usa direito.
#
# O problema de todos eles: o CATE sai da **diferença de dois modelos de outcome**,
# sem residualização nem ortogonalização. Lembre da seção 3.3: sem a ortogonalidade,
# o erro do modelo de outcome entra **em primeira ordem** na estimativa do efeito.
# Na prática, dois modos de falha:
#
# - **subestimação (atenuação):** a variável `T` compete com features de baseline
#   fortes dentro do modelo; com regularização, o modelo gasta capacidade em `g(x)`
#   e o efeito sai encolhido;
# - **superestimação:** quando os grupos são muito diferentes (overlap fraco),
#   diferenças de baseline vazam para a diferença de predições.
#
# Vamos testar o mais simples deles, o
# [`SLearner`](https://www.pywhy.org/EconML/_autosummary/econml.metalearners.SLearner.html)
# do próprio EconML, no mesmo DGP e no mesmo split:

# %%
from econml.metalearners import SLearner

s_learner = SLearner(overall_model=LGBMRegressor(**lgbm_kwargs))
s_learner.fit(Y_train, T_train, X=X_train)
cate_s = s_learner.effect(X_val)

true_tau_val = 5.0 + 3.0 * X_val[:, 0]
print(f"S-learner:  efeitos entre {cate_s.min():5.2f} e {cate_s.max():5.2f}")
print(f"Real:       efeitos entre {true_tau_val.min():5.2f} e {true_tau_val.max():5.2f}")
print(f"LinearDML:  efeitos entre {cate_val.min():5.2f} e {cate_val.max():5.2f}")

# %%
fig = go.Figure()
fig.add_scatter(x=x_sorted[:, 0], y=tau_true_sorted, name="τ(x) real", line=dict(dash="dash"))
fig.add_scatter(x=x_sorted[:, 0], y=tau_hat, name="τ̂(x) LinearDML")
fig.add_scatter(x=x_sorted[:, 0], y=cate_s[order], name="τ̂(x) S-learner")
fig.update_layout(height=400, title="O S-learner comprime os extremos do efeito",
                  xaxis_title="x₀ (frequência)", yaxis_title="efeito do cupom")
fig.show()

# %%
tester_s = DRTester(
    model_regression=LGBMRegressor(**lgbm_kwargs),
    model_propensity=LGBMClassifier(**lgbm_kwargs),
    cate=s_learner,
    cv=5,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tester_s.fit_nuisance(X_val, T_val, Y_val, X_train, T_train, Y_train)
    tester_s.get_cate_preds(X_val, X_train)
    res_s = tester_s.evaluate_all(Xval=X_val, Xtrain=X_train, n_bootstrap=500)

compare_s = pd.concat(
    [
        res.summary().set_index("treatment").add_prefix("dml_"),
        res_s.summary().set_index("treatment").add_prefix("slearner_"),
    ],
    axis=1,
)[["dml_" + c for c in cols] + ["slearner_" + c for c in cols]]
compare_s.round(3)

# %%
# Calibração comparada: quem acerta os GATEs por quartil?
df_cal_dml = res.cal.plot_data_dict[tmt]
df_cal_s = res_s.cal.plot_data_dict[tmt]

lim = [
    min(df_cal_dml[["gate", "g_cate"]].min().min(), df_cal_s[["gate", "g_cate"]].min().min()),
    max(df_cal_dml[["gate", "g_cate"]].max().max(), df_cal_s[["gate", "g_cate"]].max().max()),
]
fig = go.Figure()
fig.add_scatter(x=lim, y=lim, mode="lines", name="calibração perfeita", line=dict(dash="dash", color="gray"))
for df, name, r2 in [
    (df_cal_dml, "LinearDML", res.cal.cal_r_squared[0]),
    (df_cal_s, "S-learner", res_s.cal.cal_r_squared[0]),
]:
    fig.add_scatter(
        x=df["g_cate"], y=df["gate"], mode="markers+lines",
        name=f"{name} (R²_C = {r2:.2f})",
        error_y=dict(type="data", array=1.96 * df["se_gate"]),
        text=[f"quartil {i}" for i in df["ind"]],
    )
fig.update_layout(
    height=400, title="Calibração comparada: GATE (real, DR) vs τ̂ predito",
    xaxis_title="τ̂ médio predito no grupo", yaxis_title="GATE (E[Y^DR | grupo])",
)
fig.show()

# %%
# Qini comparado: quanto cada ranking ganha do targeting aleatório?
_, curve_s = qini_curve(cate_s, dr_val)

fig = go.Figure()
fig.add_scatter(x=q_manual, y=curve_manual, name="LinearDML")
fig.add_scatter(x=q_manual, y=curve_s, name="S-learner")
fig.add_scatter(x=[0, 1], y=[0, 0], mode="lines", name="zero", line=dict(color="gray", dash="dash"))
fig.update_layout(height=400, title="Qini comparado: quem prioriza melhor",
                  xaxis_title="fração tratada (top-q)", yaxis_title="ganho acumulado sobre aleatório")
fig.show()

# %% [markdown]
# **Leitura:** o S-learner não é um desastre como o modelo embaralhado da seção 8
# (a correlação com o efeito real é alta e ele passa no BLP), mas ele **comprime
# os extremos**: nunca prevê efeito negativo, embora o efeito real chegue a valores
# negativos para clientes de baixa frequência, e corta o topo do ranking. Uma
# política de targeting baseada nesse modelo mandaria cupom para clientes que o
# cupom pode *prejudicar*.
#
# O mecanismo: com árvores rasas e regularizadas, a variável `T` compete com o
# baseline forte `g(x)`; capturar `τ(x) = 5 + 3·x₀` exigiria interações `T × x₀`
# (splits em `T` e depois em `x₀` dentro do ramo tratado), e a regularização
# penaliza exatamente esse tipo de estrutura. Sem residualização, esse erro entra
# direto no CATE.
#
# ### E no mundo real, sem o τ(x) verdadeiro?
#
# Neste tutorial comparamos os modelos contra o efeito real porque nós mesmos
# geramos os dados. Fora daqui isso não existe: o que sobra na mão são **a tabela
# e as duas curvas acima**. E a boa notícia é que elas bastam para tomar a decisão
# certa:
#
# - **BLP sozinho não distingue:** os dois modelos passam (p ≈ 0), porque ambos
#   ordenam razoavelmente bem no agregado;
# - **a calibração separa:** `cal_r_squared` cai de 0.81 para 0.71, e o plot mostra
#   onde dói: nos quartis extremos o S-learner prevê efeitos comprimidos em relação
#   aos GATEs;
# - **o Qini separa:** a área cai (1.045 vs 1.005) e a curva do S-learner fica
#   abaixo justamente nos primeiros cortes, que é onde a política de targeting
#   opera;
# - **AUTOC confirma** a mesma leitura no topo do ranking.
#
# É exatamente o checklist do Recap aplicado a uma decisão real de modelo: as
# métricas do `DRTester` apontam o LinearDML como vencedor **sem nunca olhar o
# ground truth**. Moral: meta-learners são ótimos baselines, mas o DML existe
# justamente para tirar o efeito da sombra do baseline.

# %% [markdown]
# ## Quando **não** usar DML
#
# - **Unconfoundedness violada:** o DML só corrige confounders **observados** em X.
#   Se faltar variável no DAG, nenhum residual salva. Nesse caso, o caminho são
#   variáveis instrumentais (IV) ou análise de sensibilidade;
# - **Overlap fraco:** regiões com `e(x) ≈ 0` ou `≈ 1` explodem os pesos do label
#   DR (note o clip em 0.01 no código do EconML). Investigue o plot de propensão
#   da seção 2;
# - **Amostra pequena:** cross-fitting e bootstrap de validação precisam de volume.
#
# ## Próximos passos
#
# - `CausalForestDML` e `DRLearner` para CATE não-linear;
# - *Policy learning*: transformar o `τ̂` numa regra ótima de targeting;
# - Versão probabilística (DML + Processo Gaussiano, com bandas de incerteza por
#   indivíduo): [`projects/2026/gaussian_process_dml`](https://github.com/xrhd/mula/tree/main/projects/2026/gaussian_process_dml).
#
# ## Referências
#
# - Chernozhukov, V. et al. *Double/Debiased Machine Learning for Treatment and
#   Structural Parameters*, Econometrics Journal, 2018. [arXiv:1608.00060](https://arxiv.org/abs/1608.00060)
# - Chernozhukov, V. et al. *Generic Machine Learning Inference on Heterogeneous
#   Treatment Effects in Randomized Experiments*, 2022 (teste BLP). [arXiv:1712.04802](https://arxiv.org/abs/1712.04802)
# - Dwivedi, R. et al. *Stable Discovery of Interpretable Subgroups via Calibration
#   in Causal Studies*, 2020. [arXiv:2008.10109](https://arxiv.org/abs/2008.10109)
# - Radcliffe, N. *Using Control Groups to Target on Predicted Lift*, 2007 (Qini)
# - Künzel, S. et al. *Metalearners for estimating heterogeneous treatment effects
#   using machine learning*, PNAS, 2019 (S/T/X-learners). [arXiv:1706.03461](https://arxiv.org/abs/1706.03461)
# - Docs: [DRTester](https://www.pywhy.org/EconML/_autosummary/econml.validate.DRTester.html),
#   [LinearDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.LinearDML.html),
#   [SLearner](https://www.pywhy.org/EconML/_autosummary/econml.metalearners.SLearner.html),
#   [notebook oficial CATE validation](https://github.com/py-why/EconML/blob/main/notebooks/CATE%20validation.ipynb)
