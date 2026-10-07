# Tutorial: Double ML + validação de CATE (calibração & Qini)

Um tutorial em notebook, no estilo "aula de YouTube", sobre **causal inference com
Double Machine Learning** e, principalmente, **como avaliar um modelo de efeitos
heterogêneos (CATE) sem nunca observar o contrafactual**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xrhd/mula/blob/main/projects/2026/dml_tutorial/DML_Tutorial.ipynb)

## O que você vai aprender

- **Por que diferença de médias não é efeito causal** — confounding, na prática,
  num DGP de cupons de desconto com ground truth conhecido;
- **Como o Double ML remove o viés** — residualização, momento ortogonal de Neyman
  e cross-fitting, equação por equação (Chernozhukov et al., 2018);
- **Como usar o EconML na prática** — `LinearDML` com LightGBM de nuisance, ATE com
  intervalo de confiança, CATE por indivíduo;
- **Como validar o CATE de forma honesta** — o teste de **calibração** de
  Dwivedi et al. (2020) e o coeficiente **QINI** de Radcliffe (2007), via
  `econml.validate.DRTester`, incluindo o contraste com um modelo deliberadamente
  ruim (o que `cal_r_squared ≤ 0` e `qini_pval` alto significam).

## Rodar localmente com uv

```bash
git clone https://github.com/xrhd/mula && cd mula
uv sync --extra causal
uv run --extra causal jupyter lab projects/2026/dml_tutorial/DML_Tutorial.ipynb
```

Sem clonar o repo, também funciona com o notebook baixado:

```bash
uv run --with econml,lightgbm,plotly,nbformat,jupyterlab jupyter lab DML_Tutorial.ipynb
```

## Rodar no Google Colab

Clique no badge acima (ou no topo do notebook). Nenhum setup local é necessário:
a primeira célula instala as dependências no ambiente remoto.

## Roteiro do notebook

| seção | conceito |
|---|---|
| 1. Correlação ≠ efeito causal | DAG do confounder; viés da diferença de médias |
| 2. O DGP | dados sintéticos com `τ(x)` conhecido; propensão e overlap |
| 3. As equações do DML | modelo parcialmente linear; residualização; ortogonalidade; cross-fitting |
| 4. Fit com `LinearDML` | ATE + IC; CATE vs ground truth |
| 5. Avaliação sem contrafactual | pseudo-outcome doubly-robust (`Y^DR`) |
| 6. Calibração | GATEs por quantil; `R²_C` |
| 7. Qini | curva de uplift; ganho sobre targeting aleatório; AUTOC/TOC |
| 8. Diagnóstico | modelo quebrado reprovado pelas métricas; quando *não* usar DML |

## Editando o notebook

O notebook é mantido com [jupytext](https://jupytext.readthedocs.io/): a **fonte
de verdade** é `DML_Tutorial.py` (formato percent) e o `.ipynb` é derivado e
pareado. Após editar qualquer um dos dois:

```bash
uv run --extra causal jupytext --sync projects/2026/dml_tutorial/DML_Tutorial.py
```

Commit os dois arquivos (o `.py` é o que se revisa em PR; o `.ipynb` carrega os
outputs renderizados no GitHub).

## Referências

- Chernozhukov, V. et al. *Double/Debiased Machine Learning for Treatment and
  Structural Parameters*. Econometrics Journal, 2018. [arXiv:1608.00060](https://arxiv.org/abs/1608.00060)
- Dwivedi, R. et al. *Stable Discovery of Interpretable Subgroups via Calibration
  in Causal Studies*, 2020 (teste de calibração). [arXiv:2008.10109](https://arxiv.org/abs/2008.10109)
- Radcliffe, N. *Using Control Groups to Target on Predicted Lift*, 2007 (QINI).
- Docs: [DRTester](https://www.pywhy.org/EconML/_modules/econml/validate/drtester.html),
  [LinearDML](https://www.pywhy.org/EconML/_modules/econml/dml/dml.html#LinearDML),
  [CATE validation notebook (EconML)](https://github.com/py-why/EconML/blob/main/notebooks/CATE%20validation.ipynb)
