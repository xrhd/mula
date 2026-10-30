# LLMs: GRPO vs SDPO em Gemma-3-1B-IT / GSM8K (Tunix, TPU)

Fine-tuning por RL do Gemma-3-1B-IT (LoRA r=64) no GSM8K com o [Tunix](https://github.com/google/tunix), comparando dois algoritmos:

- **GRPO** ([`grpo_gemma.ipynb`](grpo_gemma.ipynb)): o exemplo oficial do Tunix. Cada resposta recebe uma vantagem escalar relativa ao grupo.
- **SDPO**, Self-Distillation Policy Optimization ([paper](https://arxiv.org/abs/2601.20802), [`sdpo_gemma.ipynb`](sdpo_gemma.ipynb)):
  - o próprio modelo vira *teacher*, re-promptado com uma solução certa de um "irmão" do mesmo grupo;
  - ele ensina token a token, pela JSD sobre os top-K logits;
  - usamos o híbrido do §4.5 do paper: `0.9·GRPO + 0.1·SDPO`.

A pergunta: **o SDPO aprende mais rápido, ou melhor, que o GRPO com o mesmo setup?**
O setup é G=2, LR 3e-6, β=0.08, 3364 passos e a mesma validação.

## Resultado: o SDPO converge ~5× mais rápido em passos, mas não termina melhor

![GRPO vs SDPO](runs/sdpo_logit_l09_v6e/grpo_vs_sdpo_val_score.png)

O eixo y do print do SDPO está em escala log; o do GRPO, linear.

### Tempo para convergir

Primeiro passo em que a reward média de validação (`rewards/eval/score/mean`, máx. 10) cruza cada limiar:

| limiar | GRPO | SDPO (logit, λ=0.9) | aceleração |
|---|---|---|---|
| ≥ 3 | ~1.5k | 320 | ~5× |
| ≥ 4 | ~1.8k | 320 | ~5–6× |
| ≥ 5 | ~2.0–2.1k | 384 | **~5×** |
| pico | ~5.5–5.9 (~2.0–2.2k) | **6.13 (passo 512)** | ~4× antes |
| fim (passo 3328) | **5.19** | 4.61 | — |

Os valores do GRPO são aproximados, lidos do Tensorboard do run de referência. Os do SDPO vêm dos eventos em [`runs/sdpo_logit_l09_v6e/tb`](runs/sdpo_logit_l09_v6e/tb).

- **Por que o SDPO acelera:**
  - O GRPO passa os primeiros ~1000 passos quase parado. O modelo base acerta o formato exato em só 6% das respostas, e com G=2 quase todo grupo tem duas respostas igualmente ruins, ou seja, vantagem zero.
  - No SDPO, basta **uma** resposta certa no grupo: ela vira demonstração, e o teacher ensina o formato e o raciocínio token a token.
  - O paper reporta uma aceleração parecida (~6×).
- **Tempo de relógio:** os dois runs levaram ~6h50, mas em TPUs diferentes (GRPO numa v5e-1, SDPO numa v6e-1). O passo do SDPO custa mais, por causa dos forwards do teacher e do top-K. Então a aceleração vale **em passos**, ou em amostras, e não necessariamente em tempo ou custo na mesma máquina.

### Teste (greedy, 64 exemplos)

| | acerto | parcial | formato |
|---|---|---|---|
| baseline | 48.4% | 50.0% | 6.25% |
| GRPO | 48.4% | 54.7% | **82.8%** |
| SDPO logit λ=0.9 | 45.3% | 46.9% | 67.2% |

- No acerto, é um empate estatístico (erro padrão de ~±6 pp com n=64). No formato, o GRPO termina melhor.
- O SDPO atinge o pico cedo e depois degrada: a entropia, o comprimento das respostas e a KL sobem, e aparecem loops de repetição.
- **O melhor uso aqui é com early stopping.** O checkpoint 512 do SDPO alcança, na validação, o GRPO de ~2000 passos.

## Histórico dos runs

| run | o que mostrou |
|---|---|
| [`runs/sdpo_hybrid_l09`](runs/sdpo_hybrid_l09) | 1º run, SDPO **token-level**. O runtime do Colab caiu no passo ~2369; colapso temporário de formato |
| [`runs/sdpo_hybrid_l09_v6e`](runs/sdpo_hybrid_l09_v6e) | token-level, run completo. **Colapsou para respostas vazias** no passo 884: o *unlikelihood* empurrou a massa para o EOS, e respostas de 0 tokens são um estado absorvente |
| [`runs/sdpo_logit_l09_v6e`](runs/sdpo_logit_l09_v6e) | **logit-level** + proteções: treino estável, convergência ~5× mais rápida, piora depois do pico |

## Como rodar

O `sdpo_gemma.py` (jupytext percent) é a fonte, e o `.ipynb` é gerado a partir dele:

```bash
uvx jupytext --to ipynb sdpo_gemma.py -o sdpo_gemma.ipynb
```

- Rodar no Colab com TPU (v5e-1 ou v6e-1), com `HF_TOKEN` nos secrets.
- `SDPO_SMOKE_TEST=1` roda 5 passos para validar a integração.
- O notebook escolhe o checkpoint de melhor validação para a avaliação final e para o export ao HF.
