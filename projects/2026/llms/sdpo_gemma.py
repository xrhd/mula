# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   accelerator: TPU
#   colab:
#     gpuType: V5E1
#     machine_shape: hm
#     provenance: []
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # SDPO Demo (Gemma-3-1B-IT, GSM8K)
#
# Adaptação do [`grpo_gemma.ipynb`](./grpo_gemma.ipynb) (exemplo oficial do
# [Tunix](https://github.com/google/tunix/blob/main/examples/grpo_gemma.ipynb))
# trocando o GRPO por **SDPO — Self-Distillation Policy Optimization**
# ([paper](https://arxiv.org/abs/2601.20802) ·
# [blog](https://self-distillation.github.io/SDPO.html) ·
# [código](https://github.com/lasgroup/SDPO)).

# %% [markdown]
# ## O que muda em relação ao GRPO
#
# No GRPO cada resposta recebe um único escalar (a vantagem relativa ao grupo),
# espalhado igualmente por todos os tokens. No SDPO o próprio modelo vira
# **teacher**: ele é re-promptado com uma solução bem-sucedida de um "irmão" do
# mesmo grupo e pontua, token a token, a resposta do student. A loss é uma
# divergência token a token entre student e teacher (stop-grad no teacher),
# o que dá crédito denso por token sem precisar de um modelo de valor.
#
# - **Teacher**: `π_θ'(· | reprompt(x, demo))`, onde `θ'` é uma média EMA dos
#   parâmetros LoRA do student e `demo` é o primeiro irmão bem-sucedido (nunca a
#   própria resposta). O gabarito **não** é mostrado ao teacher.
# - **Loss** (`distill_mode`):
#   - `"token"`: KL reverso estimado no token amostrado,
#     `stopgrad(log π − log q) · log π`. É barato, só precisa do logp de cada token.
#   - `"topk"`: JSD-α sobre os top-K tokens do student mais um bucket de cauda,
#     igual ao `compute_self_distillation_loss` do repo oficial.
# - Amostras sem irmão bem-sucedido ficam mascaradas (`sd_mask = 0`).
# - `grpo_lambda > 0` liga o híbrido do §4.5 do paper:
#   `λ·GRPO + (1−λ)·SDPO`.
#
# O treino é on-policy (`num_iterations = 1`, um passo por batch), então a razão
# de importância é ≡ 1 e não é preciso o clip de IS do paper.
#
# **Referência (run GRPO do `grpo_gemma.ipynb`, 64 exemplos de teste, greedy):**
#
# | | accuracy | partial | format |
# |---|---|---|---|
# | baseline | 48.4% | 50.0% | 6.25% |
# | GRPO | 48.4% | 54.7% | 82.8% |
#
# Todos os outros hiperparâmetros (LoRA, otimizador, LR, steps, G=2, β) são
# mantidos idênticos ao run GRPO, para isolar o efeito do algoritmo.

# %% [markdown]
# ## Install necessary libraries: RESTART AFTER INSTALL FOR COLAB

# %%
import importlib.util

if importlib.util.find_spec("tunix") is None:
  print("Installing required packages...")
  # %pip install -q dotenv
  # %pip install -q kagglehub
  # %pip install -q ipywidgets
  # %pip install -q tensorflow
  # %pip install -q tensorflow_datasets
  # %pip install -q tensorboard
  # %pip install -q transformers
  # %pip install -q grain
  # tunix exige jax<0.11.1; instalar jax, flax e tunix (SHA fixado) num único
  # comando deixa o pip escolher versões consistentes. O flax HEAD do notebook
  # original puxava jax 0.12 e quebrava o tunix.
  # %pip install -q 'jax[tpu]<0.11.1' flax 'google-tunix @ git+https://github.com/google/tunix@0160f6f0c78c9ec71185338a8641eb7daa4a3199'
  # %pip install git+https://github.com/google/qwix
  # %pip install -q huggingface_hub
  # %pip install -q datasets
  # %pip install -q 'numpy>2'

# %% [markdown]
# ## Logging into services
#
# Also, there is a bug where the code stalls with initializing rl_trainer/grpo_cluster with wandb. Doing this early because it fixes the bug sometimes (but not always - still bugs a lot)

# %%
import os

try:
  from google.colab import userdata

  USE_COLAB = True
  # %pip uninstall -q wandb -y  # wandb is glitchy with tunix in colab
except ImportError:
  USE_COLAB = False


def _secret(name):
  """Lê de os.environ ou dos secrets do Colab. Nunca pede input, para o
  notebook rodar headless (ex.: `colab exec -f sdpo_gemma.ipynb`)."""
  if os.environ.get(name):
    return os.environ[name]
  if USE_COLAB:
    try:
      return userdata.get(name)
    except Exception:  # secret ausente ou sem acesso a partir do colab-cli
      return None
  return None


for name in ("HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "WANDB_API_KEY"):
  if value := _secret(name):
    os.environ[name] = value

if not USE_COLAB:
  from dotenv import load_dotenv

  load_dotenv()
  import nest_asyncio

  nest_asyncio.apply()
  # Only using wandb on TPU VM because it has strange bugs on Colab
  if os.environ.get("WANDB_API_KEY"):
    import wandb

    wandb.login(key=os.environ["WANDB_API_KEY"])

# huggingface_hub lê HF_TOKEN do ambiente (ou ~/.cache/huggingface/token).
print("HF token:", "ok" if os.environ.get("HF_TOKEN") else "não encontrado")

# %% [markdown]
# ## Imports

# %%
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import csv
import functools
import json
from pathlib import Path
from pprint import pprint
import re
import shutil
import sys

from flax import nnx
import grain
from huggingface_hub import snapshot_download
import humanize
import jax
import jax.numpy as jnp
import kagglehub
import numpy as np
import optax
from orbax import checkpoint as ocp
import qwix
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params as gemma_params
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
import dataclasses
from typing import Any

import flax
from tunix.generate import utils as generate_utils
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.sft import utils as sft_utils
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

# %% [markdown]
# ## Hyperparameters
#
# Let's define the configuration we are going to use. Note that this is by no
# means a "perfect" set of hyperparameters. To get good results, you might have
# to train the model for longer.

# %%
# ====== Model ======
MODEL_ID = "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = 0.9

# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
# Adjust mesh based on your TPU memory and model size.
NUM_TPUS = len(jax.devices())
if NUM_TPUS == 8:
  MESH_COUNTS = (1, 4)
elif NUM_TPUS == 1:
  MESH_COUNTS = (1, 1)
else:
  raise ValueError(f"Unsupported number of TPUs: {NUM_TPUS}")

MESH = [
    MESH_COUNTS,
    ("fsdp", "tp"),
]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
# The number of times the policy generates multiple responses for a given prompt
# within a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = 2

# === other GRPO configs ===
# The number of iterations per batch (𝜇 in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = 0.08
# Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = 0.2

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = 1
# Increase `NUM_BATCHES` and `MAX_STEPS` for better results.
NUM_BATCHES = 3738
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 64

EVAL_EVERY_N_STEPS = 64  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = 0.1 * MAX_STEPS
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1

# ====== SDPO ======
# Uma resposta conta como "bem-sucedida" (e pode virar demonstração para os
# irmãos) quando `check_answer` dá a nota máxima: resposta exata no formato
# certo. É um sinal mais limpo que a soma heterogênea das 4 rewards (máx. 10).
SUCCESS_REWARD_FN = "check_answer"
SUCCESS_THRESHOLD = 3.0
# "token": KL reverso no token amostrado (barato). "topk": JSD sobre top-K + cauda.
DISTILL_MODE = "token"
DISTILL_TOPK = 100
# α=0 → KL(teacher‖student), α=1 → KL(student‖teacher); 0.5 é o default do paper.
JSD_ALPHA = 0.5
# θ' ← (1−r)·θ' + r·θ sobre os parâmetros LoRA, a cada passo.
TEACHER_EMA_RATE = 0.05
# 0 = SDPO puro; 0.9 = híbrido λ·GRPO + (1−λ)·SDPO (§4.5 do paper).
GRPO_LAMBDA = 0.0
DONT_REPROMPT_ON_SELF_SUCCESS = True
# Prompt do teacher = template + pergunta + demonstração (até uma geração inteira).
MAX_TEACHER_PROMPT_LENGTH = MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 64

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
CKPT_DIR = "/tmp/content/ckpts_sdpo/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": None, "top_k": 1, "top_p": None},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}


# %%
# Smoke test (SDPO_SMOKE_TEST=1): poucos passos, só para validar a integração.
# O sucesso vira "número certo" (check_numbers), porque o modelo base quase
# nunca acerta o formato exato (6% no baseline) e, com 5 passos, o caminho do
# teacher poderia nem ser exercitado.
SMOKE_TEST = os.environ.get("SDPO_SMOKE_TEST") == "1"
if SMOKE_TEST:
  NUM_BATCHES = 20
  NUM_TEST_BATCHES = 4
  MAX_STEPS = 5
  WARMUP_STEPS = 1
  EVAL_EVERY_N_STEPS = 1000
  SUCCESS_REWARD_FN = "check_numbers"
  SUCCESS_THRESHOLD = 1.5
  CKPT_DIR = "/tmp/content/ckpts_sdpo_smoke/"

# %% [markdown]
# ## Utility functions

# %%
def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")


# %% [markdown]
# ## Data preprocessing
#
# First, let's define some special tokens. We instruct the model to first reason
# between the `<reasoning>` and `</reasoning>` tokens. After
# reasoning, we expect it to provide the answer between the `<answer>` and
# `</answer>` tokens.

# %%
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = f"""You are given a problem. First, think about the problem \
and provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
"""

# Prompt do teacher, adaptado de `verl/trainer/config/actor/actor.yaml` do SDPO:
# a mesma pergunta, seguida de uma solução correta de um irmão do grupo.
TEACHER_QUESTION_TEMPLATE = """{question}
Correct solution:

{demonstration}


Correctly solve the original question."""


def build_teacher_prompt(question: str, demonstration: str) -> str:
  return TEMPLATE.format(
      system_prompt=SYSTEM_PROMPT,
      question=TEACHER_QUESTION_TEMPLATE.format(
          question=question, demonstration=demonstration.strip()
      ),
  )


# %% [markdown]
# We use OpenAI's [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k), which comprises grade school math word problems.

# %%
def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def _load_from_tfds(data_dir: str, split: str):
  import tensorflow_datasets.text.gsm8k

  return tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )


def download_kaggle_dataset(target_dir="./data/gsm8k"):
  os.makedirs(target_dir, exist_ok=True)
  src = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
  src = Path(src)
  dst = Path(target_dir)

  for csv_file in src.glob("*.csv"):  # match all CSV files
    shutil.copy2(csv_file, dst / csv_file.name)
    print(f"Copied {csv_file.name} → {dst/csv_file.name}")
  return target_dir


def get_dataset(data_dir, split="train", source="tfds") -> grain.MapDataset:
  # Download data
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  if source == "tfds":
    import tensorflow_datasets.text.gsm8k

    data = tfds.data_source(
        "gsm8k",
        split=split,
        data_dir=data_dir,
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )

  elif source == "kaggle":
    kaggle_dir = download_kaggle_dataset(data_dir)
    file_name = "main_" + split + ".csv"
    csv_path = os.path.join(kaggle_dir, file_name)  # adjust filename if needed

    data = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        data.append({
            "question": row["question"],
            "answer": row["answer"],
        })

  else:
    raise ValueError(f"Unknown source: {source}")

  def _as_text(v):
    return v if isinstance(v, str) else v.decode("utf-8")

  dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=42)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": TEMPLATE.format(
                  system_prompt=SYSTEM_PROMPT,
                  question=_as_text(x["question"]),
              ),
              # passed to reward functions
              "question": _as_text(x["question"]),
              # passed to reward functions
              "answer": extract_hash_answer(_as_text(x["answer"])),
          }
      )
  )
  return dataset


# %% [markdown]
# We split the dataset set into train and test sets as usual.

# %%
# "kaggle" exige KAGGLE_USERNAME/KAGGLE_KEY; sem input() para rodar headless.
source = os.environ.get("GSM8K_SOURCE", "tfds").strip().lower()

if source not in ("tfds", "kaggle"):
  print("Invalid choice. Defaulting to 'tfds'.")
  source = "tfds"

print(f"Using data source: {source}")

dataset = get_dataset(TRAIN_DATA_DIR, "train", source).batch(
    TRAIN_MICRO_BATCH_SIZE
)[:NUM_BATCHES]

if TRAIN_FRACTION == 1.0:
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)

  val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_DIR, "test", source).batch(
    TRAIN_MICRO_BATCH_SIZE
)[:NUM_TEST_BATCHES]

dataset_lengths = (
    len(train_dataset),
    len(val_dataset) if val_dataset is not None else 0,
    len(test_dataset),
)
print(f"dataset contains {dataset_lengths} of batches")

# %% [markdown]
# Let's see how one batch of the training dataset looks like!
#

# %%
for ele in train_dataset[:1]:
  pprint(ele)

# %% [markdown]
# ## Load the policy model and the reference model
#
# The policy model is the model which is actually trained and whose weights are
# updated. The reference model is the model with which we compute KL divergence.
# This is to ensure that the policy updates are not huge and that it does not
# deviate too much from the reference model.
#
# Typically, the reference model is the base model, and the policy model is the
# same base model, but with LoRA parameters. Only the LoRA parameters are updated.
#
# Note: We perform full precision (fp32) training. You can, however, leverage
# Qwix for QAT.
#
# To load the model, you need to be on [Kaggle](https://www.kaggle.com/) and need
# to have agreed to the Gemma license
# [here](https://www.kaggle.com/models/google/gemma/flax/).

# %%
ignore_patterns = [
    "*.pth",  # Ignore PyTorch .pth weight files
]
print(f"Downloading {MODEL_ID} from Hugging Face...")
local_model_path = snapshot_download(
    repo_id=MODEL_ID, ignore_patterns=ignore_patterns
)
print(f"Model successfully downloaded to: {local_model_path}")

EOS_TOKENS = []
generation_config_path = os.path.join(
    local_model_path, "generation_config.json"
)
if os.path.exists(generation_config_path):
  with open(generation_config_path, "r") as f:
    generation_configs = json.load(f)
  EOS_TOKENS = generation_configs.get("eos_token_id", [])
  print(f"Using EOS token IDs: {EOS_TOKENS}")

# %% [markdown]
# This code snippet serves as a workaround to re-save the pre-trained model checkpoint from Kaggle into a local format that is compatible with the [Flax NNX](https://flax.readthedocs.io/en/stable/why.html) library. Because the original checkpoint has parameter names and tensor structures that don't match the target NNX model architecture, it cannot be loaded directly.
#
# We first load the original weights into a temporary model instance, then extract and re-save the model's state into a new, properly formatted local checkpoint, which can then be successfully loaded by the final sharded NNX model.

# %%
MODEL_CP_PATH = local_model_path

model_config = None
if "gemma-3-270m" in MODEL_ID:
  model_config = gemma_lib.ModelConfig.gemma3_270m()
elif "gemma-3-1b" in MODEL_ID:
  model_config = gemma_lib.ModelConfig.gemma3_1b_it()
else:
  raise ValueError(f"Unknown model id: {MODEL_ID}")

mesh = jax.make_mesh(
    *MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0])
)
with jax.set_mesh(mesh):
  gemma3 = params_safetensors_lib.create_model_from_safe_tensors(
      MODEL_CP_PATH, (model_config), mesh
  )
  nnx.display(gemma3)


# %% [markdown]
# ### LoRA Application
#
# The `get_lora_model` function takes the base model and applies LoRA layers to it. It uses a `LoraProvider` to select specific layers (like attention and MLP layers) to be adapted. The resulting LoRA-infused model is then sharded and updated to ensure it's ready for distributed training.

# %%
def get_lora_model(base_model, mesh):
  lora_provider = qwix.LoraProvider(
      module_path=(
          ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
          ".*attn_vec_einsum"
      ),
      rank=RANK,
      alpha=ALPHA,
  )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  with jax.set_mesh(mesh):
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model


# %% [markdown]
# Now we load reference and policy Gemma models using the Flax NNX library and display their structures.

# %%
# Policy model
lora_policy = get_lora_model(gemma3, mesh=mesh)
nnx.display(lora_policy)

# %%
tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
if tokenizer.eos_id() not in EOS_TOKENS:
  EOS_TOKENS.append(tokenizer.eos_id())
  print(f"Using EOS token IDs: {EOS_TOKENS}")

# %% [markdown]
# ## Define reward functions
#
# We define four reward functions:
#
# - reward if the format of the output exactly matches the instruction given in
# `TEMPLATE`;
# - reward if the format of the output approximately matches the instruction given
# in `TEMPLATE`;
# - reward if the answer is correct/partially correct;
# - Sometimes, the text between `<answer>`, `</answer>` might not be one
#   number. So, we extract the number, and reward the model if the answer is correct.
#
# The reward functions are inspired from
# [here](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb).
#
# First off, let's define a RegEx for checking whether the format matches.

# %%
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

example = (
    f"{reasoning_start}Let me"
    f" think!{reasoning_end}{solution_start}2{solution_end}"
)
print("Example:", example)
print("Match:\t", match_format.search(example))


# %% [markdown]
# Give the model a reward of 3 points if the format matches exactly.

# %%
def match_format_exactly(prompts, completions, **kwargs):
  return [
      0 if match_format.search(response) is None else 3.0
      for response in completions
  ]


# %% [markdown]
# We also reward the model if the format of the output matches partially.

# %%
def match_format_approximately(prompts, completions, **kwargs):
  scores = []

  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.find(reasoning_start) == 0 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores


# %% [markdown]
# Reward the model if the answer is correct. A reward is also given if the answer
# does not match exactly, i.e., based on how close the answer is to the correct
# value.

# %%
def check_answer(prompts, completions, answer, **kwargs):
  responses = completions

  extracted_responses = [
      guess.group(1)
      if r is not None and (guess := match_format.search(r)) is not None
      else None
      for r in responses
  ]

  scores = []
  assert len(extracted_responses) == len(
      answer
  ), f"{extracted_responses} and {answer} have mismatching length"
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue
    # Correct answer gets 3 points!
    if guess == true_answer:
      score += 3.0
    # Match if spaces are seen
    elif guess.strip() == true_answer.strip():
      score += 1.5
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        ratio = float(guess) / float(true_answer)
        if ratio >= 0.9 and ratio <= 1.1:
          score += 0.5
        elif ratio >= 0.8 and ratio <= 1.2:
          score += 0.25
        else:
          score -= 1.0  # Penalize wrong answers
      except:
        score -= 0.5  # Penalize
    scores.append(score)
  return scores


# %% [markdown]
# Sometimes, the text between `<answer>` and `</answer>` might not be one
# number; it can be a sentence. So, we extract the number and compare the answer.

# %%
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)
match_numbers.findall(f"{solution_start}  0.34  {solution_end}")


# %%
def check_numbers(prompts, completions, answer, **kwargs):
  question = kwargs["question"]
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  print("START ============================")
  print(f"Question:\t{question[0]}")
  print(f"Answer:\t{answer[0]}")
  print(f"Response:\t{responses[0]}")
  print(f"Extracted:\t{extracted_responses[0]}")
  print("END ==============================")
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except:
      scores.append(0)
      continue
  return scores


# %% [markdown]
# ## SDPO Learner
#
# Duas classes, no mesmo formato de `GRPOConfig` / `GRPOLearner`:
#
# - `SDPOConfig(GRPOConfig)`: os hiperparâmetros do SDPO.
# - `SDPOLearner(GRPOLearner)`: reaproveita todo o rollout do GRPO e, depois dele:
#   1. decide quais respostas foram bem-sucedidas;
#   2. monta o prompt do teacher com a demonstração de um irmão;
#   3. roda o teacher (EMA dos parâmetros LoRA, sem gradiente) sobre os tokens
#      da resposta do student.
#
# A loss `"sdpo"` é registrada no `function_registry` do Tunix, que é como o
# `GRPOLearner` encontra a loss a partir de `algo_config.policy_loss_fn`.
#
# Primeiro, as funções puras: a escolha da demonstração e a JSD top-K.

# %%
def select_demonstrations(
    success: np.ndarray,
    num_generations: int,
    dont_reprompt_on_self_success: bool = True,
) -> np.ndarray:
  """Índice global do primeiro irmão bem-sucedido de cada amostra, ou -1."""
  grouped = np.asarray(success, dtype=bool).reshape(-1, num_generations)
  demo_idx = np.full(grouped.shape, -1, dtype=np.int64)
  for g, row in enumerate(grouped):
    for i in range(num_generations):
      for j in np.flatnonzero(row):
        if j != i or not dont_reprompt_on_self_success:
          demo_idx[g, i] = g * num_generations + j
          break
  return demo_idx.reshape(-1)


def _add_tail(topk_logps: jax.Array) -> jax.Array:
  """Acrescenta o bucket log(1 − Σ p_topk) para fechar a distribuição."""
  log_mass = jax.nn.logsumexp(topk_logps, axis=-1, keepdims=True)
  tail = jnp.log(jnp.clip(-jnp.expm1(jnp.minimum(log_mass, 0.0)), min=1e-10))
  return jnp.concatenate([topk_logps, tail], axis=-1)


def _kl(log_p: jax.Array, log_q: jax.Array) -> jax.Array:
  return jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1)


def topk_jsd(
    student_topk_logps: jax.Array,
    teacher_topk_logps: jax.Array,
    alpha: float,
) -> jax.Array:
  """Divergência por token sobre top-K + cauda (`compute_self_distillation_loss`).

  Os log-probs `[..., K]` já devem estar normalizados sobre o vocabulário
  inteiro. α=0 → KL(teacher‖student), α=1 → KL(student‖teacher); entre os dois,
  JSD generalizada com mistura M = (1−α)·p_s + α·p_t.
  """
  log_s = _add_tail(student_topk_logps)
  log_t = _add_tail(teacher_topk_logps)
  if alpha == 0.0:
    return _kl(log_t, log_s)
  if alpha == 1.0:
    return _kl(log_s, log_t)
  log_m = jnp.logaddexp(log_s + jnp.log(1 - alpha), log_t + jnp.log(alpha))
  return (1 - alpha) * _kl(log_s, log_m) + alpha * _kl(log_t, log_m)


# %% [markdown]
# Em seguida, os forwards. `completion_logits` devolve só os logits das posições
# da resposta. Ele usa `skip_lm_head` para não materializar os logits do prompt
# do teacher, que pode ter ~1k tokens contra 262k de vocabulário.

# %%
def completion_logits(
    model, prompt_tokens, completion_tokens, pad_id, eos_id, temperature
):
  input_tokens, positions, attn_mask, _ = common.process_ids(
      prompt_tokens, completion_tokens, pad_id, eos_id
  )
  hidden, _ = model(
      input_tokens,
      positions=positions,
      cache=None,
      attention_mask=attn_mask,
      skip_lm_head=True,
  )
  num_completion = completion_tokens.shape[1]
  logits = model.compute_final_logits(hidden[:, -num_completion - 1 : -1, :])
  if temperature not in (0.0, 1.0):
    logits = logits / temperature
  return logits.astype(jnp.float32)


@functools.partial(
    jax.jit, static_argnames=("pad_id", "eos_id", "temperature", "topk")
)
def _student_topk_idx(
    graphdef, lora_state, rest_state, prompt_tokens, completion_tokens,
    pad_id, eos_id, temperature, topk,
):
  model = nnx.merge(graphdef, lora_state, rest_state)
  logits = completion_logits(
      model, prompt_tokens, completion_tokens, pad_id, eos_id, temperature
  )
  return jax.lax.top_k(logits, topk)[1]


@functools.partial(jax.jit, static_argnames=("pad_id", "eos_id", "temperature"))
def _teacher_logps(
    graphdef, lora_state, rest_state, prompt_tokens, completion_tokens,
    pad_id, eos_id, temperature, topk_idx=None,
):
  """Log-probs do teacher no token amostrado e, opcionalmente, em `topk_idx`."""
  model = nnx.merge(graphdef, lora_state, rest_state)
  log_probs = jax.nn.log_softmax(
      completion_logits(
          model, prompt_tokens, completion_tokens, pad_id, eos_id, temperature
      ),
      axis=-1,
  )
  token_logps = jnp.take_along_axis(
      log_probs, completion_tokens[..., None], axis=-1
  )[..., 0]
  topk_logps = (
      None
      if topk_idx is None
      else jnp.take_along_axis(log_probs, topk_idx, axis=-1)
  )
  return token_logps, topk_logps


# %% [markdown]
# A loss. Com `num_iterations = 1` o treino é on-policy, então o termo GRPO
# opcional se reduz a `−A · exp(logπ − sg(logπ))`; o clip nunca é ativado.
#
# Diferente do notebook GRPO, os dois termos são agregados como média por token
# (*token-mean*), como no SDPO. O GRPO de referência usava
# `sequence-mean-token-mean`.

# %%
@flax.struct.dataclass(frozen=True)
class SDPOTrainExample(common.TrainExample):
  sd_mask: common.ArrayType | None = None  # [B] 1 = tem demonstração
  teacher_per_token_logps: common.ArrayType | None = None  # [B, C]
  topk_idx: common.ArrayType | None = None  # [B, C, K]
  teacher_topk_logps: common.ArrayType | None = None  # [B, C, K]


@function_registry.register_policy_loss_fn("sdpo")
def sdpo_loss_fn(
    model, train_example, algo_config, pad_id, eos_id, **kwargs
) -> sft_utils.LossOutput:
  completion_ids = train_example.completion_ids
  completion_mask = train_example.completion_mask.astype(jnp.float32)

  log_probs = jax.nn.log_softmax(
      completion_logits(
          model,
          train_example.prompt_ids,
          completion_ids,
          pad_id,
          eos_id,
          algo_config.temperature,
      ),
      axis=-1,
  )
  per_token_logps = jnp.take_along_axis(
      log_probs, completion_ids[..., None], axis=-1
  )[..., 0]

  def masked_mean(x, mask):
    return jnp.sum(jnp.where(mask > 0, x, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0)

  # --- self-distillation ---
  sd_mask = completion_mask * train_example.sd_mask[:, None]
  if algo_config.distill_mode == "token":
    teacher_logps = train_example.teacher_per_token_logps
    log_ratio = per_token_logps - teacher_logps
    per_token_distill = jax.lax.stop_gradient(log_ratio) * per_token_logps
    distill_value = masked_mean(log_ratio, sd_mask)
  else:
    student_topk = jnp.take_along_axis(
        log_probs, train_example.topk_idx, axis=-1
    )
    per_token_distill = topk_jsd(
        student_topk, train_example.teacher_topk_logps, algo_config.jsd_alpha
    )
    distill_value = masked_mean(per_token_distill, sd_mask)
    teacher_logps = jnp.zeros_like(per_token_logps)
  distill_loss = masked_mean(per_token_distill, sd_mask)

  # --- GRPO (opcional, híbrido do §4.5) ---
  lam = algo_config.grpo_lambda
  adv = train_example.advantages.astype(jnp.float32)[:, None]
  is_ratio = jnp.exp(per_token_logps - jax.lax.stop_gradient(per_token_logps))
  pg_loss = masked_mean(-adv * is_ratio, completion_mask)

  loss = lam * pg_loss + (1.0 - lam) * distill_loss

  aux = {
      "kl": sft_utils.WeightedMetric(jnp.array(0.0), jnp.array(1.0)),
      "sdpo/distill_loss": distill_loss,
      "sdpo/distill_value": distill_value,
      "sdpo/pg_loss": pg_loss,
      "sdpo/active_token_frac": jnp.sum(sd_mask)
      / jnp.maximum(jnp.sum(completion_mask), 1.0),
      "sdpo/student_logp": masked_mean(per_token_logps, sd_mask),
      "sdpo/teacher_logp": masked_mean(teacher_logps, sd_mask),
      "entropy": masked_mean(
          -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1), completion_mask
      ),
  }
  if train_example.ref_per_token_logps is not None:
    kl = common.compute_kl_divergence(
        per_token_logps,
        train_example.ref_per_token_logps,
        algo_config.kl_loss_mode,
        clamp_value=algo_config.kl_clamp_value,
    )
    kl_loss = masked_mean(kl, completion_mask)
    aux["kl"] = sft_utils.WeightedMetric(
        jnp.sum(jnp.where(completion_mask > 0, kl, 0.0)),
        jnp.sum(completion_mask),
        min_denom=1.0,
    )
    if algo_config.beta:
      loss = loss + algo_config.beta * kl_loss

  return sft_utils.LossOutput(
      primary_loss=sft_utils.WeightedMetric(loss, jnp.array(1.0)),
      aux_metrics=aux,
  )


# %% [markdown]
# As duas classes.

# %%
@dataclasses.dataclass(kw_only=True)
class SDPOConfig(GRPOConfig):
  """Configuração do SDPO (https://arxiv.org/abs/2601.20802).

  Attributes:
    success_reward_fn: nome da reward fn que define uma resposta bem-sucedida.
    success_threshold: score mínimo dessa reward fn para contar como sucesso.
    distill_mode: "token" (KL reverso no token amostrado) ou "topk" (JSD top-K).
    distill_topk: K do modo "topk".
    jsd_alpha: 0 → KL(teacher‖student), 1 → KL(student‖teacher).
    teacher_ema_rate: r em θ' ← (1−r)·θ' + r·θ (parâmetros LoRA).
    grpo_lambda: peso do termo GRPO; 0 = SDPO puro.
    dont_reprompt_on_self_success: nunca usar a própria resposta como demo.
    max_teacher_prompt_length: largura (left-padded) do prompt do teacher.
  """

  policy_loss_fn: str = "sdpo"
  success_reward_fn: str = "check_answer"
  success_threshold: float = 3.0
  distill_mode: str = "token"
  distill_topk: int = 100
  jsd_alpha: float = 0.5
  teacher_ema_rate: float = 0.05
  grpo_lambda: float = 0.0
  dont_reprompt_on_self_success: bool = True
  max_teacher_prompt_length: int = 1088

  def __post_init__(self):
    super().__post_init__()
    if self.distill_mode not in ("token", "topk"):
      raise ValueError(f"distill_mode inválido: {self.distill_mode!r}")
    if self.num_iterations != 1:
      raise ValueError("SDPOLearner assume treino on-policy (num_iterations=1).")
    if not 0.0 <= self.grpo_lambda <= 1.0:
      raise ValueError("grpo_lambda deve estar em [0, 1].")
    if not 0.0 < self.teacher_ema_rate <= 1.0:
      raise ValueError("teacher_ema_rate deve estar em (0, 1].")
    if not 0.0 <= self.jsd_alpha <= 1.0:
      raise ValueError("jsd_alpha deve estar em [0, 1].")


class SDPOLearner(GRPOLearner):
  """GRPOLearner cujo passo de política é a auto-destilação do SDPO."""

  def __init__(
      self,
      rl_engine,
      algo_config: SDPOConfig,
      reward_fns,
      metric_fns=None,
      data_shuffle_seed=None,
  ):
    super().__init__(
        rl_engine=rl_engine,
        algo_config=algo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )
    reward_fns = reward_fns if isinstance(reward_fns, list) else [reward_fns]
    by_name = {fn.__name__: fn for fn in reward_fns}
    if algo_config.success_reward_fn not in by_name:
      raise ValueError(
          f"{algo_config.success_reward_fn!r} não está em reward_fns:"
          f" {list(by_name)}"
      )
    self._success_fn = by_name[algo_config.success_reward_fn]

    # Teacher = mesmos pesos base + cópia EMA dos parâmetros LoRA do actor.
    actor = self.rl_engine.actor_trainer.model
    self._teacher_lora = jax.tree.map(jnp.copy, nnx.state(actor, nnx.LoRAParam))

    self.rl_engine.actor_trainer.with_rl_metrics_to_log({
        "kl": np.mean,
        "entropy": np.mean,
        "sdpo/distill_loss": np.mean,
        "sdpo/distill_value": np.mean,
        "sdpo/pg_loss": np.mean,
        "sdpo/active_token_frac": np.mean,
        "sdpo/student_logp": np.mean,
        "sdpo/teacher_logp": np.mean,
    })

  def _update_teacher_ema(self):
    r = self.algo_config.teacher_ema_rate
    actor_lora = nnx.state(self.rl_engine.actor_trainer.model, nnx.LoRAParam)
    self._teacher_lora = jax.tree.map(
        lambda t, s: (1.0 - r) * t + r * s, self._teacher_lora, actor_lora
    )

  def _encode_prompt(self, text: str, student_prompt: np.ndarray) -> np.ndarray:
    """Tokeniza como o rollout: prefixa BOS se o prompt do student o tiver."""
    tokenizer = self.rl_engine.tokenizer
    ids = list(tokenizer.encode(text))
    bos = tokenizer.bos_id()
    if student_prompt[0] == bos and (not ids or ids[0] != bos):
      ids = [bos] + ids
    return np.asarray(ids, dtype=np.int32)

  def _decode_completions(self, example) -> list[str]:
    stop = {self.rl_engine.rollout.pad_id(), self.rl_engine.rollout.eos_id()}
    rollout_config = self.rl_engine.cluster_config.rollout_config
    if isinstance(rollout_config, dict):
      rollout_config = rollout_config[rl_engine_lib.Mode.TRAIN]
    stop |= set(rollout_config.eos_tokens or [])
    completion_ids = np.asarray(example.completion_ids)
    return [
        self.rl_engine.tokenizer.decode([int(t) for t in row if int(t) not in stop])
        for row in completion_ids
    ]

  def _generate_and_compute_advantage(
      self,
      training_input,
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
  ) -> SDPOTrainExample:
    cfg = self.algo_config
    if mode == rl_engine_lib.Mode.TRAIN:
      # Equivale a atualizar o EMA logo após o passo anterior do otimizador.
      self._update_teacher_ema()

    example = super()._generate_and_compute_advantage(training_input, mode)

    def _as_text(v):
      return v if isinstance(v, str) else v.decode("utf-8")

    questions = [_as_text(q) for q in training_input["question"]]
    answers = [_as_text(a) for a in training_input["answer"]]
    completions = self._decode_completions(example)
    scores = self._success_fn(
        prompts=training_input["prompts"],
        completions=completions,
        answer=answers,
        question=questions,
    )
    success = np.asarray(scores, dtype=np.float32) >= cfg.success_threshold
    demo_idx = select_demonstrations(
        success, cfg.num_generations, cfg.dont_reprompt_on_self_success
    )

    pad_id = self.rl_engine.rollout.pad_id()
    eos_id = self.rl_engine.rollout.eos_id()
    width = cfg.max_teacher_prompt_length
    student_prompts = np.asarray(example.prompt_ids)
    batch_size = len(completions)
    sd_mask = np.zeros(batch_size, dtype=np.float32)
    num_truncated = 0
    teacher_rows = []
    for i in range(batch_size):
      student_prompt = student_prompts[i][student_prompts[i] != pad_id]
      row = student_prompt  # preenchimento para amostras sem demonstração
      if demo_idx[i] >= 0:
        ids = self._encode_prompt(
            build_teacher_prompt(questions[i], completions[demo_idx[i]]),
            student_prompt,
        )
        if len(ids) <= width:
          row, sd_mask[i] = ids, 1.0
        else:
          num_truncated += 1
      teacher_rows.append(
          generate_utils.pad_to_length(
              row, target_length=width, pad_value=pad_id, left=True
          )
      )
    teacher_prompt_ids = jnp.asarray(np.stack(teacher_rows))

    completion_shape = example.completion_ids.shape
    teacher_per_token_logps = jnp.zeros(completion_shape, jnp.float32)
    topk_shape = (*completion_shape, cfg.distill_topk)
    topk_idx = teacher_topk_logps = None
    if cfg.distill_mode == "topk":
      topk_idx = jnp.zeros(topk_shape, jnp.int32)
      teacher_topk_logps = jnp.zeros(topk_shape, jnp.float32)

    # Sem nenhuma demonstração no batch, a loss de destilação é 0: pula o teacher.
    if sd_mask.any():
      actor = self.rl_engine.actor_trainer.model
      graphdef, student_lora, rest = nnx.split(actor, nnx.LoRAParam, ...)
      static = dict(pad_id=pad_id, eos_id=eos_id, temperature=cfg.temperature)
      with jax.set_mesh(self.rl_engine.r2m[rl_engine_lib.Role.ACTOR]):
        if cfg.distill_mode == "topk":
          topk_idx = _student_topk_idx(
              graphdef, student_lora, rest, example.prompt_ids,
              example.completion_ids, topk=cfg.distill_topk, **static,
          )
        teacher_per_token_logps, teacher_topk_logps = _teacher_logps(
            graphdef, self._teacher_lora, rest, teacher_prompt_ids,
            example.completion_ids, topk_idx=topk_idx, **static,
        )

    grouped = success.reshape(-1, cfg.num_generations)
    self.rl_engine.buffer_metrics(
        {
            "sdpo/success_sample_fraction": (success.mean(), np.mean),
            "sdpo/success_group_fraction": (grouped.any(-1).mean(), np.mean),
            "sdpo/reprompt_sample_fraction": (sd_mask.mean(), np.mean),
            "sdpo/teacher_prompt_too_long": (num_truncated, np.sum),
        },
        mode=mode,
    )

    fields = {
        f.name: getattr(example, f.name) for f in dataclasses.fields(example)
    }
    return SDPOTrainExample(
        **fields,
        sd_mask=jnp.asarray(sd_mask),
        teacher_per_token_logps=teacher_per_token_logps,
        topk_idx=topk_idx,
        teacher_topk_logps=teacher_topk_logps,
    )


# %% [markdown]
# ## Evaluate
#
#
# Before we train the model, let's evaluate the model on the test set so we can
# see the improvement post training.
#
# We evaluate it in two ways:
#
# **Quantitative**
#
# * **Answer Accuracy**: percentage of samples for which the model predicts the
# correct final numerical answer  
# * **Answer (Partial) Accuracy**: percentage of samples for which the model
# predicts a final numerical answer such that the \`model answer / answer\`
# ratio lies between 0.9 and 1.1.  
# * **Format Accuracy**: percentage of samples for which the model outputs the
# correct format, i.e., reasoning between the reasoning special tokens, and the
# final answer between the \`\<start\_answer\>\`, \`\<end\_answer\>\` tokens.
#
# **Qualitative**
#
# We'll also print outputs for a few given questions so that we can compare the generated output later.
#

# %% [markdown]
# We define a helper function to generate an answer, given a prompt.

# %%
def generate(
    question, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None
):
  """Given prompt, generates text."""

  if isinstance(question, str):
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=question,
        ),
    ]
  else:
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=q,
        )
        for q in question
    ]

  out_data = sampler(
      input_strings=input_batch,
      max_generation_steps=768,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      echo=False,
      seed=seed if seed is not None else None,
      eos_tokens=EOS_TOKENS,
  )

  output = out_data.text
  if isinstance(question, str):
    return output[0]
  return output


# %% [markdown]
# Another helper function for evaluation.

# %%
def evaluate(
    dataset,
    sampler,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    corr_lst=False,
    make_lst=False,
):
  """Computes accuracy and percentage of outputs matching the format."""

  response_lst = []
  corr = 0
  partially_corr = 0
  corr_format = 0
  total = 0

  for batch in tqdm(dataset):
    answers = batch["answer"]
    questions = batch["question"]

    multiple_call_responses = [[] for _ in range(len(questions))]
    for p in range(num_passes):
      responses = generate(
          questions, sampler, temperature, top_k, top_p, seed=p
      )
      for idx, response in enumerate(responses):
        multiple_call_responses[idx].append(response)
        print(f"Question:\t{questions[idx]}")
        print(f"Correct Answer:\t{answers[idx]}")
        print(f"Response:\t{response}")
        print("-" * 50)

    for question, multiple_call_response, answer in zip(
        questions, multiple_call_responses, answers
    ):
      # check answer
      corr_ctr_per_question = 0
      partially_corr_per_question = 0
      corr_format_per_question = 0
      for response in multiple_call_response:
        extracted_response = (
            guess.group(1)
            if (guess := match_numbers.search(response)) is not None
            else "-1000000"
        )
        try:
          if float(extracted_response.strip()) == float(answer.strip()):
            corr_ctr_per_question += 1

          ratio = float(extracted_response.strip()) / float(answer.strip())
          if ratio >= 0.9 and ratio <= 1.1:
            partially_corr_per_question += 1
        except:
          print("SKIPPED")

        # check format
        if match_format.search(response) is not None:
          corr_format_per_question += 1

        if (
            corr_ctr_per_question > 0
            and partially_corr_per_question > 0
            and corr_format_per_question > 0
        ):
          break

      if corr_ctr_per_question > 0:
        corr += 1
        if corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      else:
        if not corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      if partially_corr_per_question > 0:
        partially_corr += 1
      if corr_format_per_question > 0:
        corr_format += 1

      total += 1
      if total % 10 == 0:
        print(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )

  to_return = (
      corr,
      total,
      corr / total * 100,
      partially_corr / total * 100,
      corr_format / total * 100,
  )
  if make_lst:
    return to_return, response_lst
  return to_return


# %% [markdown]
# Além das métricas do notebook de referência, medimos a **reward do ambiente**:
# a média, por exemplo de teste, de cada uma das 4 reward fns usadas no treino e
# da soma delas (máx. 10). Também medimos o comprimento médio da resposta em
# tokens; o paper do SDPO reporta respostas ~3× mais curtas que as do GRPO.

# %%
REWARD_FNS = [
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
]


def evaluate_env_reward(
    dataset, sampler, temperature=None, top_k=1, top_p=None, seed=None
):
  """Média das reward fns de treino e comprimento das respostas no `dataset`."""
  import contextlib
  import io

  totals = {fn.__name__: 0.0 for fn in REWARD_FNS}
  total_reward = total_tokens = n = 0
  for batch in tqdm(dataset):
    questions = list(batch["question"])
    answers = list(batch["answer"])
    responses = generate(questions, sampler, temperature, top_k, top_p, seed)
    prompts = [
        TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=q)
        for q in questions
    ]
    per_sample = np.zeros(len(responses))
    for fn in REWARD_FNS:
      with contextlib.redirect_stdout(io.StringIO()):  # check_numbers imprime
        scores = np.asarray(
            fn(prompts, responses, answer=answers, question=questions),
            dtype=np.float64,
        )
      totals[fn.__name__] += scores.sum()
      per_sample += scores
    total_reward += per_sample.sum()
    total_tokens += sum(len(tokenizer.encode(r)) for r in responses)
    n += len(responses)

  metrics = {f"reward/{k}": v / n for k, v in totals.items()}
  metrics["reward/total"] = total_reward / n
  metrics["response_length_tokens"] = total_tokens / n
  return metrics


# %% [markdown]
# Now let's see how the original model does on the test set. You can see the percentages of the mode outputs that are fully correct, partially correct and just correct in format. The following step might take couple of minutes to finish.

# %%
# The evaluation might take up to couple of minutes to finish.
RESULTS = {}
STAGE = "baseline"

sampler = sampler_lib.Sampler(
    transformer=lora_policy,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

num_correct, total, accuracy, partial_accuracy, format_accuracy = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{num_correct=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
    f" {format_accuracy=}%"
)
env_metrics = evaluate_env_reward(
    test_dataset, sampler, **GENERATION_CONFIGS["greedy"]
)
pprint(env_metrics)
RESULTS[STAGE] = {
    "accuracy": accuracy,
    "partial_accuracy": partial_accuracy,
    "format_accuracy": format_accuracy,
    **env_metrics,
}

# %% [markdown]
# ## Train
#
# Let's set up all the configs first - checkpointing, metric logging and training.
# We then train the model.

# %%
# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/content/tmp/tensorboard/sdpo", flush_every_n_steps=20
)

# %%
# Logs
# %load_ext tensorboard
# %tensorboard --logdir /tmp/content/tmp/tensorboard/sdpo --port=0

# %%
# Optimizer, learning rate scheduler, gradient clipping
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)
if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )

# %%
# Training config
cluster_config = rl_engine_lib.ClusterConfig(
    role_to_mesh={
        rl_engine_lib.Role.ACTOR: mesh,
        rl_engine_lib.Role.REFERENCE: mesh,
        rl_engine_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine='vanilla',
    offload_to_cpu=False,
    training_config=rl_engine_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        eos_tokens=EOS_TOKENS,
    ),
)

sdpo_config = SDPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
    success_reward_fn=SUCCESS_REWARD_FN,
    success_threshold=SUCCESS_THRESHOLD,
    distill_mode=DISTILL_MODE,
    distill_topk=DISTILL_TOPK,
    jsd_alpha=JSD_ALPHA,
    teacher_ema_rate=TEACHER_EMA_RATE,
    grpo_lambda=GRPO_LAMBDA,
    dont_reprompt_on_self_success=DONT_REPROMPT_ON_SELF_SUCCESS,
    max_teacher_prompt_length=MAX_TEACHER_PROMPT_LENGTH,
)

# %% [markdown]
# ### Setting Up the SDPO Trainer
#
# Now we initialize our system for training. First, we create an `RLEngine` instance, which brings together the **policy model (`actor`)**, a **reference model (`reference`)**, and a **tokenizer**. Our `actor` is a trainable LoRA model, while the `reference` is a fixed base model that we use to guide the training.
#
# We then create an `SDPOLearner`. It uses the **reward functions** only to decide which rollouts succeeded (`check_answer`) and pick demonstrations for the teacher. The other reward functions are still computed and logged, for comparison with the GRPO run. All metrics under `sdpo/*` show up in Tensorboard alongside the GRPO ones (`rewards/*`, `completions/*`).
#
# Tunix trainers are integrated with [Weights & Biases](https://wandb.ai/) and Tensorboard to help you visualize the training progress. You can choose how you want to use it:
#
# **Colab**: Unfortunately, WandB doesn't work nicely with Colab so we disabled it, instead use the tensorboards rendered above to monitor.
#
# **Option 1 (Type 1)**: If you're running a quick experiment or just testing things out, choose this. It creates a temporary, private dashboard right in your browser without requiring you to log in or create an account.
#
# **Option 2 (Type 2)**: If you have an existing W&B account and want to save your project's history to your personal dashboard, choose this. You'll be prompted to enter your API key or log in.

# %%
# RL cluster
rl_engine = rl_engine_lib.RLEngine(
    actor=lora_policy,
    reference=gemma3,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

# SDPO Trainer
sdpo_trainer = SDPOLearner(
    rl_engine=rl_engine,
    reward_fns=REWARD_FNS,
    algo_config=sdpo_config,
)

# %% [markdown]
# ### Training
#
# The first couple of training step might take up to 5 minutes to finish. Please be patient. If you experience long training steps, e.g. >10 minutes per step, please open a bug. Really appreciated!

# %%
sdpo_trainer.train(train_dataset, val_dataset)

# %% [markdown]
# ## Final Evaluation
#
# Let's evaluate our finetuned model!

# %%
# Load checkpoint first.

trained_ckpt_path = os.path.join(
    CKPT_DIR, "actor", str(MAX_STEPS), "model_params"
)

abs_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(lora_policy, nnx.LoRAParam),
)
checkpointer = ocp.StandardCheckpointer()
trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

nnx.update(
    lora_policy,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(lora_policy, nnx.LoRAParam),
        trained_lora_params,
    ),
)

# %%
# The evaluation might take up to couple of minutes to finish. Please be patient.
STAGE = "sdpo"

sampler = sampler_lib.Sampler(
    transformer=lora_policy,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

num_correct, total, accuracy, partial_accuracy, format_accuracy = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{num_correct=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
    f" {format_accuracy=}%"
)
env_metrics = evaluate_env_reward(
    test_dataset, sampler, **GENERATION_CONFIGS["greedy"]
)
pprint(env_metrics)
RESULTS[STAGE] = {
    "accuracy": accuracy,
    "partial_accuracy": partial_accuracy,
    "format_accuracy": format_accuracy,
    **env_metrics,
}

# %% [markdown]
# ## Comparação: baseline vs GRPO vs SDPO
#
# Os números do GRPO são do run de referência (`grpo_gemma.ipynb`). Ele não
# mediu a reward do ambiente; para preencher essas colunas, rode
# `evaluate_env_reward` no modelo `gemma-3-1b-it-grpo-gsm8k` exportado.
#
# Com 64 exemplos, o erro padrão de uma acurácia perto de 50% é de ~6 pp.
# Diferenças menores que isso não são conclusivas.

# %%
GRPO_REFERENCE = {
    "accuracy": 48.4375,
    "partial_accuracy": 54.6875,
    "format_accuracy": 82.8125,
}
rows = {
    "baseline": RESULTS["baseline"],
    "grpo (ref)": GRPO_REFERENCE,
    "sdpo": RESULTS["sdpo"],
}
columns = list(RESULTS["sdpo"])
print(f"{'':12s}" + "".join(f"{c.removeprefix('reward/'):>22s}" for c in columns))
for name, row in rows.items():
  cells = "".join(
      f"{row[c]:>22.3f}" if c in row else f"{'—':>22s}" for c in columns
  )
  print(f"{name:12s}{cells}")

# %% [markdown]
# ## Export Merged Lora Weights (Huggingface Format)
#

# %%
output_dir = f"./{MODEL_ID}-lora"
if USE_COLAB:
  output_dir = f"/tmp/content/{MODEL_ID}-lora"

if os.path.exists(output_dir):
  shutil.rmtree(output_dir)
os.makedirs(output_dir)

print(f"Saving merged LoRA model to {output_dir}")

# Use the save_lora_merged_model_as_safetensors function
gemma_params.save_lora_merged_model_as_safetensors(
    local_model_path=local_model_path,
    output_dir=output_dir,
    lora_model=lora_policy,
    rank=RANK,
    alpha=ALPHA,
)

print("\n" + "=" * 60)
print("Model saved successfully!")
print(f"Output directory: {output_dir}")
print("=" * 60)

print("\nSaved files:")
for f in os.listdir(output_dir):
  size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
  print(f"  {f:<30} {size:>10.2f} MB")

# For Colab: Download as zip
if USE_COLAB:
  from google.colab import files

  shutil.make_archive(output_dir, "zip", output_dir)
  files.download(f"{output_dir}.zip")

# %%
from huggingface_hub import HfApi
api = HfApi()
info = api.whoami()
# print(info["name"])
# print(info["auth"]["accessToken"].get("fineGrained", {}))

# %%
repo_id = f"{info['name']}/gemma-3-1b-it-sdpo-gsm8k"

api.create_repo(repo_id, private=True, exist_ok=True)
api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    commit_message="Gemma 3 1B IT + SDPO (Tunix) GSM8K, LoRA merged",
)
print(f"https://huggingface.co/{repo_id}")

# %%
