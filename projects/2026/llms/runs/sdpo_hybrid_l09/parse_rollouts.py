"""Extrai, dos prints do check_numbers, a acurácia/formato por rodada de validação.

python3 parse_rollouts.py sdpo_full.log.gz ../../grpo_gemma.ipynb

Cada bloco START..END traz a 1ª resposta do micro-batch. As perguntas de validação
se repetem a cada rodada (EVAL_EVERY_N_STEPS=64); as de treino aparecem uma vez.
"""
import collections, gzip, json, re, sys

reasoning_start, reasoning_end = "<reasoning>", "</reasoning>"
solution_start, solution_end = "<answer>", "</answer>"
match_format = re.compile(
    rf"^[\s]{{0,}}{reasoning_start}.+?{reasoning_end}.*?{solution_start}(.+?){solution_end}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL)
BLOCK = re.compile(r"START =+\nQuestion:\t(.*?)\nAnswer:\t(.*?)\nResponse:\t(.*?)\nExtracted:\t(.*?)\nEND =+", re.DOTALL)


def read_log(path):
  """Lê um log (.log ou .log.gz) ou o output da célula de treino de um .ipynb."""
  if path.endswith(".ipynb"):
    nb = json.load(open(path))
    cell = next(c for c in nb["cells"] if c["cell_type"] == "code"
                and ".train(train_dataset" in "".join(c["source"]))
    return "".join("".join(o.get("text", [])) for o in cell["outputs"]
                   if o.get("output_type") == "stream")
  opener = gzip.open if path.endswith(".gz") else open
  with opener(path, "rt", errors="replace") as f:
    return f.read()


def blocks(path):
  out = []
  for q, a, r, e in BLOCK.findall(read_log(path)):
    try:
      correct = float(e.strip()) == float(a.strip())
    except ValueError:
      correct = False
    out.append(dict(q=q.strip(), correct=correct, fmt=match_format.search(r) is not None))
  return out


def eval_rounds(bs, val_qs):
  """Agrupa blocos consecutivos de validação em rodadas."""
  rounds, cur, train_since = [], [], 0
  for b in bs:
    if b["q"] in val_qs:
      cur.append(b)
    else:
      if cur:
        rounds.append((train_since, cur)); cur = []
      train_since += 1
  if cur:
    rounds.append((train_since, cur))
  return rounds


if __name__ == "__main__":
  sdpo, grpo = blocks(sys.argv[1]), blocks(sys.argv[2])
  freq = collections.Counter(b["q"] for b in sdpo)
  val_qs = {q for q, n in freq.items() if n >= 3}
  print(f"SDPO: {len(sdpo)} blocos, {len(val_qs)} perguntas de validação | GRPO: {len(grpo)} blocos")
  for name, bs in (("SDPO", sdpo), ("GRPO", grpo)):
    rs = eval_rounds(bs, val_qs)
    print(f"\n{name}: {len(rs)} rodadas de validação (passos de treino vistos no log antes da rodada, n, acc%, formato%)")
    for since, r in rs:
      n = len(r)
      print(f"  {since:5d}  n={n:3d}  acc={100*sum(b['correct'] for b in r)/n:5.1f}  fmt={100*sum(b['fmt'] for b in r)/n:5.1f}")
    tr = [b for b in bs if b["q"] not in val_qs]
    last = tr[-150:]
    print(f"  treino, últimos {len(last)} passos: acc={100*sum(b['correct'] for b in last)/len(last):.1f}  fmt={100*sum(b['fmt'] for b in last)/len(last):.1f}")
