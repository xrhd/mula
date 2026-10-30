"""Junta os prints do Tensorboard (GRPO vs SDPO) lado a lado, com rótulos."""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK_2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)
panels = [
    ("GRPO (grpo_gemma.ipynb, TPU v5e)", "5.19 no passo 3328 · eixo y linear", "tb_grpo_val_score.png"),
    ("SDPO híbrido logit-level, λ=0.9 (TPU v6e)", "pico de 6.13 no passo 512; 4.61 no passo 3328 · eixo y em escala log", "tb_sdpo_val_score.png"),
]
imgs = [Image.open(os.path.join(HERE, f)).convert("RGB") for _, _, f in panels]
h = max(i.height for i in imgs)
imgs = [i.resize((round(i.width * h / i.height), h)) for i in imgs]
pad, head, foot = 24, 110, 70


def font(size, bold=False):
  for name in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
               "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
    try:
      return ImageFont.truetype(name, size)
    except OSError:
      pass
  return ImageFont.load_default()


W = sum(i.width for i in imgs) + pad * (len(imgs) + 1)
canvas = Image.new("RGB", (W, h + head + foot), SURFACE)
d = ImageDraw.Draw(canvas)
d.text((pad, 14), "rewards/eval/score/mean: GRPO vs SDPO (Gemma-3-1B-IT, GSM8K, mesma validação)", fill=INK, font=font(30, True))
x = pad
for (title, sub, _), img in zip(panels, imgs):
  d.text((x, 60), title, fill=INK, font=font(24, True))
  canvas.paste(img, (x, head))
  d.text((x, head + h + 16), sub, fill=INK_2, font=font(22))
  x += img.width + pad
out = os.path.join(HERE, "grpo_vs_sdpo_val_score.png")
canvas.save(out)
print(out, canvas.size)
