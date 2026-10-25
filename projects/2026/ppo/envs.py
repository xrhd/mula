"""Registry of single-player pgx games. Adding a game is one line."""

from typing import Literal, NamedTuple


class EnvSpec(NamedTuple):
  pgx_id: str
  title: str  # shown in the video
  reward_scale: float = 1.0  # multiplies training rewards (not eval/video)
  render_kind: Literal["minatar", "2048"] = "minatar"


ENVS = {
    "minatar-breakout": EnvSpec("minatar-breakout", "Breakout"),
    "minatar-freeway": EnvSpec("minatar-freeway", "Freeway"),
    "minatar-asterix": EnvSpec("minatar-asterix", "Asterix"),
    "minatar-seaquest": EnvSpec("minatar-seaquest", "Seaquest"),
    "minatar-space_invaders": EnvSpec("minatar-space_invaders", "Space Invaders"),
    # merges are worth up to thousands of points
    "2048": EnvSpec("2048", "2048", reward_scale=0.01, render_kind="2048"),
}
