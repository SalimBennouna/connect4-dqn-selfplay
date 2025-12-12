import gym
import gym_connect4  # noqa: F401

from ..utils.seed import set_seed


def make_env(cfg: dict, seed: int | None = None):
    if seed is not None:
        set_seed(seed)

    env_cfg = cfg["env"]
    env = gym.make(
        env_cfg["id"],
        height=env_cfg["height"],
        width=env_cfg["width"],
        connect=env_cfg["connect"],
    )

    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    return env