import json

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecMonitor,
)
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.callbacks import CheckpointCallback

from emulator_grid import set_emulator_grid
from mmx4_env import Mmx4Env
from callbacks import ModelArchCallback


def make_mmx4_env(port):
    def _init():
        env = Mmx4Env(port)
        return env

    return _init


def env_setup(config, default_port=6969, evaluating=False):
    n_processes = config["n_processes"]
    n_stack = config["n_stack"]

    if not evaluating:
        env_fns = [make_mmx4_env(default_port + i) for i in range(n_processes)]
    else:
        env_fns = [make_mmx4_env(default_port)]

    env = VecMonitor(
        VecFrameStack(
            VecTransposeImage(
                SubprocVecEnv(env_fns),
            ),
            n_stack=n_stack,
        )
    )

    if not evaluating:
        set_emulator_grid(n_processes)
    else:
        set_emulator_grid(n_processes + 1)

    return env


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def config_parser(config):
    if config["model"] == "A2C":
        config["model"] = A2C
    elif config["model"] == "DQN":
        config["model"] = DQN
    elif config["model"] == "PPO":
        config["model"] = PPO
    else:
        raise ValueError(f"Invalid model {config['model']}")

    if config["learn_kwargs"]["callback"] == "ModelArchCallback":
        config["learn_kwargs"]["callback"] = [
            ModelArchCallback(),
            CheckpointCallback(
                save_freq=500_000 // config["n_processes"],
                save_path="checkpoints/",
                name_prefix=config["model_name"],
                save_replay_buffer=True,
            ),
        ]

    if (
        config.get("model_kwargs", {})
        .get("policy_kwargs", {})
        .get("optimizer_class")
        == "RMSpropTFLike"
    ):
        config["model_kwargs"]["policy_kwargs"][
            "optimizer_class"
        ] = RMSpropTFLike

    if str(config["model_kwargs"]["learning_rate"]).startswith("lin_"):
        value = float(config["model_kwargs"]["learning_rate"].split("_")[-1])
        config["model_kwargs"]["learning_rate"] = linear_schedule(value)

    if str(config["model_kwargs"]["clip_range"]).startswith("lin_"):
        value = float(config["model_kwargs"]["clip_range"].split("_")[-1])
        config["model_kwargs"]["clip_range"] = linear_schedule(value)

    return config


def new_training(config):
    env = env_setup(config)

    Model = config["model"]
    model = Model(env=env, **config["model_kwargs"])

    model.learn(
        tb_log_name=config["model_name"],
        reset_num_timesteps=config["new_model"],
        **config["learn_kwargs"],
    )
    model.save(f"models/{config['model_name']}")


def continue_training(config):
    env = env_setup(config)

    Model = config["model"]
    model = Model.load(f"models/{config['model_name']}", env=env)

    model.learn(
        tb_log_name=config["model_name"],
        reset_num_timesteps=config["new_model"],
        **config["learn_kwargs"],
    )
    model.save(f"models/{config['model_name']}")


if __name__ == "__main__":
    with open("config_ppo.json") as fp:
        config = json.load(fp)
    config = config_parser(config)

    if config["new_model"]:
        new_training(config)
    else:
        continue_training(config)
