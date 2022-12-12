import os

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecMonitor,
)
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)
from torch import nn

from emulator_grid import set_emulator_grid
from mmx4_env import Mmx4Env
from callbacks import ModelArchCallback


def make_mmx4_env(port):
    def _init():
        env = Mmx4Env(port)
        return env

    return _init


def env_setup(env_config, default_port=6969, evaluating=False):
    n_envs = env_config["n_envs"]
    n_stack = env_config["n_stack"]

    if not evaluating:
        env_fns = [make_mmx4_env(default_port + i) for i in range(n_envs)]
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
        set_emulator_grid(n_envs)
    else:
        set_emulator_grid(n_envs + 1)

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
    if config["model"] == "PPO":
        config["model"] = PPO
    else:
        raise ValueError(f"Invalid model {config['model']}")

    if str(config["model_kwargs"]["learning_rate"]).startswith("lin_"):
        value = float(config["model_kwargs"]["learning_rate"].split("_")[-1])
        config["model_kwargs"]["learning_rate"] = linear_schedule(value)

    if str(config["model_kwargs"]["clip_range"]).startswith("lin_"):
        value = float(config["model_kwargs"]["clip_range"].split("_")[-1])
        config["model_kwargs"]["clip_range"] = linear_schedule(value)

    if "policy_kwargs" in config["model_kwargs"].keys():
        config["model_kwargs"]["policy_kwargs"]["activation_fn"] = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }[config["model_kwargs"]["policy_kwargs"]["activation_fn"]]

    return config


def train_model(config):
    env = env_setup(config["env"])
    Model: PPO = config["model"]

    if os.path.exists(f"models/{config['model_name']}.zip"):
        model = Model.load(f"models/{config['model_name']}", env=env)
        reset_num_timesteps = False
    elif config["past_model"] is not None:
        model = Model.load(f"models/{config['past_model']}", env=env)
        reset_num_timesteps = True
    else:
        model = Model(
            env=env,
            tensorboard_log="logs/mmx4/",
            verbose=0,
            seed=666,
            device="auto",
            _init_setup_model=True,
            **config["model_kwargs"],
        )
        reset_num_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=config["save_freq"] // config["env"]["n_envs"],
        save_path=f"checkpoints/{config['model_name']}/",
        name_prefix="model",
        save_replay_buffer=True,
    )

    model.learn(
        tb_log_name=config["model_name"],
        reset_num_timesteps=reset_num_timesteps,
        # callback=[ModelArchCallback(), checkpoint_callback],
        callback=[checkpoint_callback],
        progress_bar=True,
        **config["learn_kwargs"],
    )
    model.save(f"models/{config['model_name']}")


if __name__ == "__main__":
    with open("models_configs/zero_zoo_cr.yml") as fp:
        config = yaml.safe_load(fp)
    config = config_parser(config)

    train_model(config)
