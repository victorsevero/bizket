from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecMonitor,
)

from emulator_grid import set_emulator_grid
from mmx4_env import Mmx4Env


def make_mmx4_env(port):
    def _init():
        env = Mmx4Env(port)
        return env

    return _init


def env_setup(n_processes, default_port=6969, evaluating=False):
    if not evaluating:
        env_fns = [make_mmx4_env(default_port + i) for i in range(n_processes)]
    else:
        env_fns = [make_mmx4_env(default_port)]

    env = VecMonitor(
        VecFrameStack(
            VecTransposeImage(
                SubprocVecEnv(env_fns),
            ),
            n_stack=3,
        )
    )

    if not evaluating:
        set_emulator_grid(n_processes)
    else:
        set_emulator_grid(n_processes + 1)

    return env


def new_training(n_processes, model_name):
    env = env_setup(n_processes)

    model = A2C(
        policy="CnnPolicy",
        env=env,
        tensorboard_log="mmx4_logs/",
        verbose=1,
        seed=666,
    )
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1000 // (5 * n_processes),
        tb_log_name=model_name,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    model.save(model_name)


def continue_training(n_processes, model_name):
    env = env_setup(n_processes)

    model = A2C.load(model_name, env=env)
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1000 // (5 * n_processes),
        tb_log_name=model_name,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    model.save(model_name)


if __name__ == "__main__":
    total_timesteps = 1_000_000
    model_name = "defaultA2c_3stk_normRew"
    new_training(n_processes=1, model_name="test")
    # continue_training(n_processes=8, model_name=model_name)
