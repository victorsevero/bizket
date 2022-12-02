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


def env_setup(n_processes, default_port=6969):
    env_fns = [make_mmx4_env(default_port + i) for i in range(n_processes)]

    env = VecMonitor(
        VecFrameStack(
            VecTransposeImage(
                SubprocVecEnv(env_fns),
            ),
            n_stack=3,
        ),
    )
    set_emulator_grid(n_processes)

    return env


if __name__ == "__main__":
    n_processes = 8
    env = env_setup(n_processes)

    model = A2C(
        policy="CnnPolicy",
        env=env,
        tensorboard_log="mmx4_logs/",
        verbose=1,
        seed=666,
    )
    model.learn(
        total_timesteps=100_000,
        log_interval=1000 // (5 * n_processes),
        tb_log_name=f"default_a2c_3_stack",
        progress_bar=True,
    )
    model.save("a2c")
