from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from emulator_grid import set_emulator_grid
from mmx4_env import Mmx4Env


def make_mmx4_env(port):
    def _init():
        env = Mmx4Env(port)
        return env

    return _init


if __name__ == "__main__":
    N_PROCESSES = 1
    DEFAULT_PORT = 6969
    env_fns = [make_mmx4_env(DEFAULT_PORT + i) for i in range(N_PROCESSES)]

    env = VecMonitor(SubprocVecEnv(env_fns))
    set_emulator_grid(N_PROCESSES)

    model = A2C(
        policy="CnnPolicy",
        env=env,
        tensorboard_log="mmx4_logs/",
        verbose=1,
        seed=666,
    )
    model.learn(
        total_timesteps=200_000,
        log_interval=1000 / (5 * N_PROCESSES),
        tb_log_name=f"default_a2c_10frames_gray",
        progress_bar=True,
    )
    model.save("a2c_mmx4")
