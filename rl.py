import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from cluster import close_emulators
from mmx4_env import Mmx4Env, N_PROCESSES, handles, server


def make_env(Env, rank):
    def _init():
        env = Env(rank, time=120)
        return env

    return _init


try:
    tmp_path = "tmp/"
    env = SubprocVecEnv(
        [make_env(Mmx4Env, i) for i in range(N_PROCESSES)],
        start_method="fork",
    )
    model = A2C(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        seed=666,
    )
    model.learn(total_timesteps=400, log_interval=100, progress_bar=True)
    model.save("a2c_mmx4")
    # obs = env.reset()
    # done = np.array(env.num_envs * [False])
    # while not done.any():
    #     env.step_async(
    #         [env.action_space.sample() for _ in range(env.num_envs)]
    #     )
    #     obs, reward, done, info = env.step_wait()
finally:
    close_emulators(handles)
    server.close()


# env = DummyVecEnv([lambda: env])
# model = A2C.load("a2c_mmx4")
# obs = env.reset()
# done = False
# while not terminated:
#     action, _ = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
# close_emulators(handles)
