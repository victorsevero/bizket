from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from emulator_grid import close_emulators
from mmx4_env import Mmx4Env, N_PROCESSES


def make_env(Env, rank):
    def _init():
        env = Env(rank, time=120)
        return env

    return _init


try:
    env = VecMonitor(
        SubprocVecEnv(
            [make_env(Mmx4Env, i) for i in range(N_PROCESSES)],
            start_method="fork",
        )
    )

    model = A2C(
        policy="CnnPolicy",
        env=env,
        tensorboard_log="mmx4_logs/",
        verbose=1,
        seed=666,
    )
    model.learn(
        total_timesteps=1_000_000,
        log_interval=1000,
        tb_log_name=f"default_a2c",
        progress_bar=True,
    )
    model.save("a2c_mmx4")

    # obs = env.reset()
    # done = np.array(env.num_envs * [False])
    # while not done.any():
    #     env.step_async(
    #         [env.action_space.sample() for _ in range(env.num_envs)]
    #     )
    #     obs, reward, done, info = env.step_wait()
    # for i, instance_info in enumerate(info):
    #     print(
    #         f"{i}: Player={instance_info['player_hp']};"
    #         + f" Boss={instance_info['boss_hp']}"
    #     )
finally:
    ...
    # close_emulators()
    # server.close()

# env = DummyVecEnv([lambda: env])
# model = A2C.load("a2c_mmx4")
# obs = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs)
#     _ = env.step(action)
# close_emulators(handles)
