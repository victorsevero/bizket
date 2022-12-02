from stable_baselines3 import A2C

from rl import env_setup


def enjoy(path):
    env = env_setup(1)
    model = A2C.load(path, env=env)

    obs = env.reset()
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, _ = env.step(action)


if __name__ == "__main__":
    enjoy("a2c_3_stack")