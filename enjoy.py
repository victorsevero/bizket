import json

from stable_baselines3.common.evaluation import evaluate_policy

from rl import config_parser, env_setup


def enjoy(config):
    env = env_setup(config)
    Model = config["model"]

    model = Model.load(f"models/{config['model_name']}", env=env)

    evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=1,
        deterministic=True,
    )


if __name__ == "__main__":
    with open("config_best.json") as fp:
        config = json.load(fp)
    config = config_parser(config)
    enjoy(config)
