import yaml
from stable_baselines3.common.evaluation import evaluate_policy

from rl import config_parser, env_setup


def enjoy(config):
    config["env"]["n_envs"] = 1
    env = env_setup(config["env"])

    Model = config["model"]

    model = Model.load(f"models/{config['model_name']}", env=env)
    # for i in range(500_000, 10_500_000, 500_000):
    # model = Model.load(
    #     f"checkpoints/Ppo_zoo_3stk_fs4_hw84_{i}_steps",
    #     env=env,
    # )

    evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=1,
        deterministic=True,
    )


if __name__ == "__main__":
    with open("models_configs/best_model.yml") as fp:
        config = yaml.safe_load(fp)
    config = config_parser(config)
    enjoy(config)
