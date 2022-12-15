from pathlib import Path

import yaml
import optuna
from optuna.visualization import plot_optimization_history
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


from rl import env_setup


def sample_ppo_params(trial: optuna.Trial):
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    n_steps = trial.suggest_categorical(
        "n_steps",
        [32, 64, 128, 256, 512, 1024, 2048, 4096],
    )
    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64, 128, 256, 512],
    )

    if (config["env"]["n_envs"] * n_steps) % batch_size > 0:
        raise optuna.TrialPruned()

    gamma = trial.suggest_categorical(
        "gamma",
        [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda",
        [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm",
        [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5],
    )
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    activation_fn = trial.suggest_categorical(
        "activation_fn",
        ["tanh", "relu", "elu", "leaky_relu"],
    )

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def optimize_agent(trial):
    """Train the model and optimize
    Optuna maximises the negative log likelihood, so we
    need to negate the reward here
    """
    model_params = sample_ppo_params(trial)

    model = PPO(
        policy="CnnPolicy",
        env=env,
        tensorboard_log="logs/z1_optimization",
        verbose=0,
        seed=666,
        **model_params,
    )
    model.learn(100_000, log_interval=1)
    reward, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=1,
        deterministic=True,
    )

    return reward


if __name__ == "__main__":
    with open("models_configs/z1_opt.yml") as fp:
        config = yaml.safe_load(fp)

    env = env_setup(config["env"])
    name = "ppo"

    db_path = "studies/z1.db"
    Path(db_path).touch(exist_ok=True)

    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        load_if_exists=True,
    )

    n_trials = 100
    n_trials -= len([x for x in study.trials if x.state.name == "COMPLETE"])

    try:
        study.optimize(
            optimize_agent,
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_params["policy_kwargs"] = {
            "activation_fn": best_params["activation_fn"],
            "ortho_init": best_params["ortho_init"],
        }
        best_params.pop("activation_fn")
        best_params.pop("ortho_init")

        with open(f"{name}_opt.yml", "w") as fp:
            yaml.safe_dump(study.best_params, fp)
    finally:
        fig = plot_optimization_history(study)
        fig.show()
