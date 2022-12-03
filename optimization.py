import json
import pickle

import optuna
from torch import nn
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

from rl import env_setup


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


def sample_a2c_params(trial: optuna.Trial):
    """
    Sampler for A2C hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical(
        "gamma",
        [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
    )
    normalize_advantage = trial.suggest_categorical(
        "normalize_advantage",
        [False, True],
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm",
        [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5],
    )
    # Toggle PyTorch RMS Prop (different from TF one, cf doc)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda",
        [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
    )
    n_steps = trial.suggest_categorical(
        "n_steps",
        [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )
    lr_schedule = trial.suggest_categorical(
        "lr_schedule",
        ["linear", "constant"],
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu"]
    )

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def optimize_agent(trial):
    """Train the model and optimize
    Optuna maximises the negative log likelihood, so we
    need to negate the reward here
    """
    global env, eval_env
    model_params = sample_a2c_params(trial)

    model = A2C(
        policy="CnnPolicy",
        env=env,
        verbose=0,
        seed=666,
        **model_params,
    )
    model.learn(1000)
    mean_reward, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=1,
        deterministic=True,
    )

    return mean_reward


if __name__ == "__main__":
    env = env_setup(8)
    eval_env = env_setup(8, default_port=6977, evaluating=True)

    name = "a2c"
    study = optuna.create_study(
        study_name=name,
        storage="sqlite:///studies.db",
        direction="maximize",
        load_if_exists=True,
    )
    n_trials = 100
    study.optimize(optimize_agent, n_trials=n_trials, show_progress_bar=True)

    with open(f"{name}_params.pkl", "wb") as fp:
        pickle.dump(study.best_params, fp)
    with open(f"{name}_params.json", "w") as fp:
        json.dump(study.best_params, fp)
