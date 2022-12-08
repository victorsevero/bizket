import json

import optuna
from torch import nn
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from rl import env_setup, linear_schedule


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
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn = trial.suggest_categorical(
        "activation_fn",
        ["tanh", "relu"],
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


def sample_dqn_params(trial: optuna.Trial):
    """
    Sampler for DQN hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64, 100, 128, 256, 512],
    )
    buffer_size = trial.suggest_categorical(
        "buffer_size",
        [int(1e4), int(5e4), int(1e5), int(1e6)],
    )
    exploration_final_eps = trial.suggest_float(
        "exploration_final_eps",
        0,
        0.2,
    )
    exploration_fraction = trial.suggest_float(
        "exploration_fraction",
        0,
        0.5,
    )
    target_update_interval = trial.suggest_categorical(
        "target_update_interval",
        [1, 1000, 5000, 10000, 15000, 20000],
    )
    learning_starts = trial.suggest_categorical(
        "learning_starts",
        [0, 1000, 5000, 10000, 20000],
    )

    train_freq = trial.suggest_categorical(
        "train_freq",
        [1, 4, 8, 16, 128, 256, 1000],
    )
    subsample_steps = trial.suggest_categorical(
        "subsample_steps",
        [1, 2, 4, 8],
    )
    gradient_steps = max(train_freq // subsample_steps, 1)

    net_arch = trial.suggest_categorical(
        "net_arch",
        ["tiny", "small", "medium"],
    )

    net_arch = {
        "tiny": [64],
        "small": [64, 64],
        "medium": [256, 256],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    return hyperparams


def small_grid_sample_a2c_params(trial: optuna.Trial):
    """
    Sampler for A2C hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical(
        "gamma",
        [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
    )
    lr_schedule = trial.suggest_categorical(
        "lr_schedule",
        ["linear", "constant"],
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    activation_fn = trial.suggest_categorical(
        "activation_fn",
        ["tanh", "relu"],
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
        "gamma": gamma,
        "learning_rate": learning_rate,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
        ),
    }


def sample_ppo_params(trial: optuna.Trial):
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical(
        "batch_size",
        [8, 16, 32, 64, 128, 256, 512],
    )
    n_steps = trial.suggest_categorical(
        "n_steps",
        [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )
    gamma = trial.suggest_categorical(
        "gamma",
        [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    lr_schedule = trial.suggest_categorical(
        "lr_schedule",
        ["linear", "constant"],
    )
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
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
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    activation_fn = trial.suggest_categorical(
        "activation_fn",
        ["tanh", "relu", "elu", "leaky_relu"],
    )

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
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
    # model_params = small_grid_sample_a2c_params(trial)
    # model_params = sample_dqn_params(trial)
    model_params = sample_ppo_params(trial)

    try:
        model = PPO(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            seed=666,
            **model_params,
        )
        model.learn(100_000)
        reward, _ = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=1,
            deterministic=True,
        )

        return reward
    except:
        return -1


if __name__ == "__main__":
    with open("config_ppo.json") as fp:
        config = json.load(fp)
    n_processes = config["n_processes"]

    env = env_setup(config)
    eval_env = env_setup(config, default_port=6977, evaluating=True)

    name = "ppo"
    # name = "dqn"
    study = optuna.create_study(
        study_name=name,
        storage="sqlite:///studies.db",
        direction="maximize",
        load_if_exists=True,
    )
    n_trials = 100
    # completed = False
    # while not completed:
    # try:
    study.optimize(optimize_agent, n_trials=n_trials, show_progress_bar=True)
    # completed = True
    # except RuntimeError:
    #     completed_trials = [
    #         x for x in study.trials if x.state.name == "COMPLETE"
    #     ]
    #     n_completed_trials = len(completed_trials)
    #     n_trials -= n_completed_trials

    with open(f"params_{name}.json", "w") as fp:
        json.dump(study.best_params, fp, indent=4)
