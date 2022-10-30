import os
import pickle
from datetime import datetime

import numpy as np
import pygad
import pygad.nn
import pygad.gann
from tqdm import trange, tqdm
import seaborn as sns

from server import Server
from emulator_grid import start_emulator, set_emulator_grid, close_emulators


sns.set_theme()


def bot():
    global model_kwargs, handles
    population_vectors = pygad.gann.population_as_vectors(
        population_networks=gann.population_networks
    )

    model = pygad.GA(
        initial_population=population_vectors.copy(),
        **model_kwargs,
    )
    try:
        model.run()
    finally:
        close_emulators(handles)
        try:
            server.close()
        except:
            ...

        now_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        results_dir = f"results/{now_str}"
        os.makedirs(results_dir)
        model.save(f"{results_dir}/model")
        model.plot_fitness(save_dir=f"{results_dir}/fitness.png")


def run_saved_model(filename=None):
    model = load_model(filename)

    weights = model.best_solution(model.last_generation_fitness)[0]
    weights = [weights[:20].reshape(4, 5), weights[20:].reshape(5, 5)]
    nn = gann.population_networks[0]
    pygad.nn.update_layers_trained_weights(nn, weights)

    data = game_loop(nn, 0)
    print(calc_fitness(data))


def load_model(filename=None):
    global gann, n_processes
    if filename is None:
        filename = sorted(list_dirs())[-1] + "/model.pkl"

    with open(filename, "rb") as fp:
        model = pickle.load(fp)

    return model


def fitness_func(solution, sol_idx):
    global gann, server, best_fitness, n_processes

    conn_idx = sol_idx % n_processes

    data = game_loop(gann.population_networks[sol_idx], conn_idx)

    fitness = calc_fitness(data)
    if fitness > best_fitness:
        best_fitness = fitness

    return fitness


def game_loop(last_layer, conn_idx):
    global server

    server.load_state(conn_idx)
    for _ in range(60 // 4 * 60):
        data = server.get_msg(conn_idx)
        data_inputs = msg_to_nn_inputs(data)
        buttons = pygad.nn.predict(
            last_layer=last_layer,
            data_inputs=data_inputs,
        )
        joypad_msg = buttons_map(buttons)
        server.send_msg(joypad_msg, conn_idx)
        if (data["player_hp"] == 0) or (data["boss_hp"] == 0):
            break

    return data


def msg_to_nn_inputs(data):
    data_inputs = data.copy()
    data_inputs.pop("player_hp")
    data_inputs.pop("boss_hp")

    data_inputs = abs_to_relative(data_inputs)

    data_inputs = np.array([[*data_inputs.values()]])
    return data_inputs


def abs_to_relative(data):
    data = data.copy()
    data["boss_x"] = data["boss_x"] - data["player_x"]
    data["boss_y"] = data["boss_y"] - data["player_y"]
    data["player_x"] = data["player_x"] - (0x12EF + 0x115D) / 2
    data["player_y"] = data["player_y"] - (0x01BA + 0x008D) / 2

    data["boss_x"] = scaler(
        data["boss_x"],
        value_min=0x115D - 0x12EF,
        value_max=0x12EF - 0x115D,
    )
    data["boss_y"] = scaler(
        data["boss_y"],
        value_min=0x008D - 0x01BA,
        value_max=0x01BA - 0x008D,
        scale_min=1,
        scale_max=-1,  # TODO: try centralizing char to jump distance (0 here and change Y center point above)
    )
    data["player_x"] = scaler(
        data["player_x"],
        value_min=(0x115D - 0x12EF) / 2,
        value_max=(0x12EF - 0x115D) / 2,
    )
    data["player_y"] = scaler(
        data["player_y"],
        value_min=(0x008D - 0x01BA) / 2,
        value_max=(0x01BA - 0x008D) / 2,
        scale_min=1,
        scale_max=-1,
    )

    return data


def scaler(value, value_min, value_max, scale_min=-1, scale_max=1):
    value_std = (value - value_min) / (value_max - value_min)
    return value_std * (scale_max - scale_min) + scale_min


def calc_fitness(data):
    boss_weight = 32
    # boss_weight = 1
    fitness = scaler(
        data["player_hp"] - boss_weight * data["boss_hp"],
        value_min=-boss_weight * 48,
        value_max=32,
        scale_min=0,
        scale_max=100,
    )
    return fitness


def buttons_map(buttons):
    joypad_msg = ""
    for button in buttons:
        if button == 0:
            joypad_msg += "l"
        elif button == 1:
            joypad_msg += "r"
        elif button == 2:
            joypad_msg += "x"
        elif button == 3:
            joypad_msg += "o"
        elif button == 4:
            joypad_msg += "s"
        else:
            raise ValueError(f"{button} is not a valid button.")

    return joypad_msg


def callback_generation(model):
    global gann, p_bar

    population_matrices = pygad.gann.population_as_matrices(
        population_networks=gann.population_networks,
        population_vectors=model.population,
    )

    gann.update_population_trained_weights(
        population_trained_weights=population_matrices
    )
    fitness = model.best_solutions_fitness[model.generations_completed - 1]

    tqdm.write(f"Fitness = {fitness:#.3g}%")
    p_bar.update()
    p_bar.refresh()


def list_dirs(path="results/"):
    return [path + x for x in os.listdir(path) if os.path.isdir(path + x)]


if __name__ == "__main__":
    n_processes = 8

    gann_kwargs = {
        "num_solutions": n_processes,
        "num_neurons_input": 4,
        "num_neurons_hidden_layers": [5],
        "num_neurons_output": 5,
        "hidden_activations": "relu",
        "output_activation": "softmax",
    }
    gann = pygad.gann.GANN(**gann_kwargs)

    model_kwargs = {
        "num_generations": 100,
        "num_parents_mating": 1,
        "fitness_func": fitness_func,
        "keep_elitism": 1,
        "crossover_type": None,
        "mutation_type": "random",
        "mutation_probability": 0.1,
        "mutation_by_replacement": True,
        "random_mutation_min_val": -1.0,
        "random_mutation_max_val": 1.0,
        "gene_space": {"low": -1, "high": 1},
        "on_generation": callback_generation,
        "save_best_solutions": True,
        "suppress_warnings": True,
        "parallel_processing": ["process", n_processes],
        "random_seed": 666,
    }

    server = Server(n_connections=n_processes)
    for _ in range(n_processes):
        start_emulator()
        server.accept_connection()
    handles = set_emulator_grid()

    best_solution = None
    best_fitness = -1

    p_bar = trange(
        model_kwargs["num_generations"],
        desc="Generation",
        position=0,
    )

    # nn = load_model()
    # population_networks = []
    # for _ in range(gann_kwargs["num_solutions"]):
    #     population_networks.append(deepcopy(nn))

    # gann.population_networks = population_networks
    bot()

    # server = Server(n_connections=1)
    # server.accept_connection()
    # run_saved_model()
