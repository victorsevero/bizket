import os
import pickle
from datetime import datetime
from copy import deepcopy

import numpy as np
import pygad
import pygad.nn
import pygad.gann
from tqdm import trange, tqdm

from server import Server


def bot():
    global best_solution, model_kwargs
    population_vectors = pygad.gann.population_as_vectors(
        population_networks=gann.population_networks
    )

    model = pygad.GA(
        initial_population=population_vectors.copy(),
        **model_kwargs,
    )

    model.run()
    now_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = f"results/{now_str}"
    os.makedirs(results_dir)
    model.plot_fitness(save_dir=f"{results_dir}/fitness.png")
    with open(f"{results_dir}/model.pkl", "wb") as fp:
        pickle.dump(best_solution, fp)


def run_saved_model(filename=None):
    nn = load_model(filename)
    game_loop(nn)


def load_model(filename=None):
    if filename is None:
        filename = sorted(list_dirs())[-1] + "/model.pkl"

    with open(filename, "rb") as fp:
        nn = pickle.load(fp)

    return nn


def fitness_func(solution, sol_idx):
    global gann, server, best_solution, best_fitness, p_bar1

    data = game_loop(gann.population_networks[sol_idx])

    fitness = calc_fitness(data)
    if fitness > best_fitness:
        best_fitness = fitness
        best_solution = deepcopy(gann.population_networks[sol_idx])

    p_bar1.update()
    p_bar1.refresh()

    return fitness


def game_loop(last_layer):
    global server

    server.load_state()
    for _ in range(60 * 60):
        data = server.get_msg()
        data_inputs = msg_to_nn_inputs(data)
        buttons = pygad.nn.predict(
            last_layer=last_layer,
            data_inputs=data_inputs,
        )
        joypad_msg = buttons_map(buttons)
        server.send_msg(joypad_msg)
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
        scale_max=-1,  # TODO: try centralizing Zero to jump distance (0 here and change Y center point above)
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
    global gann, p_bar0, p_bar1

    population_matrices = pygad.gann.population_as_matrices(
        population_networks=gann.population_networks,
        population_vectors=model.population,
    )

    gann.update_population_trained_weights(
        population_trained_weights=population_matrices
    )
    fitness = model.best_solutions_fitness[model.generations_completed - 1]
    tqdm.write(f"Fitness = {fitness:.3f}")
    p_bar0.update()
    p_bar0.refresh()
    p_bar1.n = 0
    p_bar1.refresh()


def list_dirs(path="results/"):
    return [path + x for x in os.listdir(path) if os.path.isdir(path + x)]


if __name__ == "__main__":
    gann_kwargs = {
        "num_solutions": 10,
        "num_neurons_input": 4,
        "num_neurons_hidden_layers": [10, 10],
        "num_neurons_output": 5,
        "hidden_activations": "relu",
        "output_activation": "softmax",
    }
    gann = pygad.gann.GANN(**gann_kwargs)

    model_kwargs = {
        "num_generations": 200,
        "num_parents_mating": 1,
        "fitness_func": fitness_func,
        "keep_elitism": 2,
        "on_generation": callback_generation,
        "save_best_solutions": True,
        "suppress_warnings": True,
    }

    server = Server()
    best_solution = None
    best_fitness = -49

    p_bar0 = trange(
        model_kwargs["num_generations"],
        desc="Generation",
        position=0,
    )
    p_bar1 = trange(
        gann_kwargs["num_solutions"],
        desc="Individual",
        position=1,
    )

    # nn = load_model()
    # population_networks = []
    # for _ in range(gann_kwargs["num_solutions"]):
    #     population_networks.append(deepcopy(nn))

    # gann.population_networks = population_networks
    # bot()

    run_saved_model()
