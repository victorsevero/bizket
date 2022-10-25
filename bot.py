import os
import pickle
from datetime import datetime
from copy import deepcopy

import numpy as np
import pygad
import pygad.nn
import pygad.gann

from server import Server


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


def x_scaler(x):
    x_min = 4445
    x_max = 4847
    x_std = (x - x_min) / (x_max - x_min)
    return 2 * x_std - 1


def y_scaler(y):
    y_min = 141
    y_max = 442
    y_std = (y - y_min) / (y_max - y_min)
    return 2 * y_std - 1


def msg_to_nn_inputs(data):
    data_inputs = data.copy()
    data_inputs.pop("player_hp")
    data_inputs.pop("boss_hp")

    data_inputs["player_x"] = x_scaler(data_inputs["player_x"])
    data_inputs["player_y"] = y_scaler(data_inputs["player_y"])
    data_inputs["boss_x"] = x_scaler(data_inputs["boss_x"])
    data_inputs["boss_y"] = y_scaler(data_inputs["boss_y"])

    data_inputs = np.array([[*data_inputs.values()]])
    return data_inputs


def game_loop(last_layer):
    global server

    server.load_state()
    for _ in range(60 * 20):
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


def calc_fitness(data):
    return data["player_hp"] - data["boss_hp"]


def fitness_func(solution, sol_idx):
    global gann, server, best_solution, best_fitness

    data = game_loop(gann.population_networks[sol_idx])
    print(f"{sol_idx} ", end="")

    fitness = calc_fitness(data)
    if fitness > best_fitness:
        best_fitness = fitness
        best_solution = deepcopy(gann.population_networks[sol_idx])
    return fitness


def callback_generation(ga):
    global gann

    population_matrices = pygad.gann.population_as_matrices(
        population_networks=gann.population_networks,
        population_vectors=ga.population,
    )

    gann.update_population_trained_weights(
        population_trained_weights=population_matrices
    )
    print("")
    print(f"Generation = {ga.generations_completed}")
    print(
        f"Fitness    = {ga.best_solutions_fitness[ga.generations_completed-1]}"
    )
    pass


def bot():
    global best_solution
    population_vectors = pygad.gann.population_as_vectors(
        population_networks=gann.population_networks
    )

    ga = pygad.GA(
        num_generations=10,
        num_parents_mating=2,
        fitness_func=fitness_func,
        initial_population=population_vectors.copy(),
        keep_elitism=1,
        on_generation=callback_generation,
        save_best_solutions=True,
    )

    ga.run()
    now_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = f"results/{now_str}"
    os.makedirs(results_dir)
    ga.plot_fitness(save_dir=f"{results_dir}/fitness.png")
    with open(f"{results_dir}/model.pkl", "wb") as fp:
        pickle.dump(best_solution, fp)


def reshape_weights(weights):
    i = gann.num_neurons_input
    j = gann.num_neurons_hidden_layers[0]
    k = gann.num_neurons_output

    w1 = weights[: i * j].reshape(i, j)
    w2 = weights[i * j :].reshape(j, k)
    return [w1, w2]


def list_dirs(path="results/"):
    return [path + x for x in os.listdir(path) if os.path.isdir(path + x)]


def run_saved_model(filename=None):
    if filename is None:
        filename = sorted(list_dirs())[-1] + "/model.pkl"

    with open(filename, "rb") as fp:
        nn = pickle.load(fp)

    data = game_loop(nn)
    print(calc_fitness(data))


if __name__ == "__main__":
    gann = pygad.gann.GANN(
        num_solutions=10,
        num_neurons_input=4,
        num_neurons_hidden_layers=[5],
        num_neurons_output=5,
        hidden_activations="relu",
        output_activation="softmax",
    )

    server = Server()
    best_solution = None
    best_fitness = -49

    bot()
    # run_saved_model()

# list 10 lists 2 arrays (4, 5), (5, 5)
