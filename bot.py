import pickle

import numpy as np

import matplotlib.pyplot as plt
import pygad
import pygad.nn
import pygad.gann

from server import Server


gann = pygad.gann.GANN(
    num_solutions=10,
    num_neurons_input=4,
    num_neurons_hidden_layers=[5],
    num_neurons_output=5,
    hidden_activations="relu",
    output_activation="softmax",
)

server = Server()
server.load_state()


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


def fitness_func(solution, sol_idx):
    global gann, server, best_solution, best_solution_fitness

    server.load_state()
    done = False
    i = 0
    while (not done) and i < 60 * 60:
        data = server.get_msg()
        data_inputs = msg_to_nn_inputs(data)
        buttons = pygad.nn.predict(
            last_layer=gann.population_networks[sol_idx],
            data_inputs=data_inputs,
        )
        joypad_msg = buttons_map(buttons)
        server.send_msg(joypad_msg)
        done = (data["player_hp"] == 0) or (data["boss_hp"] == 0)
        i += 1

    print(f"{sol_idx} ", end="")

    fitness = data["player_hp"] - data["boss_hp"]
    if fitness > best_solution_fitness:
        best_solution = gann.population_networks[sol_idx]
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
    population_vectors = pygad.gann.population_as_vectors(
        population_networks=gann.population_networks
    )

    ga = pygad.GA(
        num_generations=200,
        num_parents_mating=2,
        fitness_func=fitness_func,
        initial_population=population_vectors.copy(),
        keep_elitism=1,
        on_generation=callback_generation,
        save_best_solutions=True,
    )

    best_solution = None
    best_solution_fitness = -49
    ga.run()
    ga.plot_fitness(save_dir="fitness.png")
    ga.save("model")
    with open("nn.pkl", "wb") as fp:
        pickle.dump(best_solution, fp)


if __name__ == "__main__":
    bot()
