import numpy as np

import matplotlib.pyplot as plt
import pygad
import pygad.nn
import pygad.gann

from server import Server


gann = pygad.gann.GANN(
    num_solutions=2,
    num_neurons_input=4,
    num_neurons_hidden_layers=[5],
    num_neurons_output=5,
    hidden_activations=["relu"],
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


def fitness_func(solution, sol_idx):
    global gann, server

    server.load_state()
    done = False
    i = 0
    while (not done) and i < 60 * 60:
        data = server.get_msg()
        data_inputs = data.copy()
        data_inputs.pop("player_hp")
        data_inputs.pop("boss_hp")
        data_inputs = np.array([[*data_inputs.values()]])
        buttons = pygad.nn.predict(
            last_layer=gann.population_networks[sol_idx],
            data_inputs=data_inputs,
        )
        joypad_msg = buttons_map(buttons)
        server.send_msg(joypad_msg)
        done = (data["player_hp"] == 0) or (data["boss_hp"] == 0)
        i += 1

    fitness = data["player_hp"] - data["boss_hp"]
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

    print(f"Generation = {ga.generations_completed}")
    print(f"Fitness    = {ga.best_solution()[1]}")


def bot():
    population_vectors = pygad.gann.population_as_vectors(
        population_networks=gann.population_networks
    )

    ga = pygad.GA(
        num_generations=10,
        num_parents_mating=2,
        initial_population=population_vectors.copy(),
        fitness_func=fitness_func,
        mutation_percent_genes=10,
        on_generation=callback_generation,
    )

    ga.run()

    ga.plot_fitness()
    plt.savefig("test.png")

    solution, solution_fitness, solution_idx = ga.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution: {solution_idx}")


if __name__ == "__main__":
    bot()
