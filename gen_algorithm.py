import numpy as np
import matplotlib.pyplot as plt
from utils import print_title, plot_polynomial, random_one, unpack_bits, mutate_bits


def start(negative_points, positive_points):
    param_population_size = 1000  # 2000
    param_crossover_probability = 0.4  # 0.4
    param_mutation_probability = 0.12  # 0.02
    param_generation_number = 20  # 20

    print_title('GA: start - polynomial degree 2')
    print_title('GA: initialise population of functions arguments')
    args_population = initialise(args_population_size=param_population_size)
    print(args_population)
    print_title('GA: evaluate negative and positive points groups')
    args_population_fitness_values = evaluate(args_population, negative_points, positive_points)
    print(args_population_fitness_values)
    print_title('GA: selection, sort population by fitness values')
    args_population_selected, best_value, worst_value, mean_value = selection_sort(args_population, args_population_fitness_values)

    print(args_population_selected)
    print('%.2f\t%.2f\t%.2f' % (best_value.astype(float), mean_value.astype(float), worst_value.astype(float)), end='\n')

    for i in range(param_generation_number - 1):
        args_population_crossovered = crossover(args_population, args_population_selected, param_crossover_probability)
        args_population_mutated = mutation(args_population_crossovered, param_mutation_probability)
        args_population_fitness_values = evaluate(args_population_mutated, negative_points, positive_points)
        args_population_selected, best_value, worst_value, mean_value = selection_sort(args_population_mutated, args_population_fitness_values)
        print('%.2f\t%.2f\t%.2f' % (best_value.astype(float), mean_value.astype(float), worst_value.astype(float)), end='\n')

    a = args_population_selected[0, 0]
    b = args_population_selected[1, 0]
    c = args_population_selected[2, 0]
    plot_polynomial(a, b, c)

    print('\n')
    print(args_population_selected)


def initialise(args_population_size):
    # Initialize the initial population of parameters of functions
    # [[a1,a2,a3,...,an],[b1,b2,b3,...,bn]]
    # 1 <= a <= 10
    # 0 <= b <= -50
    # 0 <= c <= 100
    a_params = np.random.uniform(low=0.01, high=1, size=args_population_size)
    b_params = np.random.randint(low=-20, high=20, size=args_population_size)
    c_params = np.random.randint(low=-200, high=200, size=args_population_size)
    # return np.random.randint(low=1, high=10, size=(2, args_population_size))
    return np.concatenate(([a_params], [b_params], [c_params]))


def evaluate(args_population, negative_points, positive_points):

    args_population_fitness_values = np.empty((0, 2))

    for n in range(0, args_population.shape[1]):
        a = args_population[0, n]
        b = args_population[1, n]
        c = args_population[2, n]
        # plot_polynomial(a, b, c)
        fitness_value = 0
        for i in range(0, negative_points.shape[1]):
            y_polynomial = (a * (negative_points[0, i] ** 2)) + (b * (negative_points[0, i])) + c
            if y_polynomial > negative_points[1, i]:
                fitness_value += 1
        for i in range(0, positive_points.shape[1]):
            y_polynomial = (a * (positive_points[0, i] ** 2)) + (b * (positive_points[0, i])) + c
            if y_polynomial < positive_points[1, i]:
                fitness_value += 1
        args_population_fitness_values = np.append(args_population_fitness_values, fitness_value)

    # args_population_evaluated = np.concatenate((args_population_negative, args_population_positive), axis=1)
    return args_population_fitness_values


def selection_sort(args_pop, args_pop_fitness_values):
    args_pop_fitness_values_selected = np.empty((0, 2))

    args_pop_selected = np.zeros((3, args_pop.shape[1]), dtype=np.float16)
    args_pop_fitness_values_sort_sequence = np.argsort(args_pop_fitness_values)[::-1][:args_pop.shape[1]]
    best_value = args_pop_fitness_values[args_pop_fitness_values_sort_sequence[0]]
    worst_value = args_pop_fitness_values[args_pop_fitness_values_sort_sequence[len(args_pop_fitness_values_sort_sequence) - 1]]
    mean_value = np.mean(args_pop_fitness_values)

    for i in range(0, args_pop.shape[1]):
        selected_i = args_pop_fitness_values_sort_sequence[i]
        args_pop_selected[0, i] = args_pop[0, selected_i]
        args_pop_selected[1, i] = args_pop[1, selected_i]
        args_pop_selected[2, i] = args_pop[2, selected_i]
        args_pop_fitness_values_selected = np.append(args_pop_fitness_values_selected, args_pop_fitness_values[selected_i])

    # print(args_pop_fitness_values_selected)
    return args_pop_selected, best_value*100/120, worst_value*100/120, mean_value*100/120


def crossover(args_pop, args_pop_selected, crossover_probability):
    args_pop_crossovered = np.zeros((3, args_pop.shape[1]))
    # crossover_point = np.uint8(1/2)
    # args_pop_selected_bits = np.empty((3, args_pop.shape[1], 8), dtype=np.uint8)
    # args_pop_selected_bits[0] = unpack_bits(args_pop_selected[0], 8)
    # args_pop_selected_bits[1] = unpack_bits(args_pop_selected[1], 8)
    # args_pop_selected_bits[2] = unpack_bits(args_pop_selected[2], 8)
    # print(args_pop_selected_bits)

    crossovered_pop_size = 0

    while crossovered_pop_size < args_pop.shape[1]:
        for i in range(0, args_pop.shape[1]):
            for j in range(0, args_pop.shape[1]):
                if random_one(crossover_probability):
                    a_i_bits = np.binary_repr(int(args_pop_selected[0, i]*100))
                    a_j_bits = np.binary_repr(int(args_pop_selected[0, j]*100))
                    args_pop_crossovered[0, i] = int(a_i_bits[:len(a_i_bits)-1] + a_j_bits[len(a_j_bits)-1:], 2)/100
                    b_i_bits = np.binary_repr(int(args_pop_selected[1, i]))
                    b_j_bits = np.binary_repr(int(args_pop_selected[1, j]))
                    args_pop_crossovered[1, i] = int(b_i_bits[:len(b_i_bits)-1] + b_j_bits[len(b_j_bits)-1:], 2)
                    c_i_bits = np.binary_repr(int(args_pop_selected[2, i]))
                    c_j_bits = np.binary_repr(int(args_pop_selected[2, j]))
                    args_pop_crossovered[2, i] = int(c_j_bits[:len(c_i_bits)-1] + c_j_bits[len(c_j_bits)-1:], 2)
                    crossovered_pop_size += 1
                    break
                if crossovered_pop_size >= args_pop.shape[1]:
                    break
            if crossovered_pop_size >= args_pop.shape[1]:
                break

    return args_pop_crossovered


def mutation(args_pop, mutation_probability):
    args_pop_mutated = np.zeros((3, args_pop.shape[1]))

    for i in range(0, args_pop.shape[1]):
        a_i_bits = np.binary_repr(int(args_pop[0, i] * 100))
        a_i_bits_mutated = a_i_bits[0] + mutate_bits(a_i_bits[1:], mutation_probability)
        args_pop_mutated[0, i] = int(a_i_bits_mutated, 2) / 100
        b_i_bits = np.binary_repr(int(args_pop[1, i]))
        b_i_bits_mutated = mutate_bits(b_i_bits, mutation_probability)
        args_pop_mutated[1, i] = int(b_i_bits_mutated, 2)
        c_i_bits = np.binary_repr(int(args_pop[2, i]))
        c_i_bits_mutated = mutate_bits(c_i_bits, mutation_probability)
        args_pop_mutated[2, i] = int(c_i_bits_mutated, 2)

    return args_pop_mutated
