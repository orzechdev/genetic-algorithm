import numpy as np
import matplotlib.pyplot as plt
from utils import print_title, plot_polynomial, random_one, unpack_bits, mutate_bits


def start(negative_points, positive_points, y_polynomial, initial_pop, param_crossover_probability, param_mutation_probability, param_generation_number):
    pop_size = initial_pop.shape[1]

    print_title('GA: start - polynomial degree 2')
    print_title('GA: initialise population of functions arguments')
    print(initial_pop)
    print_title('GA: evaluate negative and positive points groups')
    args_number = initial_pop.shape[0]
    args_pop_selected = np.zeros((args_number, 0), dtype=np.float16)
    args_pop_all, args_pop_fitness_values = evaluate(initial_pop, negative_points, positive_points, y_polynomial, args_pop_selected)
    print(args_pop_fitness_values)
    print_title('GA: selection, sort population by fitness values')
    args_pop_selected, best_value, worst_value, mean_value = selection_sort(args_pop_all, args_pop_fitness_values, pop_size)

    print(args_pop_selected)
    print('%.2f\t%.2f\t%.2f' % (best_value.astype(float), mean_value.astype(float), worst_value.astype(float)), end='\n')

    for i in range(param_generation_number - 1):
        # print(i)
        args_pop_crossovered = crossover(args_pop_selected, param_crossover_probability)
        args_pop_mutated = mutation(args_pop_crossovered, param_mutation_probability)
        args_pop_all, args_pop_fitness_values = evaluate(args_pop_mutated, negative_points, positive_points, y_polynomial, args_pop_selected)
        args_pop_selected, best_value, worst_value, mean_value = selection_sort(args_pop_all, args_pop_fitness_values, pop_size)
        print('%.2f\t%.2f\t%.2f' % (best_value.astype(float), mean_value.astype(float), worst_value.astype(float)), end='\n')

    args = args_pop_selected[:, 0][::-1]
    x = np.linspace(-10, 20, 1000)
    y = y_polynomial(x, *args)
    np.set_printoptions(suppress=True)
    plt.plot(x, y, '-b', label=str(np.around(args, decimals=3)[::-1]).strip('[]'))
    plt.legend()

    print('\n')
    print(args_pop_selected)


def evaluate(args_pop, negative_points, positive_points, y_polynomial, args_pop_selected):

    args_pop_fitness_values = np.empty((0, 2), dtype=np.float16)

    args_pop_all = np.insert(args_pop, [1], args_pop_selected, axis=1)

    # print(args_pop)
    # print(args_pop_selected)

    for n in range(0, args_pop_all.shape[1]):
        args = args_pop_all[:, n][::-1]
        # plot_polynomial(a, b, c)
        fitness_value = 0
        for i in range(0, negative_points.shape[1]):
            y_value = y_polynomial(negative_points[0, i], *args)
            # y_polynomial = (a * (negative_points[0, i] ** 2)) + (b * (negative_points[0, i])) + c
            if y_value > negative_points[1, i]:
                fitness_value += 1
        for i in range(0, positive_points.shape[1]):
            y_value = y_polynomial(positive_points[0, i], *args)
            # y_polynomial = (a * (positive_points[0, i] ** 2)) + (b * (positive_points[0, i])) + c
            if y_value < positive_points[1, i]:
                fitness_value += 1
        args_pop_fitness_values = np.append(args_pop_fitness_values, fitness_value)

    # args_pop_evaluated = np.concatenate((args_pop_negative, args_pop_positive), axis=1)
    return args_pop_all, args_pop_fitness_values


def selection_sort(args_pop, args_pop_fitness_values, pop_size_selected):
    args_pop_fitness_values_selected = np.empty((0, 2), dtype=np.float16)

    # print(args_pop)

    args_number = args_pop.shape[0]

    args_pop_selected = np.zeros((args_number, pop_size_selected), dtype=np.float16)
    args_pop_fitness_values_sort_sequence = np.argsort(args_pop_fitness_values)[::-1][:args_pop.shape[1]]
    best_value = args_pop_fitness_values[args_pop_fitness_values_sort_sequence[0]]
    worst_value = args_pop_fitness_values[args_pop_fitness_values_sort_sequence[len(args_pop_fitness_values_sort_sequence) - 1]]
    mean_value = np.mean(args_pop_fitness_values)

    for i in range(0, pop_size_selected):
        selected_i = args_pop_fitness_values_sort_sequence[i]
        for j in range(0, args_number):
            args_pop_selected[j, i] = args_pop[j, selected_i]
        args_pop_fitness_values_selected = np.append(args_pop_fitness_values_selected, args_pop_fitness_values[selected_i])

    # print(args_pop_fitness_values_selected)
    return args_pop_selected, best_value*100/120, worst_value*100/120, mean_value*100/120


def crossover(args_pop_selected, crossover_probability):
    args_number = args_pop_selected.shape[0]
    args_pop_crossovered = np.zeros((args_number, 0), dtype=np.float16)
    # crossover_point = np.uint8(1/2)
    # args_pop_selected_bits = np.empty((3, args_pop.shape[1], 8), dtype=np.uint8)
    # args_pop_selected_bits[0] = unpack_bits(args_pop_selected[0], 8)
    # args_pop_selected_bits[1] = unpack_bits(args_pop_selected[1], 8)
    # args_pop_selected_bits[2] = unpack_bits(args_pop_selected[2], 8)
    # print(args_pop_selected_bits)

    # individual_current = 0

    for i in range(0, args_pop_selected.shape[1]):
        for j in range(0, args_pop_selected.shape[1]):
            if random_one(crossover_probability):
                args = np.zeros((args_number, 1), dtype=np.float16)
                for k in range(0, args_number):
                    if k < args_number - 2:
                        a_i_bits = np.binary_repr(int(args_pop_selected[k, i]*1000))
                        a_j_bits = np.binary_repr(int(args_pop_selected[k, j]*1000))
                        # print(a_i_bits + ' - ' + a_j_bits)
                        a_i_bits_len = len(a_i_bits)
                        a_i_bits_half = a_i_bits if a_i_bits_len == 1 else a_i_bits[:int(a_i_bits_len/2)]
                        a_j_bits_len = len(a_j_bits)
                        a_j_bits_half = a_j_bits if a_j_bits_len == 1 else a_j_bits[int(a_j_bits_len - (a_j_bits_len/2)):]
                        # print(a_i_bits_half + ' + ' + a_j_bits_half)
                        bits_crossed = a_i_bits_half + a_j_bits_half
                        # args_pop_crossovered[k, individual_current] = int(bits_crossed, 2)/100
                        args[k, 0] = int(bits_crossed, 2)/1000
                        # print(str(i) + ',' + str(j) + ' -> ' + bits_crossed + ' -> ' + args_pop_selected[k, i].astype(str) + ' + ' + args_pop_selected[k, j].astype(str) + ' -> ' + args[k, 0].astype(str))
                    else:
                        p_i_bits = np.binary_repr(int(args_pop_selected[k, i]))
                        p_j_bits = np.binary_repr(int(args_pop_selected[k, j]))
                        p_i_bits_len = len(p_i_bits)
                        p_i_bits_half = p_i_bits if p_i_bits_len == 1 else p_i_bits[:int(p_i_bits_len/2)]
                        p_j_bits_len = len(p_j_bits)
                        p_j_bits_half = p_j_bits if p_j_bits_len == 1 else p_j_bits[int(p_j_bits_len - (p_j_bits_len/2)):]
                        # print(p_i_bits_half + ' - ' + p_j_bits_half)
                        bits_crossed = p_i_bits_half + p_j_bits_half
                        # args_pop_crossovered[k, individual_current] = int(bits_crossed, 2)
                        args[k, 0] = int(bits_crossed, 2)
                    # if args_pop_crossovered[k, individual_current] == 0:
                    #     print('crossover: ' + str(k) + ', ' + args_pop_selected[k, i].astype(str) + ' to ' + args_pop_crossovered[k, individual_current].astype(str))
                if args_pop_crossovered.shape[1] == 0:
                    args_pop_crossovered = args
                else:
                    args_pop_crossovered = np.insert(args_pop_crossovered, [1], args, axis=1)
                # print(args_pop_crossovered)
                # individual_current += 1
                # break

    #
    # IF THRE IS AN ERROR WITH 0 AS FIRST PARAMETER TRY TO PRINT args_pop_crossovered
    #

    # print(args_pop_crossovered)

    return args_pop_crossovered


def mutation(args_pop, mutation_probability):
    args_number = args_pop.shape[0]
    args_pop_mutated = np.zeros((args_number, args_pop.shape[1]), dtype=np.float16)

    for i in range(0, args_pop.shape[1]):
        for k in range(0, args_number):
            if k < args_number - 2:
                a_i_bits = np.binary_repr(int(args_pop[k, i] * 1000))
                a_i_bits_mutated = a_i_bits[0] + mutate_bits(a_i_bits[1:], mutation_probability)
                args_pop_mutated[k, i] = int(a_i_bits_mutated, 2) / 1000
            else:
                p_i_bits = np.binary_repr(int(args_pop[k, i]))
                p_i_bits_mutated = mutate_bits(p_i_bits, mutation_probability)
                args_pop_mutated[k, i] = int(p_i_bits_mutated, 2)
            # if args_pop_mutated[k, i] == 0:
            #     print('mutation: ' + str(k) + ', ' + args_pop[k, i].astype(str) + ' to ' + args_pop_mutated[k, i].astype(str))

    return args_pop_mutated
