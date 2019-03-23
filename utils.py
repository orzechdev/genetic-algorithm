import numpy as np
import matplotlib.pyplot as plt
import random

color_start = '\033[34m'
color_end = '\033[0m'


def print_title(text):
    print(color_start + text + color_end)


def plot_polynomial(a, b, c):
    # create 1000 equally spaced points between 0 and 100
    x = np.linspace(-10, 20, 1000)

    # calculate the y value for each element of the x vector
    y = (a * (x ** 2)) + (b * x) + c

    plt.plot(x, y)


def random_one(probability):
    return 0 if random.random() > probability else 1


def unpack_bits(x, num_bits):
    x_shape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2**np.arange(num_bits).reshape([1, num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(x_shape + [num_bits])


def mutate_bits(bits_string, mutation_probability):
    bits_string_mutated = ''
    for c in bits_string:
        should_mutate = random_one(mutation_probability)
        if should_mutate and c == '1':
            bits_string_mutated = bits_string_mutated + '0'
        elif should_mutate and c == '0':
            bits_string_mutated = bits_string_mutated + '1'
        else:
            bits_string_mutated = bits_string_mutated + c
    return bits_string_mutated
