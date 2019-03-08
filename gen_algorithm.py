import numpy as np


def start():
    print('GA: start')
    population = initialise(population_size=50)
    print(population)


def initialise(population_size):
    print('GA: initialise')
    return np.random.randint(100, size=(2, population_size))


def evaluate():
    pass


def selection():
    pass


def crossover():
    pass


def mutation():
    pass

