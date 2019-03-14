import numpy as np
from utils import print_title


def start(points):
    print_title('GA: start - polynomial degree 2')
    print_title('GA: initialise population of functions arguments')
    args_population = initialise(args_population_size=50)
    print(args_population)
    print_title('GA: evaluate negative and positive points groups')
    args_population_evaluated = evaluate(args_population, points)
    print(args_population_evaluated)


def initialise(args_population_size):
    # Initialize the initial population of parameters of functions
    # [[a1,a2,a3,...,an],[b1,b2,b3,...,bn]]
    return np.random.randint(10, size=(2, args_population_size))


def evaluate(args_population, points):
    # Evaluate to which group (negative or positive) assign each point
    # [[[x1,y1],[x2,y2],...,[xn,yn]],[[x3,y3],[x4,y4],...,[xn,yn]]]
    args_population_evaluated = []

    args_population_negative = np.empty((0, 2))
    args_population_positive = np.empty((0, 2))

    for n in range(0, args_population.shape[1]):
        a = args_population[0, n]
        b = args_population[1, n]
        for i in range(0, points.shape[1]):
            y_polynomial = a*(points[0, i]**2) + b*(points[0, i])
            if y_polynomial > points[1, i]:
                args_population_negative = np.concatenate((args_population_negative, [[points[0, i], points[1, i]]]))
            else:
                args_population_positive = np.concatenate((args_population_positive, [[points[0, i], points[1, i]]]))
            # x_arr = np.roots(a, b, -points[1, i])
        print('Negative points: ' + f'{args_population_negative.shape[0]}')
        print('Positive points: ' + f'{args_population_positive.shape[0]}')
        args_population_evaluated.append([args_population_negative, args_population_positive])
        args_population_negative = np.empty((0, 2))
        args_population_positive = np.empty((0, 2))

    # args_population_evaluated = np.concatenate((args_population_negative, args_population_positive), axis=1)
    return args_population_evaluated


def selection():
    pass


def crossover():
    pass


def mutation():
    pass

