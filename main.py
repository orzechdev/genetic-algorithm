import numpy as np
import pickle
import matplotlib.pyplot as plt
from gen_algorithm import start
from utils import print_title


def main():
    print_title('Genetic Algorithm starts')
    # generate_and_save_points()
    negative_points = get_negative_points()
    positive_points = get_positive_points()
    print_title('Negative and positive points')
    print(negative_points)
    print(positive_points)
    plt.plot(negative_points[0], negative_points[1], 'r.')
    plt.plot(positive_points[0], positive_points[1], 'b.')
    plt.axis([-10, 20, -10, 20])

    find_polynomial(2, negative_points, positive_points)

    plt.show()
    print_title('Genetic Algorithms end')


def generate_and_save_points():
    # random_points = np.random.randint(100, size=(2, 1000))
    random_points_negative_x = np.random.normal(14, 2, 20)
    random_points_negative_y = np.random.normal(4, 2, 20)
    random_points_negative_x = np.append(random_points_negative_x, np.random.normal(10, 2, 20))
    random_points_negative_y = np.append(random_points_negative_y, np.random.normal(-2, 2, 20))
    random_points_negative_x = np.append(random_points_negative_x, np.random.normal(3, 2, 20))
    random_points_negative_y = np.append(random_points_negative_y, np.random.normal(-4, 2, 20))

    random_points_positive_x = np.random.normal(0, 2, 20)
    random_points_positive_y = np.random.normal(14, 2, 20)
    random_points_positive_x = np.append(random_points_positive_x, np.random.normal(2, 2, 20))
    random_points_positive_y = np.append(random_points_positive_y, np.random.normal(12, 2, 20))
    random_points_positive_x = np.append(random_points_positive_x, np.random.normal(4, 2, 20))
    random_points_positive_y = np.append(random_points_positive_y, np.random.normal(14, 2, 20))

    random_points_negative = np.concatenate(([random_points_negative_x], [random_points_negative_y]))
    random_points_positive = np.concatenate(([random_points_positive_x], [random_points_positive_y]))

    with open('random_points_negative', 'wb') as fp:
        pickle.dump(random_points_negative, fp)

    with open('random_points_positive', 'wb') as fp:
        pickle.dump(random_points_positive, fp)


def find_polynomial(degree, negative_points, positive_points):
    param_pop_size = 1000  # 2000
    param_crossover_probability = 0.4  # 0.4
    param_mutation_probability = 0.12  # 0.02
    param_generation_number = 10  # 20

    polynomials = {
        2: init_polynomial_pop_degree_two(param_pop_size),
        3: init_polynomial_pop_degree_three(param_pop_size),
        4: init_polynomial_pop_degree_four(param_pop_size),
        5: init_polynomial_pop_degree_five(param_pop_size)
    }

    initial_pop = polynomials.get(degree, lambda: 'Invalid Polynomial Degree')

    start(
        negative_points,
        positive_points,
        y_polynomial,
        initial_pop,
        param_crossover_probability,
        param_mutation_probability,
        param_generation_number
    )


def get_negative_points():
    with open('random_points_negative', 'rb') as fp:
        return pickle.load(fp)


def get_positive_points():
    with open('random_points_positive', 'rb') as fp:
        return pickle.load(fp)


def init_polynomial_pop_degree_two(args_pop_size):
    a_params = np.random.uniform(low=0.01, high=1, size=args_pop_size)
    b_params = np.random.randint(low=-20, high=20, size=args_pop_size)
    c_params = np.random.randint(low=-200, high=200, size=args_pop_size)
    # return np.random.randint(low=1, high=10, size=(2, args_pop_size))
    return np.concatenate(([a_params], [b_params], [c_params]))


def init_polynomial_pop_degree_three(args_pop_size):
    a_params = np.random.uniform(low=0.01, high=1, size=args_pop_size)
    b_params = np.random.uniform(low=0.01, high=1, size=args_pop_size)
    c_params = np.random.randint(low=-20, high=20, size=args_pop_size)
    d_params = np.random.randint(low=-200, high=200, size=args_pop_size)
    # return np.random.randint(low=1, high=10, size=(2, args_pop_size))
    return np.concatenate(([a_params], [b_params], [c_params], [d_params]))


def init_polynomial_pop_degree_four(args_pop_size):
    a_params = np.random.uniform(low=0.01, high=1, size=args_pop_size)
    b_params = np.random.uniform(low=0.01, high=1, size=args_pop_size)
    c_params = np.random.uniform(low=0.01, high=1, size=args_pop_size)
    d_params = np.random.randint(low=-20, high=20, size=args_pop_size)
    e_params = np.random.randint(low=-200, high=200, size=args_pop_size)
    # return np.random.randint(low=1, high=10, size=(2, args_pop_size))
    return np.concatenate(([a_params], [b_params], [c_params], [d_params], [e_params]))


def init_polynomial_pop_degree_five(args_pop_size):
    a_params = np.random.uniform(low=0.01, high=1, size=args_pop_size)
    b_params = np.random.uniform(low=0.01, high=1, size=args_pop_size)
    c_params = np.random.uniform(low=0.01, high=1, size=args_pop_size)
    d_params = np.random.uniform(low=0.01, high=1, size=args_pop_size)
    e_params = np.random.randint(low=-20, high=20, size=args_pop_size)
    f_params = np.random.randint(low=-200, high=200, size=args_pop_size)
    # return np.random.randint(low=1, high=10, size=(2, args_pop_size))
    return np.concatenate(([a_params], [b_params], [c_params], [d_params], [e_params]))


def y_polynomial(x, *argv):
    y = 0
    for idx, arg in enumerate(argv):
        if idx == 0:
            y = arg
        elif idx == 1:
            y = y + (arg * x)
        else:
            y = y + (arg * (x ** idx))

    return y


main()

