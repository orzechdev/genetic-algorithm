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
    plt.axis([0, 100, 0, 100])

    start(negative_points, positive_points)
    plt.show()
    print_title('Genetic Algorithms end')


def generate_and_save_points():
    # random_points = np.random.randint(100, size=(2, 1000))
    random_points_negative_x = np.random.normal(70, 10, 500)
    random_points_negative_y = np.random.normal(20, 10, 500)
    random_points_positive_x = np.random.normal(20, 10, 500)
    random_points_positive_y = np.random.normal(80, 10, 500)

    random_points_negative = np.concatenate(([random_points_negative_x], [random_points_negative_y]))
    random_points_positive = np.concatenate(([random_points_positive_x], [random_points_positive_y]))

    with open('random_points_negative', 'wb') as fp:
        pickle.dump(random_points_negative, fp)

    with open('random_points_positive', 'wb') as fp:
        pickle.dump(random_points_positive, fp)


def get_negative_points():
    with open('random_points_negative', 'rb') as fp:
        return pickle.load(fp)


def get_positive_points():
    with open('random_points_positive', 'rb') as fp:
        return pickle.load(fp)


main()

