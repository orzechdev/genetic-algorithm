import numpy as np
import pickle
from gen_algorithm import start
from utils import print_title


def main():
    print_title('Genetic Algorithm starts')
    # generate_and_save_points()
    points = get_points()
    print_title('Points')
    print(points)

    start(points)
    print_title('Genetic Algorithms end')


def generate_and_save_points():
    random_points = np.random.randint(100, size=(2, 1000))
    with open('points', 'wb') as fp:
        pickle.dump(random_points, fp)


def get_points():
    with open('points', 'rb') as fp:
        return pickle.load(fp)


main()

