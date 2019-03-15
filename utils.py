import numpy as np
import matplotlib.pyplot as plt

color_start = '\033[34m'
color_end = '\033[0m'


def print_title(text):
    print(color_start + text + color_end)


def plot_polynomial(a, b, c):
    # create 1000 equally spaced points between 0 and 100
    x = np.linspace(0, 100, 1000)

    # calculate the y value for each element of the x vector
    y = (a * (x ** 2)) + (b * x) + c

    plt.plot(x, y)
