import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point


def histogram(data, STEPS=0.025, d_min=None, d_max=None):
    values = np.arange(data.min() if d_min is None else d_min,
                       (data.max() if d_max is None else d_max) + STEPS, STEPS)
    n, _ = np.histogram(data, values)
    return values, n


def plot_histogram(data, STEPS=0.025, d_min=None, d_max=None):
    spec_color = ['blue', 'red']
    for i in range(len(data[0])):
        block = data[:, i]
        values = np.arange(block.min() if d_min is None else d_min,
                           (block.max() if d_max is None else d_max) + STEPS, STEPS)
        plt.hist(block, values, lw=1, ec="black", fc=spec_color[i], alpha=0.5)
    plt.xlabel('Satisfaction value')
    plt.ylabel('Count')
    plt.legend(['Task'] + ([] if len(data[0]) == 1 else ['Safety']))


def plot_2Dhistogram(data, STEPS=0.025, d_min=None, d_max=None):
    values = np.arange(data.min() if d_min is None else d_min, data.max()
    if d_max is None else d_max + STEPS, STEPS)
    plt.hist2d(data[:, 0], data[:, 1], bins=[values, values])
    plt.xlabel('Satisfaction value task')
    plt.ylabel('Satisfaction value safety')
    plt.colorbar(label='Count')


def signed_dist(poly, point):
    dist = poly.exterior.distance(point)
    return dist if poly.contains(point) else -dist


EPS = 0.25
PT = np.array([0.5, 0.1, 0.2, 0.1])  # x_center, y_center, length/2, height/2
PLACEMAT = Polygon(np.array([PT[:2] + [-PT[2], -PT[3]],
                             PT[:2] + [-PT[2], PT[3]],
                             PT[:2] + [PT[2], PT[3]],
                             PT[:2] + [PT[2], -PT[3]],
                             PT[:2] + [-PT[2], -PT[3]]]))
P_L = PT[:2] - [PT[2], PT[3] / 2]
P_R = PT[:2] + [PT[2], PT[3] / 2]
P_O = PT[:2] + [0, PT[3]]
P_B = PT[:2] - [0, PT[3]]

# orientations
NORTH = np.pi / 2
SOUTH = 3 * np.pi / 2
EAST = 0
WEST = np.pi
ANGLE_EPS = 0.2

leftOf = lambda a, b: b[0] - a[0]
rightOf = lambda a, b: - leftOf(a, b)

belowOf = lambda a, b: b[1] - a[1]
aboveOf = lambda a, b: - belowOf(a, b)

and_ = lambda a, b: np.minimum(a, b)
or_ = lambda a, b: np.maximum(a, b)
not_ = lambda a: -a

closeTo = lambda a, b: EPS - np.linalg.norm(a[:2] - b[:2])
farFrom = lambda a, b: -closeTo(a, b)

# 0 <= d - eps - angle
# angle - eps - d >= 0

onPlaceMat = lambda a: signed_dist(PLACEMAT, Point(a[:2]))
centerAbovePlaceMat = lambda a: and_(closeTo(a, P_O), and_(rightOf(a, P_L), and_(leftOf(a, P_R), aboveOf(a, P_O))))
centerRightPlaceMat = lambda a: and_(closeTo(a, P_R), and_(rightOf(a, P_R), and_(aboveOf(a, P_B), belowOf(a, P_O))))
orientation = lambda a, b: and_((a[-1] - 1 if np.isclose(b, 0) and a[-1] > 0.5 else a[-1]) * 2 * np.pi + ANGLE_EPS - b,
                                b + ANGLE_EPS - (
                                    a[-1] - 1 if np.isclose(b, 0) and a[-1] > 0.5 else a[-1]) * 2 * np.pi)  # /(2*np.pi)
