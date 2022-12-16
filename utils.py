

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def read_train_shapes(file):
    train_shapes = []
    shapes = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            train_shapes.append(line)
    shapes = np.array(train_shapes, dtype=np.double).T
    num_points =shapes.shape[1] // 2
    train_shapes = np.zeros(shapes.shape)
    for i in range(num_points):
        train_shapes[:, 2 * i] = shapes[:, i]
        train_shapes[:, 2 * i + 1] = shapes[:, i + num_points]
    return train_shapes



def show_shape(shape, c='g'):
    shape = shape.reshape(-1, 2)
    shape_x = shape[:, 0]
    shape_y = shape[:, 1]
    tck, u = interpolate.splprep([shape_x, shape_y], k=3, s=0)
    u = np.linspace(0, 1, num=150, endpoint = False)
    interp_points = interpolate.splev(u, tck)
    shape_x = interp_points[0]
    shape_y = interp_points[1]
    plt.plot(shape_x * 256, shape_y * 256, '-', color=c)
    # plt.axis('off')


if __name__=='__main__':
    file = r'.\data\shapes\shapes.asf'
    train_shapes = read_train_shapes(file)
    print(train_shapes.shape)
    show_shape(train_shapes[0])
    plt.show()