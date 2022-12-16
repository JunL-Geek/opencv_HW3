import argparse
import os

import numpy as np
import scipy.spatial as sp
from matplotlib import pyplot as plt

from utils import read_train_shapes, show_shape

def align_shapes(data, epsilon, time):
    """
    align shape
    :param data: N x L * 2
    :param epsilon: error to converge
    :param time: epoch to iterate
    :return:
    """
    N = data.shape[0]
    data = data.astype(np.double).reshape(N, -1, 2)

    for epoch in range(time):
        target = np.mean(data, axis=0)
        target = target - np.mean(target, axis=0)
        target = target / np.linalg.norm(target)
        disparity = 0

        for i in range(N):

            data_i = data[i]
            data_i -= np.mean(data_i, axis=0)
            data_i /= np.linalg.norm(data_i)
            M = (data_i.T.dot(target)).T
            U, W, Vt = np.linalg.svd(M)
            R = U.dot(Vt)
            scale = W.sum()
            data_i = np.dot(data_i, R.T) * scale
            data[i] = data_i
            disparity += np.sum(np.square(target - data_i))

        if disparity < epsilon:
            break;

    return data
    # for i in range(N):
    #     mtx1, mtx2, disparity = sp.procrustes(data[i], target)
    #     print(mtx1)
    #     print(mtx2)
    #     print(disparity)


def train(energy_percentage, model_file_path, train_shapes):
    """
    train statistic shape model
    :param energ_percentage: number of eigenvalues preserved
    :param model_file: file to save the output of the model
    :param train_shapes: shapes to be trained
    :return:
    """

    train_shapes = train_shapes.reshape(train_shapes.shape[0], -1)
    mean_shape = np.mean(train_shapes, axis=0)
    cov_mat = np.cov(train_shapes.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    for i in range(1, 11):
        model_minus = mean_shape + 3 * np.sqrt(eigenvalues[-i]) * eigenvectors[:, -i]
        model_plus = mean_shape - 3 * np.sqrt(eigenvalues[-i]) * eigenvectors[:, -i]
        fig = plt.figure(figsize=(11,3))
        fig.suptitle("PCA Model" + str(i))
        plt.subplot(1, 3, 1)
        show_shape(model_minus, c='r')
        plt.subplot(1, 3, 2)
        show_shape(mean_shape, c='g')
        plt.subplot(1, 3, 3)
        show_shape(model_plus, c='b')
        plt.show()

    num_eigenvalues = int(energy_percentage / 100 * train_shapes.shape[0])
    model_minus, model_plus = mean_shape, mean_shape

    for i in range(1, num_eigenvalues + 1):
        model_minus -= 3 * np.sqrt(eigenvalues[-i]) * eigenvectors[:, -i]
        model_plus += 3 * np.sqrt(eigenvalues[-i]) * eigenvectors[:, -i]

    fig = plt.figure(figsize=(11,3))
    fig.suptitle("PCA Model" + str(energy_percentage) + '%')
    plt.subplot(1, 3, 1)
    show_shape(model_minus, c='r')
    plt.subplot(1, 3, 2)
    show_shape(mean_shape, c='g')
    plt.subplot(1, 3, 3)
    show_shape(model_plus, c='b')
    save_file = os.path.join(model_file_path, 'PCA_Model_' + str(energy_percentage) + '%.png')
    if os.path.exists(save_file):
        os.remove(save_file)
    plt.savefig(save_file)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--energy_percentage', default=20, type=int)
    parser.add_argument('--model_path', default=r'.\model', type=str, metavar='PATH')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    shapes_file = r'.\data\shapes\shapes.asf'
    train_shapes = read_train_shapes(shapes_file)
    shapes = align_shapes(train_shapes, 0.001, 50)
    args = parse_args()
    energy_percentage = args.energy_percentage
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    train(energy_percentage, model_path, shapes)

    # shapes_file = r'.\data\shapes\shapes.asf'
    # train_shapes = read_train_shapes(shapes_file)
    # shapes = align_shapes(train_shapes, 0.001, 50)
    # shapes = shapes.reshape(shapes.shape[0], -1)
    # # for i in range(shapes.shape[0]):
    # #     show_shape(shapes[i])
    # model_save_path = r'.\model'
    # if not os.path.exists(model_save_path):
    #     os.mkdir(model_save_path)
    # energy_percentage = 0.2
    # train(energy_percentage, model_save_path, shapes)





