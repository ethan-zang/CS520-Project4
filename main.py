import imageio
from typing import List
import numpy as np
import os
import random
import sys
import math

from PIL import Image
from typing import Tuple, Dict

import matplotlib.pyplot as plt


def retrieve_pixels() -> Tuple[np.array, np.array]:
    """
    Retrieves rgb and grayscale values from image.
    Returns:
        Tuple[np.array, np.array] which represents the rgb pixel array and grayscale pixel array
    """

    # Get file path to image
    root_dir = os.getcwd()
    file_path = os.path.join(root_dir, 'image.jpg')

    # Open image
    im = Image.open(file_path)
    image = (im.copy())
    im.close()

    # Convert to gray scale
    rgb = np.array(image)
    rgb = rgb[..., :3]
    gray_list = []

    for j in range(len(rgb)):
        for k in range(len(rgb[j])):
            gray_list.append(0.21 * int(rgb[j][k][0]) + 0.72 * int(rgb[j][k][1]) + 0.07 * int(rgb[j][k][2]))

    gray = np.reshape(gray_list, (-1, rgb.shape[1]))
    return rgb, gray


def cluster_pixels(rgb: np.array) -> Tuple[np.array, np.array]:
    """
    Obtain the 5 representative colors of the image through k-means clustering
    :param rgb: np.array of colored image pixels
    :return: np.array of the 5 representative colors
    """

    # Turn into 2D numpy array
    flattened_rgb = rgb.reshape(rgb.shape[0]*rgb.shape[1], rgb.shape[2])

    # df = pd.DataFrame(data=flattened_rgb)
    # print(df)

    # Obtain initial centroids
    centroids = initialize_centroids(flattened_rgb)
    print("initial centroids", centroids)

    # Recalculate centroid 10 times
    for i in range(3):
        centroids = calculate_new_centroids(centroids, flattened_rgb)
        print("Iteration: ", i)
        print(centroids)

    # Assign pixels to a color
    np.set_printoptions(threshold=sys.maxsize)
    pixel_color_array = assign_pixels_to_color(centroids, flattened_rgb)
    unique, counts = np.unique(pixel_color_array[:, 3], return_counts=True)
    print('Number of pixels per color: ', dict(zip(unique, counts)))

    # Group all pixels by color
    color_dict = {}
    for i in range(len(centroids)):
        color_dict['color_' + str(i)] = pixel_color_array[np.where(pixel_color_array[:, 3] == i)]

    # Plot centroids and pixels according to color they correspond to
    ax = plt.axes(projection='3d')
    for i in range(len(centroids)):
        ax.scatter3D(color_dict['color_' + str(i)][:, 0], color_dict['color_' + str(i)][:, 1], color_dict['color_' + str(i)][:, 2], alpha=0.1, color=centroids[i]/np.array([[255.0, 255.0, 255.0]]))
    ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='black')
    plt.show()

    return centroids.astype('uint8'), pixel_color_array


def assign_pixels_to_color(centroids: np.array, flattened_rgb: np.array) -> np.array:
    """
    Assign all the pixels to a color
    Args:
        centroids: np.array of all centroids
        flattened_rgb: np.array of all pxels

    Returns:
        np.array representing flattened_rgb appended with a column representing the color assignment
    """
    pixel_colors = np.zeros(shape=(len(flattened_rgb), 1))
    index = 0

    for pixel in flattened_rgb:

        min_distance = 255 * math.sqrt(3) + 1
        min_centroid_i = 0

        # Determine centroid that pixel is closest to
        for i in range(len(centroids)):
            curr_distance = calculate_distance(pixel, centroids[i])
            if curr_distance < min_distance:
                min_centroid_i = i
                min_distance = curr_distance

        # Assign the pixel to a color
        pixel_colors[index] = min_centroid_i
        index += 1

    pixel_color_array = np.concatenate((flattened_rgb, pixel_colors), 1)
    return pixel_color_array


def calculate_new_centroids(centroids: np.array, flattened_rgb: np.array) -> np.array:
    """
    Calculate the new centroids given the existing centroids and colored pixels
    Args:
        centroids: np.array of shape (5, 3)
        flattened_rgb: pixels of colored image

    Returns:
        newly calculated pixels, np.array of shape (5,3)
    """

    new_centroids = np.empty(shape=(len(centroids), 3))
    element_count = np.zeros(shape=len(centroids))

    for pixel in flattened_rgb:

        min_distance = 255 * math.sqrt(3) + 1
        min_centroid_i = 0

        # Determine centroid that pixel is closest to
        for i in range(len(centroids)):
            curr_distance = calculate_distance(pixel, centroids[i])
            if curr_distance < min_distance:
                min_centroid_i = i
                min_distance = curr_distance

        # Add the pixel values to the corresponding centroid index
        new_centroids[min_centroid_i] += pixel
        element_count[min_centroid_i] += 1

    print("centroid sums before dividing", new_centroids)
    print("element count: ", element_count)

    # Divide the sum of the pixel values for each pixel for each centroid by the number of pixels for that centroid
    for i in range(len(centroids)):
        new_centroids[i] /= element_count[i]

    return new_centroids.astype(int)


def initialize_centroids(flattened_rgb: np.array) -> np.array:
    """
    Randomly choose 5 points as initial centroids
    Args:
        flattened_rgb: array of pixels from the colored image

    Returns:
        np.array of shape (5, 3) with the 5 centroids
    """

    # Generate 5 random numbers from 0 to # of pixels
    random_nums = random.sample(range(0, len(flattened_rgb)), 5)

    initial_centroids = []

    # Retrieve pixels using random numbers as indices
    for i in random_nums:
        initial_centroids.append(flattened_rgb[i])

    return np.array(initial_centroids)


def calculate_distance(curr_pixel: np.array, centroid: np.array) -> float:
    """
    Calculate the distance between a pixel and the centroid being compared
    Args:
        curr_pixel: np.array of shape 3
        centroid: np.array of shape 3

    Returns:
        Euclidean distance between the 2 cells
    """

    distance = 0

    for i in range(3):
        distance += (int(curr_pixel[i]) - int(centroid[i]))**2

    distance = math.sqrt(distance)

    return distance


def main():
    print('hello world')
    rgb, gray = retrieve_pixels()

    representative_colors, pixel_color_array = cluster_pixels(rgb)

    num_rows = rgb.shape[0]
    num_cols = rgb.shape[1]
    new_rgb = np.zeros(shape=(num_rows, num_cols, rgb.shape[2]))

    # Fill in left half of new_rgb with new colors
    for i in range(num_rows):
        for j in range(0, int(num_cols/2)):
            color_index = int(pixel_color_array[(i * num_cols + j)][3])
            print(color_index)
            new_rgb[i][j] = representative_colors[color_index]

    np.set_printoptions(threshold=5)
    print(rgb)
    print(new_rgb)
    plt.imshow(new_rgb.astype('uint8'))
    plt.show()

    # print(gray)
    # print(gray.shape)
    # plt.imshow(gray)
    # plt.show()

    print('goodbye world')


if __name__ == '__main__':
    main()