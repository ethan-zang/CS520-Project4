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
from queue import PriorityQueue, Queue


def retrieve_pixels() -> Tuple[np.array, np.array]:
    """
    Retrieves rgb and grayscale values from image.
    Returns:
        Tuple[np.array, np.array] which represents the rgb pixel array and grayscale pixel array
    """

    # Get file path to image
    root_dir = os.getcwd()
    file_path = os.path.join(root_dir, 'flag.jpg')

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


def calculate_gray_score(gray: np.array, i: int, j: int) -> int:
    """
    Calculate the 'grayness' of a 3x3 grid
    :param gray: array of gray pixels
    :param i: x coordinate of pixel
    :param j: y coordinate of pixel
    :return: gray score, the sum of the gray values of each pixel in the grid
    """
    score = 0

    for x in range(-1, 2):
        for y in range(-1, 2):

            score += int(gray[i + x][j + y])

            # Prioritize center cell
            # if x == 0 and y == 0:
            #     score += 5*int(gray[i + x][j + y])
            #
            # else:
            #     score += int(gray[i + x][j + y])

    return score

def get_similar_gray_patches(gray: np.array, left_gray_scores_dict: Dict[int, List[Tuple[int, int]]], right_i: int, right_j: int) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Retrieve the 6 most similar gray patches on the left hand side
    :param gray: array of gray pixels
    :param left_gray_scores_dict: gray scores for left hand side
    :param right_i: x coordinate of right pixel
    :param right_j: y coordinate of right pixel
    :return: List of the 6 most similar gray patches
    """
    # gray_patch_queue = PriorityQueue()
    right_gray_score = calculate_gray_score(gray, right_i, right_j)
    new_score_key_up = right_gray_score
    new_score_key_down = right_gray_score

    # Retrieve 6 similar patches from dictionary
    similar_gray_patch_coordinates = left_gray_scores_dict.get(right_gray_score)
    if similar_gray_patch_coordinates is None:
        similar_gray_patch_coordinates = []
    similar_gray_patches = [(0, x) for x in similar_gray_patch_coordinates]
    # print("similar patches", len(similar_gray_patches), (right_i, right_j))

    # Add more coordinates until you hit 6
    while len(similar_gray_patches) < 6:

        new_score_key_up += 1
        new_score_key_down -= 1

        # Go one key up
        similar_gray_patch_up_coordinates = left_gray_scores_dict.get(new_score_key_up)
        if similar_gray_patch_up_coordinates is None:
            similar_gray_patch_up_coordinates = []
        similar_gray_patches_up = [(new_score_key_up-right_gray_score, x) for x in similar_gray_patch_up_coordinates]
        similar_gray_patches.extend(similar_gray_patches_up)

        # Go one key down
        similar_gray_patch_down_coordinates = left_gray_scores_dict.get(new_score_key_down)
        if similar_gray_patch_down_coordinates is None:
            similar_gray_patch_down_coordinates = []
        similar_gray_patches_down = [(right_gray_score - new_score_key_down, x) for x in similar_gray_patch_down_coordinates]
        similar_gray_patches.extend(similar_gray_patches_down)
        # print("papa")

    return similar_gray_patches[:6]


    # Iterate through left half of gray, skipping edges
    # for i in range(1, gray.shape[0]-1):
    #     for j in range(1, int(gray.shape[1]/2) - 1):
    #         left_gray_score = calculate_gray_score(gray, i, j)
    #         score_difference = abs(right_gray_score-left_gray_score)
    #
    #         # Add gray patch to priority queue if there's space
    #         if gray_patch_queue.qsize() < 6:
    #             gray_patch_queue.put((-1*score_difference, (i, j)))
    #
    #         # Check to see if current gray patch is more similar; if so add to queue
    #         else:
    #             lowest_similarity_patch = gray_patch_queue.get()
    #
    #             # Current patch is more similar
    #             if -1*lowest_similarity_patch[0] > score_difference:
    #                 gray_patch_queue.put((-1*score_difference, (i, j)))
    #
    #             # If not, put the lowest similarity patch back
    #             else:
    #                 gray_patch_queue.put(lowest_similarity_patch)

    # return list(gray_patch_queue.queue)


def get_left_gray_scores(gray: np.array) -> Dict[int, List[Tuple[int, int]]]:
    """
    Get a list of all the gray scores for the left side of the image
    :param gray: array of gray pixels
    :return: Dictionary with gray score as the key and coordinates as the value
    """
    left_gray_scores_dict = {}

    # Iterate through left half of gray, skipping edges
    for i in range(1, gray.shape[0]-1):
        for j in range(1, int(gray.shape[1]/2) - 1):
            left_gray_score = calculate_gray_score(gray, i, j)

            if left_gray_score in left_gray_scores_dict:
                curr_list = left_gray_scores_dict.get(left_gray_score)
                curr_list.append((i, j))
                left_gray_scores_dict.update({left_gray_score: curr_list})
            else:
                left_gray_scores_dict[left_gray_score] = [(i, j)]

    return left_gray_scores_dict



def color_right_side(gray: np.array, new_rgb: np.array, representative_colors: np.array, pixel_color_array: np.array) -> np.array:
    """
    Color the right side of the grayscale image
    :param gray: array of gray pixels
    :param new_rgb: array that will be colored
    :param representative_colors: 5 colors that will color the image
    :param pixel_color_array: the current colors for each pixel
    :return: a fully colored rgb image
    """
    print("starting right side coloring")

    num_rows = new_rgb.shape[0]
    num_cols = new_rgb.shape[1]

    # similar_gray_patch_queue = None

    left_gray_scores_dict = get_left_gray_scores(gray)
    print(left_gray_scores_dict)

    # Fill in right half of new_rgb with new colors, ignoring edges
    for i in range(1, num_rows-1):
        for j in range(int(num_cols / 2), num_cols-1):

            # Retrieve most 6 most similar patch centers and their similarity scores
            similar_gray_patches = get_similar_gray_patches(gray, left_gray_scores_dict, i, j)
            # print("retrieving 6 patches")
            # print(similar_gray_patches)

            # Find representative color for each patch and add patch to
            patches_for_each_color: List[List[Tuple[int, Tuple[int, int]]]] = [[] for _ in range(5)]
            max_color_frequency = 0
            for patch in similar_gray_patches:
                color_index = int(pixel_color_array[(patch[1][0] * num_cols + patch[1][1])][3])
                # color_index = new_rgb[patch[1][0]][patch[1][1]][3]
                # color_counts[color_index] += 1

                patches_for_each_color[color_index].append(patch)

                if len(patches_for_each_color[color_index]) > max_color_frequency:
                    max_color_frequency = len(patches_for_each_color[color_index])

            # print("patches for each color", patches_for_each_color, (i, j))
            # Retrieve all color indices with that max frequency
            most_frequent_color_indices = []
            for x in range(len(patches_for_each_color)):
                if len(patches_for_each_color[x]) == max_color_frequency:
                    most_frequent_color_indices.append(x)

            # If there is a most represented color, make the corresponding pixel that color
            if len(most_frequent_color_indices) == 1:
                new_rgb[i][j] = representative_colors[most_frequent_color_indices[0]]
                # print("no tie", (i, j), representative_colors[most_frequent_color_indices[0]])

            # Otherwise, break ties based on similarity score
            else:
                # print("there is a tie", (i, j))
                potential_patches = []

                # Put all patches that map to the most represented colors into potential_patches
                for x in range(len(patches_for_each_color)):
                    if len(patches_for_each_color[x]) == max_color_frequency:
                        for patch in patches_for_each_color[x]:
                            potential_patches.append(patch)

                # Select the color that is mapped to the most similar patch
                best_similarity_score = 100000
                most_similar_patch = None
                for patch in potential_patches:
                    if patch[0] < best_similarity_score:
                        best_similarity_score = patch[0]
                        most_similar_patch = patch

                # Make the original pixel the same color as that of the most similar patch
                new_rgb[i][j] = new_rgb[most_similar_patch[1][0]][most_similar_patch[1][1]]

    return new_rgb


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
            # print(color_index)
            new_rgb[i][j] = representative_colors[color_index]

    np.set_printoptions(threshold=5)
    print(rgb)
    print(new_rgb)
    plt.imshow(new_rgb.astype('uint8'))
    plt.show()

    # Color right hand side
    newest_rgb = color_right_side(gray, new_rgb, representative_colors, pixel_color_array)
    plt.imshow(newest_rgb.astype('uint8'))
    plt.show()

    # print(gray)
    # print(gray.shape)
    # plt.imshow(gray, cmap='gray')
    # plt.show()

    print('goodbye world')


if __name__ == '__main__':
    main()