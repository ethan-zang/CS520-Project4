import imageio
import numpy as np
import os

from PIL import Image
from typing import Tuple

def retrieve_pixels() -> Tuple[np.array, np.array]:
    """
    Retrieves rgb and grayscale values from image.
    Returns: Tuple[np.array, np.array] which represents the rgb pixel array and grayscale pixel array

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

def main():
    rgb, gray = retrieve_pixels()
    print(rgb)
    print(gray)


if __name__ == '__main__':
    main()