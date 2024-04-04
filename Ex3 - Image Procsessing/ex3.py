import numpy as np
import cv2

MAX_INTENSITY = 255


def save_image_to_file(image, output_name):
    """
    Saves an image to the disk.

    Parameters:
    - image: numpy.ndarray, the image to be saved.
    - output_name: str, the name of the output file including its file extension (e.g., 'image.jpg').

    Returns:
    None, the image is saved on the disk.
    """
    cv2.imwrite("Exercise Outputs/" + output_name + ".jpg", image)


def read_image_from_file(image_path):
    """
       Reads an image from the specified path.

       Parameters:
       - image_path: str, path to the image file.

       Returns:
       - numpy.ndarray, the image read from the disk.
       """
    return cv2.imread(image_path)


def calculate_gaussian_pyramid(image, levels):
    """
    Calculates the Gaussian pyramid for an image for a specified number of levels.

    Parameters:
    - image: numpy.ndarray, the input image.
    - levels: int, the number of levels in the pyramid.

    Returns:
    - list of numpy.ndarray, the Gaussian pyramid of the image.
    """
    # OpenCV's functions may return results in float32, so convert to float32 to avoid errors.
    # Also, for image processing, the precision of float32 is often sufficient. The extra precision of float64
    # is usually unnecessary for the visual detail represented in images.
    gaussian_pyramid = [np.float32(image)]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)  # blur and downsample
        gaussian_pyramid.append(np.float32(image))
    return gaussian_pyramid


def calculate_laplacian_pyramid(gaussian_pyramid, levels):
    """
   Calculates the Laplacian pyramid from a Gaussian pyramid.

   Parameters:
   - gaussian_pyramid: list of numpy.ndarray, the Gaussian pyramid of the image.
   - levels: int, the number of levels in the Gaussian pyramid.

   Returns:
   - list of numpy.ndarray, the Laplacian pyramid of the image.
   """
    laplacian_pyramid = []
    for i in range(levels - 1, 0, -1):
        # Calculate the size of the next level
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        # Expand the current level
        expended_gaussian_in_following_level = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        gaussian_in_current_level = gaussian_pyramid[i - 1]
        # Subtract the expanded level from the next higher level to get the Laplacian
        laplacian = np.subtract(gaussian_in_current_level, expended_gaussian_in_following_level)
        # Insert at the beginning to build the pyramid in the correct order
        laplacian_pyramid.insert(0, laplacian)

    # Add the smallest level of the Gaussian pyramid at the end of the Laplacian pyramid
    laplacian_pyramid.append(gaussian_pyramid[levels - 1])

    return laplacian_pyramid


def sum_pyramid_levels(laplacian_pyramid):
    """
    Reconstructs an image from its Laplacian pyramid.

    Parameters:
    - laplacian_pyramid: list of numpy.ndarray, the Laplacian pyramid of the image.

    Returns:
    - numpy.ndarray, the reconstructed image.
    """

    # Start with the largest level of the Laplacian pyramid
    blended_image = laplacian_pyramid[-1]

    # Loop through the pyramid in reverse order (starting from the second-largest level)
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # Up sample the current reconstructed image to the size of the next level to be added
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        expanded_laplacian = cv2.pyrUp(blended_image, dstsize=size)
        # Add the expanded image to the current level of the Laplacian pyramid. cv2.add ensures that operations on such
        # images remain within valid range (up to 255).
        blended_image = cv2.add(np.float32(laplacian_pyramid[i]), expanded_laplacian)

    return blended_image


def blend_images_using_mask(image1, image2, mask):
    """
    Blends two images using a mask, through the use of Laplacian pyramids.

    Parameters:
    - image1: numpy.ndarray, the first input image.
    - image2: numpy.ndarray, the second input image.
    - mask: numpy.ndarray, the mask to control the blending. It should be normalized to [0, 1].

    Returns:
    None, the blended image is saved on the disk.
    """
    smallest_dimension = min(image1.shape[0], image1.shape[1])
    number_of_pyramid_levels = int(np.floor(np.log2(smallest_dimension))) + 1

    # Create Gaussian pyramids for the input images and the mask. Lower levels are bigger (closer to the original image)
    image1_gaussian = calculate_gaussian_pyramid(image1, number_of_pyramid_levels)
    image2_gaussian = calculate_gaussian_pyramid(image2, number_of_pyramid_levels)
    mask_gaussian = calculate_gaussian_pyramid(mask, number_of_pyramid_levels)

    # Create Laplacian pyramids for the input images. Lower levels are smaller than the higher levels
    image1_laplacian = calculate_laplacian_pyramid(image1_gaussian, number_of_pyramid_levels)
    image2_laplacian = calculate_laplacian_pyramid(image2_gaussian, number_of_pyramid_levels)

    # Create Blended Laplacian pyramid of the input images and the gaussian pyramid of the mask
    blended_laplacian_pyramid = []
    # Loop on all possible triplets of laplacian1, laplacian2 and mask each level
    for laplacian1, laplacian2, blurred_mask in zip(image1_laplacian, image2_laplacian, mask_gaussian):
        blended_level = blend_laplacian_pyramid_level_using_mask(laplacian1, laplacian2, blurred_mask)
        blended_laplacian_pyramid.append(blended_level)

    blended_image = sum_pyramid_levels(blended_laplacian_pyramid)

    # Use np.clip to ensure values are within [0, 255] range
    blended_image_clipped = np.clip(blended_image, 0, 255).astype(np.uint8)

    # save_image_to_file(blended_image_clipped, "blended_image")


def blend_laplacian_pyramid_level_using_mask(laplacian1, laplacian2, blurred_mask):
    """
    Calculates a blended level for the Laplacian pyramid, using a mask.

    Parameters:
    - laplacian1: numpy.ndarray, the Laplacian pyramid level of the first image.
    - laplacian2: numpy.ndarray, the Laplacian pyramid level of the second image.
    - blurred_mask: numpy.ndarray, the Gaussian pyramid level of the mask.

    Returns:
    - numpy.ndarray, the blended Laplacian level.
    """
    part_of_laplacian1 = blurred_mask * laplacian1
    part_of_laplacian2 = (1 - blurred_mask) * laplacian2
    return part_of_laplacian1 + part_of_laplacian2


def generate_gaussian_masks(rows, cols, sigma):
    """
    Generates Gaussian low-pass and high-pass masks for image filtering in the frequency domain.

    Parameters:
    - rows (int): The number of rows in the mask, corresponding to the image height.
    - cols (int): The number of columns in the mask, corresponding to the image width.
    - sigma (float): The standard deviation of the Gaussian distribution, controlling the spread of the filter.

    Returns:
    - Tuple of numpy.ndarray: The first element is the low-pass Gaussian mask, which emphasizes lower frequencies.
      The second element is the high-pass mask, obtained by subtracting the low-pass mask from 1, emphasizing higher
      frequencies.
    """
    # This grid represents the positions of each pixel in the frequency domain, centered around the origin (0, 0),
    # with size of the input images.
    x = np.linspace(-cols // 2, cols // 2 - 1, cols)
    y = np.linspace(-rows // 2, rows // 2 - 1, rows)
    X, Y = np.meshgrid(x, y)

    # Calculate the Euclidean distance of each point from the center of the frequency domain.
    d = np.sqrt(X ** 2 + Y ** 2)

    # Apply the Gaussian formula to create a low-pass mask.
    low_pass_mask = np.exp((d ** 2) / -(2 * sigma ** 2))

    # Create a complementary high-pass mask.
    high_pass_mask = 1 - low_pass_mask

    return low_pass_mask, high_pass_mask


def hybrid_images(image1, image2):
    """
    Creates a hybrid image by blending the low-frequency components of the first image with the high-frequency
    components of the second image. This method is based on Fourier transform techniques.

    Parameters:
    - image1 (numpy.ndarray): The first input image from which low-frequency components are extracted.
    - image2 (numpy.ndarray): The second input image from which high-frequency components are extracted.

    Returns:
    - numpy.ndarray: The resulting hybrid image, stored as an 8-bit image.
    """
    # Convert input images to grayscale for color not to affect the result.
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the Fourier transform of both images.
    fourier_image1 = np.fft.fft2(image1_gray)
    fourier_shifted_image1 = np.fft.fftshift(fourier_image1)
    fourier_image2 = np.fft.fft2(image2_gray)
    fourier_shifted_image2 = np.fft.fftshift(fourier_image2)

    # Generate Gaussian masks for filtering.
    sigma = 9  # hyperparameter adjusted to the images
    rows, cols = image1_gray.shape
    low_pass_mask, high_pass_mask = generate_gaussian_masks(rows, cols, sigma)

    # Apply the low-pass filter to the first image and the high-pass filter to the second image.
    low_frequencies = fourier_shifted_image1 * low_pass_mask
    high_frequencies = fourier_shifted_image2 * high_pass_mask

    # Combine the filtered images.
    fourier_combined = low_frequencies + high_frequencies
    fourier_combined_shifted_back = np.fft.ifftshift(fourier_combined)
    spatial_combined = np.fft.ifft2(fourier_combined_shifted_back)

    # Convert the result to real values and ensure it is in the valid range [0, 255].
    hybrid_image = np.abs(spatial_combined)

    hybrid_image_clipped = np.clip(hybrid_image, 0, 255).astype(np.uint8)

    # save_image_to_file(hybrid_image_clipped, "hybrid_image")


def visualize_gaussian_pyramid(gaussian_pyramid):
    """
    Visualizes a Gaussian pyramid by stacking images horizontally, adjusting for color channels dynamically.

    Parameters:
    - gaussian_pyramid: list of numpy.ndarray, the Gaussian pyramid to visualize.

    Returns:
    - numpy.ndarray, an image visualizing the Gaussian pyramid.
    """
    total_width = sum([img.shape[1] for img in gaussian_pyramid])
    max_height = max([img.shape[0] for img in gaussian_pyramid])

    # Create a blank canvas accordingly
    pyramid_image = np.zeros((max_height, total_width, gaussian_pyramid[0].ndim), dtype=np.uint8)

    current_x = 0
    for img in gaussian_pyramid:
        displayable_img = cv2.convertScaleAbs(img)  # Ensure img is in the correct format for visualization
        pyramid_image[:img.shape[0], current_x:current_x + img.shape[1]] = displayable_img
        current_x += img.shape[1]

    return pyramid_image


def visualize_laplacian_pyramid(laplacian_pyramid):
    """
    Visualizes a Laplacian pyramid by stacking images horizontally without aggressive normalization,
    aiming to preserve natural contrast across levels.

    Parameters:
    - laplacian_pyramid: list of numpy.ndarray, the Laplacian pyramid to visualize.

    Returns:
    - numpy.ndarray, an image visualizing the Laplacian pyramid.
    """
    total_canvas_width = sum([img.shape[1] for img in laplacian_pyramid])
    max_canvas_height = max([img.shape[0] for img in laplacian_pyramid])

    pyramid_image = np.zeros((max_canvas_height, total_canvas_width, laplacian_pyramid[0].ndim), dtype=np.uint8)

    current_x = 0
    for img in laplacian_pyramid:
        # Adjust the image to have its zero point at 128, with both positive and negative values represented
        displayable_img = img.astype(np.float32)
        displayable_img += 128  # Shift zero to 128 to visualize negative values and keep the middle gray
        np.clip(displayable_img, 0, 255, out=displayable_img)  # Clip values to ensure they're within range
        displayable_img = displayable_img.astype(np.uint8)

        # Place the adjusted image on the canvas
        pyramid_image[:displayable_img.shape[0], current_x:current_x + displayable_img.shape[1]] = displayable_img
        current_x += displayable_img.shape[1]

    return pyramid_image


def visualize_pyramids(image):
    """
    Visualizes both Gaussian and Laplacian pyramids of an image. The function calculates the pyramids and then
    calls the visualization functions for each pyramid.

    Parameters:
    - image: The input image for which the pyramids will be visualized.

    Returns:
    None, the visualizations are saved to files.
    """
    smallest_dimension = min(image.shape[0], image.shape[1])
    number_of_pyramid_levels = int(np.floor(np.log2(smallest_dimension))) + 1  # For the original image
    gaussian_pyramid = calculate_gaussian_pyramid(image, number_of_pyramid_levels)
    laplacian_pyramid = calculate_laplacian_pyramid(gaussian_pyramid, number_of_pyramid_levels)
    # Visualize and save the Gaussian pyramid
    gauss_vis = visualize_gaussian_pyramid(gaussian_pyramid)
    # save_image_to_file(gauss_vis, "gaussian_pyramid")
    # Visualize and save the Laplacian pyramid
    laplace_vis = visualize_laplacian_pyramid(laplacian_pyramid)
    # save_image_to_file(laplace_vis, "laplacian_pyramid")


# def main():
#     """
#     The main function that executes the blending of two images using a mask, creates a hybrid image, saves
#     result images, and visualizes the Gaussian and Laplacian pyramids for the first input image.
#
#     Parameters:
#     - image1_path: The file path to the first image.
#     - image2_path: The file path to the second image.
#     - mask_path: The file path to the mask image.
#
#     Returns:
#     None, the results are saved to files.
#     """
#     image1 = read_image_from_file(image1_path)
#     image2 = read_image_from_file(image2_path)
#     mask = read_image_from_file(mask_path)
#     binary_mask = mask / MAX_INTENSITY  # The mask is in the range [0, 255], so we need to normalize it to [0, 1]
#     blend_images_using_mask(image1, image2, binary_mask)
#
#     close_image = read_image_from_file(close_image_path)
#     far_image = read_image_from_file(far_image_path)
#     far_image = cv2.resize(far_image, (close_image.shape[1], close_image.shape[0]))  # Ensure the images have the same size
#
#     hybrid_images(close_image, far_image)
#
#     visualize_pyramids(image1)
#
#
# if __name__ == '__main__':
#     # Define file paths for the input images and the mask
#     image1_path ="/Users/srashkovits/PycharmProjects/image/ex3/Exercise Inputs/classic example/apple.jpeg"
#     image2_path = "/Users/srashkovits/PycharmProjects/image/ex3/Exercise Inputs/classic example/orange.jpeg"
#     mask_path = "/Users/srashkovits/PycharmProjects/image/ex3/Exercise Inputs/classic example/classic mask.jpeg"
#
#     close_image_path = "/Users/srashkovits/PycharmProjects/image/ex3/Exercise Inputs/blended animals/cat.png"
#     far_image_path = "/Users/srashkovits/PycharmProjects/image/ex3/Exercise Inputs/blended animals/dog.png"
#
#     # Execute the main function with the specified file paths
#     main()
