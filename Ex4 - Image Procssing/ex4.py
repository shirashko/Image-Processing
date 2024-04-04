import cv2
import numpy as np

# --------- Utility functions --------- #

MAX_INTENSITY = 255.0


def load_image(path):
    """
    Load an image from a file.
    :param path: the path to the image file
    :return: the loaded image
    """
    return cv2.imread(path)


def save_image(image, filename):
    """Save an image to a file in the specified outputs' directory."""
    output_dir = 'Exercise Outputs/'
    path = output_dir + filename  # Prepend the directory to the filename
    cv2.imwrite(path, image)


def get_normalized_three_channels_mask(mask):
    """
    Normalize the mask to [0, 1] and expand it to 3 channels.
    :param mask: the mask to normalize
    :return: the normalized mask with 3 channels
    """
    mask = mask.astype(np.float32) / MAX_INTENSITY
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def create_binary_mask(image):
    """
    Create a binary mask by marking non-black pixels in the warped image as white to differentiate areas of interest.
    This function assumes that the image is in color and that only the background "canvas" is black.
    :param image: the input image
    :return: the binary mask of the image
    """
    return np.any(image > 0, axis=2).astype(np.uint8) * MAX_INTENSITY


# --------- Visualizations --------- #
def visualize_sift_keypoints(high_res_image, keypoints_a, keypoints_b, low_res_image):
    """
    Visualize the SIFT keypoints on the low-resolution and high-resolution images.
    :param high_res_image: the high-resolution image
    :param image: the high-resolution image
    :param keypoints_a: the keypoints of the low-resolution image
    :param keypoints_b: the keypoints of the high-resolution image
    :param low_res_image: the low-resolution image
    """
    image_with_keypoints = cv2.drawKeypoints(low_res_image, keypoints_a, None,
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save_image(image_with_keypoints, 'image_low_res_with_sift_keypoints.jpg')
    image_with_keypoints = cv2.drawKeypoints(high_res_image, keypoints_b, None,
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    save_image(image_with_keypoints, 'image_high_res_with_sift_keypoints.jpg')


def visualize_matches(good_matches, high_res_image, keypoints_a, keypoints_b, low_res_image):
    """
    Visualize the matches between the low-resolution and high-resolution images.
    :param good_matches: the good matches after applying the ratio test
    :param high_res_image: high-resolution image
    :param keypoints_a: keypoints of the low-resolution image
    :param keypoints_b: keypoints of the high-resolution image
    :param low_res_image: low-resolution image
    """
    matched_image = cv2.drawMatches(low_res_image, keypoints_a, high_res_image, keypoints_b, good_matches, None,
                                    flags=2)
    save_image(matched_image, 'matched_image.jpg')


# --------- Blend Images --------- #

# Define the number of levels in the pyramids (adjustable)
PYRAMID_LEVELS = 2


def create_gaussian_pyramid(image, levels):
    """
    Calculate the Gaussian pyramid of an image.
    :param image: the input image
    :param levels: the number of levels in the pyramid
    :return: the Gaussian pyramid with the specified number of levels
    """
    gaussian_pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid


def create_laplacian_pyramid(gaussian_pyramid):
    """
    Calculate the Laplacian pyramid from the Gaussian pyramid.
    :param gaussian_pyramid: the Gaussian pyramid
    :return: the Laplacian pyramid
    """
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def blend_pyramids(laplacian_pyramid1, laplacian_pyramid2, gaussian_pyramid_mask):
    """
    Blend two Laplacian pyramids using a Gaussian pyramid mask.
    :param laplacian_pyramid1: the first Laplacian pyramid
    :param laplacian_pyramid2: the second Laplacian pyramid
    :param gaussian_pyramid_mask: the Gaussian pyramid mask
    :return: the blended Laplacian pyramid
    """
    blended_pyramid = []
    for l1, l2, gm in zip(laplacian_pyramid1, laplacian_pyramid2, gaussian_pyramid_mask):
        blended = l1 * gm + l2 * (1 - gm)
        blended_pyramid.append(blended)
    return blended_pyramid


def reconstruct_from_pyramid(laplacian_pyramid):
    """
    Reconstruct the image from the Laplacian pyramid.
    :param laplacian_pyramid: the Laplacian pyramid
    :return: the reconstructed image
    """
    image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(image, laplacian_pyramid[i])
    return image


def blend_images(image1, image2, mask):
    """
    Blend two images using a mask and a specified number of levels in the pyramids.
    :param image1: the first image in RGB format
    :param image2: the second image in RGB format
    :param mask: the mask, assumed to be normalized to [0, 1] and expanded to 3 channels
    :return: the blended image
    """
    # Generate pyramids
    laplacian_pyramid1 = create_laplacian_pyramid(create_gaussian_pyramid(image1, PYRAMID_LEVELS))
    laplacian_pyramid2 = create_laplacian_pyramid(create_gaussian_pyramid(image2, PYRAMID_LEVELS))
    gaussian_pyramid_mask = create_gaussian_pyramid(mask.astype(np.float32), PYRAMID_LEVELS)

    # Blend pyramids and reconstruct the image from the blended pyramid
    blended_pyramid = blend_pyramids(laplacian_pyramid1, laplacian_pyramid2, gaussian_pyramid_mask)
    blended_image = reconstruct_from_pyramid(blended_pyramid)
    return blended_image


# --------- Image Alignment --------- #

NEIGHBORS_ASPECT_RATIO_THRESHOLD_FOR_MATCH = 0.75


def detect_and_compute_features(image, detector):
    """
    Detect features and compute descriptors using the SIFT algorithm.
    :param image: the input image
    :param detector: the SIFT detector
    :return: the keypoints and descriptors
    """
    return detector.detectAndCompute(image, None)


def match_features(descriptor_a, descriptor_b, ratio_threshold=NEIGHBORS_ASPECT_RATIO_THRESHOLD_FOR_MATCH):
    """
    Match features using 2-NN matching with the BFMatcher and apply the ratio test
    to filter matches based on the ratio of the distances of the nearest to the second-nearest match.
    :param descriptor_a: the descriptors of the first image
    :param descriptor_b: the descriptors of the second image
    :param ratio_threshold: the ratio threshold for the ratio test
    :return: the good matches after applying the ratio test (i.e., the matches that pass the test)
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptor_a, descriptor_b, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]
    return good_matches


def find_homography(keypoints_a, keypoints_b, good_matches):
    """
    Find homography matrix using matched points and the RANSAC algorithm to robustly estimate the transformation.
    :param keypoints_a: the keypoints of the first image
    :param keypoints_b: the keypoints of the second image
    :param good_matches: the good matches after applying the ratio test
    :return: the homography matrix and the mask of inliers
    """
    pts_a = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_b = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return cv2.findHomography(pts_b, pts_a, cv2.RANSAC)  # uses the RANSAC algorithm


def warp_image(image, H, dimensions):
    """
    Warp image using a homography matrix with backward warping to align images based on the calculated homography.
    :param image: the input image
    :param H: the homography matrix
    :param dimensions: the dimensions of the output image
    :return: the warped image
    """
    return cv2.warpPerspective(image, H, dimensions)


# --------- Power of used algorithms --------- #


def match_features_1nn(descriptor_a, descriptor_b, factor=2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(descriptor_a, descriptor_b)
    # Find the minimum distance in all matches
    min_distance = min(matches, key=lambda x: x.distance).distance
    # Filter matches based on the factor times the minimum distance
    filtered_matches = [m for m in matches if m.distance < factor * min_distance]
    return filtered_matches


def find_homography_without_ransac(keypoints_a, keypoints_b, good_matches):
    """
    Find homography matrix using matched points without using the RANSAC algorithm.
    :param keypoints_a: the keypoints of the first image
    :param keypoints_b: the keypoints of the second image
    :param good_matches: the good matches after applying the ratio test
    :return: the homography matrix and the mask of inliers
    """
    pts_a = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_b = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_b, pts_a)  # No RANSAC used
    return H


def using_1_nn_similarity_measure(descriptors_a, descriptors_b, high_res_image, keypoints_a, keypoints_b,
                                  low_res_image):
    # 1-NN matching
    matches_1nn = match_features_1nn(descriptors_a, descriptors_b)
    # Visualize 1-NN matches
    matched_image_1nn = cv2.drawMatches(low_res_image, keypoints_a, high_res_image, keypoints_b, matches_1nn, None,
                                        flags=2)
    save_image(matched_image_1nn, 'matched_image_1nn.jpg')
    H, _ = find_homography(keypoints_a, keypoints_b, matches_1nn)
    height, width, channels = low_res_image.shape
    # Warp the high-resolution image to the perspective of the low-resolution image
    c = warp_image(high_res_image, H, (width, height))
    save_image(c, 'warped_image_with_1_NN.jpg')
    # Create and save binary mask
    mask = create_binary_mask(c)
    save_image(mask, 'mask_image_with_1_NN.jpg')
    binary_mask = get_normalized_three_channels_mask(mask)
    # Blend images using the mask
    blended_image_1_NN = blend_images(c, low_res_image, binary_mask)
    save_image(blended_image_1_NN, 'blended_image_with_1_NN.jpg')


def matching_without_ransac(good_matches, height, high_res_image, keypoints_a, keypoints_b, width):
    H_without_ransac = find_homography_without_ransac(keypoints_a, keypoints_b, good_matches)
    c_without_ransac = warp_image(high_res_image, H_without_ransac, (width, height))
    save_image(c_without_ransac, 'warped_image_without_ransac.jpg')
    mask_without_ransac = create_binary_mask(c_without_ransac)
    save_image(mask_without_ransac, 'mask_image_without_ransac.jpg')


# --------- Main --------- #

def main():
    """
    Main function to blend two images, one with low resolution and one with high resolution, using feature matching and
    pyramid blending techniques.

    The steps involved are:
    1. Load a low-resolution image, and a high-resolution image.
    2. Use SIFT to detect and match features between the low-res and high-res images.
    3. Calculate a homography matrix based on matched features to align the images.
    4. Warp the high-resolution image to the perspective of the low-resolution image using the homography matrix.
    5. Create a binary mask from the warped image and convert it to a 3-channel mask.
    6. Blend the warped high-resolution image and the original low-resolution image using the mask.
    7. Save the final blended image to disk.
    """
    # Paths to images
    low_res_path = '/Users/srashkovits/PycharmProjects/image/ex4/Exercise Inputs/lake_low_res.jpg'
    high_res_path = '/Users/srashkovits/PycharmProjects/image/ex4/Exercise Inputs/lake_high_res.png'

    # Load images
    low_res_image = load_image(low_res_path)
    high_res_image = load_image(high_res_path)

    # Feature Extraction and Matching using SIFT
    sift = cv2.SIFT_create()
    keypoints_a, descriptors_a = detect_and_compute_features(low_res_image, sift)
    keypoints_b, descriptors_b = detect_and_compute_features(high_res_image, sift)

    visualize_sift_keypoints(high_res_image, keypoints_a, keypoints_b, low_res_image)

    # Match features and apply ratio test
    good_matches = match_features(descriptors_a, descriptors_b)
    visualize_matches(good_matches, high_res_image, keypoints_a, keypoints_b, low_res_image)

    # Find homography
    H, _ = find_homography(keypoints_a, keypoints_b, good_matches)
    print("Homography Matrix:\n", H)

    # Warp the high-resolution image to the perspective of the low-resolution image
    height, width, _ = low_res_image.shape
    c = warp_image(high_res_image, H, (width, height))
    save_image(c, 'warped_image.jpg')

    # Create and save binary mask
    mask = create_binary_mask(c)
    save_image(mask, 'mask_image.jpg')
    binary_mask = get_normalized_three_channels_mask(mask)

    # Blend images using the mask
    blended_image = blend_images(c, low_res_image, binary_mask)
    save_image(blended_image, 'blended_image.jpg')

    # Power of chosen algorithms
    matching_without_ransac(good_matches, height, high_res_image, keypoints_a, keypoints_b, width)
    using_1_nn_similarity_measure(descriptors_a, descriptors_b, high_res_image, keypoints_a, keypoints_b, low_res_image)


if __name__ == "__main__":
    main()
