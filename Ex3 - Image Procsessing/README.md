# Image Blending and Hybrid Image Creation

Repository for Exercise 3 of the Image Processing course, which involves the creation of blended and hybrid images using the techniques of image pyramids and frequency domain manipulation.

## Exercise Objectives

The main objectives of this exercise are:

- To seamlessly blend two images using a given mask.
- To create a hybrid image where one image is visible at close range, and another is visible from a distance.

These objectives are achieved by manipulating image frequencies through Gaussian and Laplacian pyramids for image blending and by employing high-pass and low-pass filtering in the frequency domain for hybrid images.

## Getting Started

To run the image processing scripts and generate your own blended and hybrid images, you will need to set up your environment and install the necessary dependencies.

### Prerequisites

- Python
- Image processing libraries as detailed in `requirements.txt`

### Installation

Install the required Python libraries by executing the following command:

```bash
pip install -r requirements.txt
```

## Running the Scripts

The repository contains Python scripts for both image blending and hybrid image creation:

- `image_blending.py`: Blends two images based on a mask.
- `hybrid_image.py`: Creates a hybrid image from two distinct source images.

Run the scripts in the following manner, replacing `path_to_image_1`, `path_to_image_2`, and `path_to_mask` with your image file paths:

```bash
python image_blending.py path_to_image_1 path_to_image_2 path_to_mask
python hybrid_image.py path_to_image_close path_to_image_far
```

## Project Structure

- `ex3.py`: Contains functions for image blending and hybrid image creation.
- `requirements.txt`: Lists all Python package dependencies.
- `images/`: A directory that should contain all your source and mask images.

## Methodology

The exercise is based on the principle that images can be represented and manipulated at different frequency levels to achieve various effects:

### Image Blending

1. Load two images and a binary mask.
2. Create Gaussian pyramids for each image.
3. Construct Laplacian pyramids for blending.
4. Reconstruct the blended image from the blended pyramid.

### Hybrid Images

1. Load two images to be combined.
2. Transform each image into the frequency domain.
3. Apply a high-pass filter to the first image and a low-pass filter to the second image.
4. Combine the images in the frequency domain.
5. Transform the combined image back into the spatial domain.

## Results

Your final images will be a blend of two images based on the mask provided and a hybrid image that changes its appearance based on the viewing distance.

## Conclusion

This exercise explores the power and limitations of image processing using simple algorithms. The resulting images underscore the impact of choosing appropriate blending masks, image pairs, and parameter settings.
