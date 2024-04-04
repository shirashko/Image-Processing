# High-Resolution Image Blending

Repository of Exercise 4 of the Image Processing course, where we focus on blending high-resolution segments seamlessly into low-resolution images using various image processing techniques.

## Exercise Overview

This exercise challenges us to integrate high-resolution parts of an image into their corresponding low-resolution counterparts. The task involves using feature extraction, feature matching, RANSAC for homography estimation, and image warping to achieve a seamless blend.

## Getting Started

Follow the instructions below to set up your environment to run the blending algorithm.

### Prerequisites

- Python
- OpenCV, NumPy, and other libraries listed in `requirements.txt`

### Installation

To install the required libraries, execute the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Running the Blending Script

To run the image blending script:

```bash
python ex4.py --low_res <path_to_low_res_image> --high_res <path_to_high_res_segment>
```

Replace `<path_to_low_res_image>` with the path to your low-resolution image and `<path_to_high_res_segment>` with the path to your high-resolution image segment.

## Project Structure

- `ex4.py`: Contains the main script for image blending.
- `requirements.txt`: Lists the necessary Python package dependencies.

## Methodology

The approach involves several steps:
1. **Feature Extraction and Matching**: Identifies key points and matches them between the high and low-resolution images.
2. **RANSAC Homography Estimation**: Robustly estimates a transformation matrix to align the images.
3. **Image Warping**: Aligns the high-resolution segment with the low-resolution image.
4. **Blending**: Smoothly integrates the aligned high-resolution details into the low-resolution base.

### Detailed Algorithm Steps:

- Load the images and identify keypoints using SIFT.
- Match features between images, filtering by distance ratio.
- Estimate a homography matrix with RANSAC to align images accurately.
- Apply backward warping to align the high-resolution image with the low-resolution image's perspective.
- Create a binary mask for blending from the warped image.
- Perform pyramid blending to merge images seamlessly.

## Visual Results

You can view the blending results, showing how the high-resolution details are smoothly integrated into the low-resolution image, in the `results` directory.

## Conclusion

This exercise illustrates the practical application of complex image processing techniques to combine images of different resolutions. It demonstrates the effectiveness of feature-based image registration and the challenges posed by blending disparate resolution images.
