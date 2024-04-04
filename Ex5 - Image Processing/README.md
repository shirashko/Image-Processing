# Deep Learning - Deep Style Image Prior

This repository contains the implementation for Exercise 5 in the Image Processing course, focusing on image reconstruction tasks utilizing StyleGAN2, a state-of-the-art generative adversarial network.

## Overview

The exercise explores image reconstruction through GAN Inversion and Latent Optimization, employing StyleGAN2, pre-trained on the FFHQ dataset. The task involves using StyleGAN2 as an image prior to reconstruct images subjected to various forms of degradation, such as decolorization, and masking.

## Setup

The project is implemented using Google Colab to leverage GPU acceleration for neural network computations.

### Important Notes

- Ensure to work with a copy of the notebook in your Google Drive to avoid code loss.
- It is crucial that all images are aligned properly before usage, adhering to the preprocessing steps consistent with the training dataset of StyleGAN2.

## Tasks

- **Image Alignment**: Running an image alignment script on chosen images.
- **GAN Inversion**: Implementing the inversion of a pre-trained GAN to reconstruct a given image.
- **Image Reconstruction Tasks**: Reconstructing images that underwent degradation, adjusting the model to treat these degraded images as if they belong to the trained domain.

### Image Reconstruction Challenges

- **Image Colorization**: Colorizing a grayscale image to produce a natural and realistic colored version.
- **Image Inpainting**: Filling in missing parts of an image to create a complete and coherent picture.

## Usage

To execute the tasks, follow the interactive instructions provided in the Colab Notebook. The process involves running the notebook's cells sequentially and observing the optimization process of the latent vectors.

## Results

Include the results of each task as specified in the exercise's guidelines:

- Progression visuals during the optimization process.
- The effect of `latent_dist_reg_weight` and `num_steps` on the results.
- Final reconstruction results compared to the original images.

## Conclusion

The exercise demonstrates the versatility of GANs as priors for image reconstruction tasks, highlighting the potential of machine learning models to understand and recreate complex image patterns.

## Submission

Submit a zip file containing:

- The Colab notebook (`.ipynb` file)
- A PDF report summarizing the methodology, results, and insights
- Original and degraded images, reconstruction results, and corresponding latent files

## References

1. Aviv Gabbay and Yedid Hoshen. Style generator inversion for image enhancement and animation, 2019.
2. Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN, 2020.
