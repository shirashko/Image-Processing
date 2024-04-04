# Audio Denoising Exercise

Repository for the Audio Denoising exercise, part of the Image Processing course. This project focuses on the application of Fast Fourier Transform (FFT) and Short-Time Fourier Transform (STFT) methods to remove noise from audio signals.

## Exercise Overview

The objective of this exercise is to enhance audio quality by reducing noise using frequency domain analysis. Two noisy audio samples, each with distinct noise characteristics, are provided. The task involves applying tailored denoising techniques to these samples to mitigate the disruptive noise without access to the clean audio.

## Getting Started

These instructions will guide you on how to set up the project and run the denoising algorithms on your own machine.

### Prerequisites

- Python
- Required Python packages as listed in `requirements.txt`

### Installation

Install the necessary Python libraries by running the following command:

```bash
pip install -r requirements.txt
```

## Running the Denoising Algorithms

There are two separate functions in the script, each corresponding to a specific audio noise problem:

- `q1(audio_path)` for the first audio file with consistent noise across the spectrum.
- `q2(audio_path)` for the second audio file with noise present only in certain parts of the audio.

To execute the denoising functions, use:

```python
import ex2

denoised_audio_q1 = ex2.q1('path_to_q1_audio_file')
denoised_audio_q2 = ex2.q2('path_to_q2_audio_file')
```

Replace `'path_to_q1_audio_file'` and `'path_to_q2_audio_file'` with the actual file paths of the noisy audio files.

## Project Structure

- `ex2.py`: The Python script containing the denoising algorithms.
- `ex2.pdf`: The report describing the solutions' approach, algorithms, and results.
- `requirements.txt`: A list of Python package dependencies.

## Methodology

The denoising process involves transforming the audio signal into the frequency domain using FFT and STFT. By analyzing the spectral data, the algorithms developed for this exercise effectively identify and suppress the noise components in the audio.

### Key Steps for Each Algorithm:

**For the First Audio (Q1):**
1. Load the audio and apply FFT.
2. Identify and nullify the frequency with the highest magnitude.
3. Apply inverse FFT to reconstruct the denoised audio.

**For the Second Audio (Q2):**
1. Load the audio and apply STFT.
2. Isolate the noise in the specified time-frequency region.
3. Apply inverse STFT to reconstruct the denoised audio.

## Conclusion

This exercise demonstrated the power of spectral analysis for audio signal processing. The findings emphasized the need for different denoising strategies depending on the noise characteristics and the audio signal's context.
