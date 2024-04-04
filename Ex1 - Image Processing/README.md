# Scene Cut Detection in Videos

This project is part of the Image Processing course exercises, concentrating on detecting scene cuts within videos through histogram analysis. The goal is to analyze videos to identify transitions between two distinct scenes. This repository contains the implementation for this exercise, following the guidelines and tasks as outlined in the course material.

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

Ensure you have Python installed on your system. Additionally, you will need to install specific Python libraries listed in the `requirements.txt` file.

### Installing

To install the required libraries, execute the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Running the Script

Use the following command structure in your terminal to run the scene cut detection algorithm:

```bash
python ex1.py <video_path> <video_type>
```

- `<video_path>`: The path to the video file you want to analyze.
- `<video_type>`: Specifies the category of the analyzed video, where 1 corresponds to videos without quantization effects and 2 denotes videos that include quantization artifacts.

### Example

```bash
python ex1.py 'path/to/video.mp4' 1
```

This will run the scene cut detection on the specified video and output the frame numbers where a scene transition is detected.

## Project Structure

- `ex1.py`: The main Python script for the scene cut detection algorithm.
- `requirements.txt`: Lists all the dependencies required to run the project.

## Methodology

The algorithm leverages histogram analysis to detect scene changes in videos. It first converts frames into grayscale to simplify the analysis, then examines the histogram of each frame to identify significant changes indicative of scene transitions. The methodology, including specific adjustments for different video categories, is elaborated in the report accompanying this code.
