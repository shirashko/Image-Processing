import time

import mediapy as mp
import numpy as np
import matplotlib.pyplot as plt

# Number of intensity levels
NUMBER_OF_INTENSITY_LEVELS = 256


def read_video_in_grayscale(video_path):
    """
    Reads a video from the specified path and returns it as a grayscale image.
    Each frame is a 2D array with dimensions [frame_height, frame_width].
    :param video_path: The path to the video file.
    :return: An array of grayscale frames in the video.
    """
    return mp.read_video(video_path, output_format='gray')


def read_video_in_rgb(video_path):
    """
    Reads a video from the specified path as an RGB image.
    :param video_path: The path to the video file.
    :return: A 4D `numpy` array with dimensions (frame, height, width, channel)
    """
    return mp.read_video(video_path, output_format='rgb')


def cumulative_histogram(frame):
    """
    Computes and returns the cumulative histogram of a frame.
    The histogram represents the cumulative count of pixel intensities up to each bin.
    :param frame: A frame from the video.
    :return: The cumulative histogram of the frame.
    """
    hist, _ = np.histogram(frame, bins=NUMBER_OF_INTENSITY_LEVELS, range=(0, 255))
    return np.cumsum(hist)


def compute_consecutive_frames_distance(frames):
    """
    Calculates the distance between each pair of consecutive frames based on the L1 norm
    of the difference in their cumulative histograms.
    :param frames: An array of frames (grayscale).
    :return: A list of differences between each pair of consecutive frames.
    """
    histograms = np.array([cumulative_histogram(frame) for frame in frames])
    bins_differences = np.diff(histograms, axis=0)
    consecutive_histograms_distances = np.sum(np.abs(bins_differences), axis=1)
    return consecutive_histograms_distances.tolist()


def find_scene_cut(frames):
    """
    Identifies the scene cut in the video based on the maximum consecutive frame distance.
    :param frames: An array of grayscale frames from the video.
    :return: The index of the frame immediately before the scene cut, identified by the maximum difference.
    """
    distances = compute_consecutive_frames_distance(frames)
    max_diff = max(distances)
    cut_frame_index = distances.index(max_diff)
    return cut_frame_index


def _find_index_of_second_farthest_consecutive_frames(frames):
    """
    Identifies the second possible scene cut in the video based on the second maximum consecutive frames distance.
    :param frames: An array of grayscale frames from the video.
    :return: The index of the frame with the second maximum distance.
    """
    distances = compute_consecutive_frames_distance(frames)
    sorted_distances = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)
    second_max_diff_index = sorted_distances[1][0]
    print(f"Index of the frame with the second highest difference: {second_max_diff_index}")
    return second_max_diff_index


def apply_min_max_normalization(distances):
    """
    Normalizes the frame differences to a range of [0, 1].
    :param distances: A list of raw frame differences.
    :return: A list of normalized frame differences.
    """
    min_diff = min(distances)
    max_diff = max(distances)
    return [(diff - min_diff) / (max_diff - min_diff) for diff in distances]


def _plot_consecutive_frames_distances(cut_frame_index, distances, video_type, video_number):
    """
    Plots the frame differences with markers indicating the scene cut.
    :param cut_frame_index: Index of the frame where the scene cut is detected.
    :param distances: A list of consecutive frame distances.
    :param video_type: Type of the video.
    :param video_number: Number of the video in the sequence.
    """
    normalized_distances = apply_min_max_normalization(distances)
    plt.figure(figsize=(10, 6))
    plt.plot(normalized_distances, marker='o', markersize=3)
    plt.title(f'Video {video_number}: Frame Distances for video from category {video_type}')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance Scaled to [0,1] Range')
    plt.axvline(x=cut_frame_index, color='r', linestyle='--', label=f'Cut at frame {cut_frame_index}')
    plt.legend()
    plt.xlim(0, len(distances) - 1)
    plt.show()


def _plot_scene_cut_frames(rgb_frames, cut_frame_index, video_number):
    """
    Plots the frames at the scene cut with the number of pixels in rows and columns indicated.
    :param rgb_frames: An array of RGB frames.
    :param cut_frame_index: Index of the frame where the scene cut is detected.
    :param video_number: Number of the video in the sequence.
    """
    if 0 <= cut_frame_index < len(rgb_frames) - 1:
        frame_before_cut = rgb_frames[cut_frame_index]
        frame_after_cut = rgb_frames[cut_frame_index + 1]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(frame_before_cut)
        plt.title(f'Video {video_number} - Frame before cut')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(frame_after_cut)
        plt.title(f'Video {video_number} - Frame after cut')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def _plot_second_highest_distance_frame(frames, second_max_diff_index, video_number):
    """
    Plots the frames around the second-highest difference index.

    :param frames: An array of all frames in the video.
    :param second_max_diff_index: Index of the frame with the second-highest difference.
    :param video_number: Number of the video in the sequence.
    """
    if 0 <= second_max_diff_index < len(frames) - 1:
        frame_before = frames[second_max_diff_index]
        frame_after = frames[second_max_diff_index + 1]

        plt.figure(figsize=(12, 6))

        # Plot frame before the second-highest difference
        plt.subplot(1, 2, 1)
        plt.imshow(frame_before)
        plt.title(f'Video {video_number} - Frame before second highest cut (Index: {second_max_diff_index})')
        plt.axis('off')

        # Plot frame after the second-highest difference
        plt.subplot(1, 2, 2)
        plt.imshow(frame_after)
        plt.title(f'Video {video_number} - Frame after second highest cut (Index: {second_max_diff_index + 1})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def _plot_histograms_of_consecutive_frames(frames, first_index, second_index, video_number):
    """
    Plots side-by-side histograms for the 'before' and 'after' frames of the farthest and second-farthest cuts.
    :param frames: An array of all frames in the video.
    :param first_index: Index of the frame with the farthest consecutive distance.
    :param second_index: Index of the frame with the second farthest consecutive distance.
    :param video_number: Number of the video in the sequence.
    """
    # Plot histograms for the frame before and after the farthest cut
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_individual_histogram(frames[first_index], f'Video {video_number} - Histogram Before Farthest Cut')
    plt.subplot(1, 2, 2)
    plot_individual_histogram(frames[first_index + 1], f'Video {video_number} - Histogram After Farthest Cut')
    plt.show()

    # Plot histograms for the frame before and after the second farthest cut
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_individual_histogram(frames[second_index], f'Video {video_number} - Histogram Before Second Farthest Cut')
    plt.subplot(1, 2, 2)
    plot_individual_histogram(frames[second_index + 1], f'Video {video_number} - Histogram After Second Farthest Cut')
    plt.show()


def plot_individual_histogram(frame, title):
    """
    Plots the normalized histogram of a single frame.
    :param frame: The frame for which to plot the histogram.
    :param title: Title for the histogram plot.
    """
    # Flatten the frame to a 1D array if it's not already
    flattened_frame = frame.flatten()

    # Compute the histogram
    histogram, bin_edges = np.histogram(flattened_frame, bins=NUMBER_OF_INTENSITY_LEVELS, range=(0, 255))

    # Normalize the histogram
    total_pixels = flattened_frame.size
    normalized_histogram = histogram / total_pixels

    # Calculate the center of each bin
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.bar(bin_centers, normalized_histogram, align='center', width=1)
    plt.title(title)
    plt.xlabel('Intensity Level')
    plt.ylabel('Normalized Pixel Count')
    plt.grid(True)
    plt.xlim(0, 255)


def _plot_cumulative_histograms_of_consecutive_frames(frames, first_index, second_index, video_number):
    """
    Plots side-by-side cumulative histograms for the 'before' and 'after' frames of the farthest and second-farthest cuts.
    :param frames: An array of all frames in the video.
    :param first_index: Index of the frame with the farthest consecutive distance.
    :param second_index: Index of the frame with the second farthest consecutive distance.
    :param video_number: Number of the video in the sequence.
    """
    # Plot cumulative histograms for the frame before and after the farthest cut
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_individual_cumulative_histogram(frames[first_index],
                                         f'Video {video_number} - Cumulative Histogram Before Farthest Cut')
    plt.subplot(1, 2, 2)
    plot_individual_cumulative_histogram(frames[first_index + 1],
                                         f'Video {video_number} - Cumulative Histogram After Farthest Cut')
    plt.show()

    # Plot cumulative histograms for the frame before and after the second farthest cut
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_individual_cumulative_histogram(frames[second_index],
                                         f'Video {video_number} - Cumulative Histogram Before Second Farthest Cut')
    plt.subplot(1, 2, 2)
    plot_individual_cumulative_histogram(frames[second_index + 1],
                                         f'Video {video_number} - Cumulative Histogram After Second Farthest Cut')
    plt.show()


def plot_individual_cumulative_histogram(frame, title):
    """
    Plots the normalized cumulative histogram of a single frame.
    :param frame: The frame for which to plot the histogram.
    :param title: Title for the histogram plot.
    """
    flattened_frame = frame.flatten()
    histogram, bin_edges = np.histogram(flattened_frame, bins=NUMBER_OF_INTENSITY_LEVELS, range=(0, 255))
    cumulative_histogram = np.cumsum(histogram)
    normalized_cumulative_histogram = cumulative_histogram / flattened_frame.size
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.bar(bin_centers, normalized_cumulative_histogram, align='center', width=1)
    plt.title(title)
    plt.xlabel('Intensity Level')
    plt.ylabel('Normalized Cumulative Pixel Count')
    plt.grid(True)
    plt.xlim(0, 255)


def process_video_and_plot(videos):
    """
    Process the videos and plot the results
    :param videos: list of tuples of video paths and their types
    """
    for i, (video_path, video_type) in enumerate(videos, start=1):
        print(f"Processing video number {i} of category {video_type}")
        rgb_frames = read_video_in_rgb(video_path)
        grayscale_frames = read_video_in_grayscale(video_path)
        distances = compute_consecutive_frames_distance(grayscale_frames)
        cut_frame_index = find_scene_cut(grayscale_frames)
        second_max_diff_index = _find_index_of_second_farthest_consecutive_frames(grayscale_frames)

        _plot_consecutive_frames_distances(cut_frame_index, distances, video_type, i)
        _plot_scene_cut_frames(rgb_frames, cut_frame_index, i)
        _plot_second_highest_distance_frame(rgb_frames, second_max_diff_index, i)
        _plot_histograms_of_consecutive_frames(grayscale_frames, cut_frame_index, second_max_diff_index, i)
        _plot_cumulative_histograms_of_consecutive_frames(grayscale_frames, cut_frame_index, second_max_diff_index, i)
        time.sleep(1)


def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number
    for which the scene cut was detected (i.e. the last frame
    index of the first scene and the first frame index of the second scene)
    """
    grayscale_frames = read_video_in_grayscale(video_path)
    cut_frame_index = find_scene_cut(grayscale_frames)
    return cut_frame_index, cut_frame_index + 1
