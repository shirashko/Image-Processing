import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf

# We need to choose a window size that divides in the number of samples (2000). This is the closest number to
# the default window size that divides in 2000.
Q2_WINDOW_SIZE = 250


def load_audio(file_path):
    """
    Load an audio file and return its sample rate and audio data.

    Parameters:
    - file_path (str): Path to the audio file.

    Returns:
    - tuple: (sample_rate, audio) where 'sample_rate' is an integer representing the audio sample rate,
             and 'audio' is a numpy array containing the audio data.
        """
    return wav.read(file_path)


def generate_spectrogram(audio, sample_rate):
    """
   Generate a spectrogram from an audio signal.

   Parameters:
   - audio (numpy array): The audio data.
   - sample_rate (int): The sample rate of the audio.

   Returns:
   - tuple: (frequencies, times, spectrogram) where 'frequencies' and 'times' are numpy arrays representing
            the frequencies and times in the spectrogram, and 'spectrogram' is a 2D numpy array of the spectrogram data.
       """
    frequencies, times, spectrogram = scipy.signal.spectrogram(audio, fs=sample_rate)
    return frequencies, times, spectrogram


def generate_and_display_spectrogram(audio, sample_rate, title):
    """
    Generates, displays, and returns the spectrogram of an audio signal.

    Parameters:
    - audio (numpy array): Audio data array.
    - sample_rate (int): Sample rate of the audio in Hertz.
    - title (str): Title for the spectrogram plot.

    Returns:
    - frequencies (numpy array): Frequencies present in the audio (Hz).
    - times (numpy array): Time segments for the spectrogram (seconds).
    - spectrogram (2D numpy array): Amplitude or power of frequencies at each time segment.

    Note:
    This function plots the spectrogram for visualization.
    """
    # Generate the spectrogram
    frequencies, times, spectrogram = scipy.signal.spectrogram(audio, fs=sample_rate, nperseg=Q2_WINDOW_SIZE)

    # Convert to dB scale to avoid log of zero, if required
    # spectrogram = 10 * np.log10(np.abs(spectrogram) + 1e-10)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f"Audio {title} Spectrogram in Log Scale")
    plt.title(f"Audio {title} Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    return frequencies, times, spectrogram


def plot_centered_fft_magnitude(sample_rate, audio):
    """
    Compute and plot the centered magnitude of the Fourier Transform of an audio signal,
    highlighting the peak frequency and the conjunction frequencies at +/- 1060 Hz.

    Parameters:
    sample_rate (int): Sample rate of the audio.
    audio (numpy array): Audio data.
    """
    fft_result = np.fft.fft(audio)
    frequencies = np.fft.fftfreq(len(audio), d=1 / sample_rate)
    fft_result_shifted = np.fft.fftshift(fft_result)
    frequencies_shifted = np.fft.fftshift(frequencies)

    peak_index = np.argmax(np.abs(fft_result_shifted))
    peak_frequency = frequencies_shifted[peak_index]

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies_shifted, np.abs(fft_result_shifted), color='blue')

    # Highlighting the conjunction frequencies
    conjunction_freq = int(np.abs(peak_frequency))
    plt.axvline(x=conjunction_freq, color='green', linestyle='--')
    plt.text(conjunction_freq, np.max(np.abs(fft_result_shifted)) * 0.9, f'{conjunction_freq} Hz', color='green',
             verticalalignment='bottom', fontsize=10)
    plt.axvline(x=-conjunction_freq, color='green', linestyle='--')
    plt.text(-conjunction_freq, np.max(np.abs(fft_result_shifted)) * 0.9, f'-{conjunction_freq} Hz', color='green',
             verticalalignment='bottom', fontsize=10)

    # Setting frequency labels as integers and enhancing readability
    freq_labels = np.arange(int(frequencies_shifted.min()), int(frequencies_shifted.max()) + 1, 500)
    plt.xticks(freq_labels, rotation=45)
    plt.xlim(frequencies_shifted.min(), frequencies_shifted.max())

    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [Hz]')
    plt.title('Centered FFT Magnitude Spectrum')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_audio(file_path, sample_rate, audio):
    """
    Save an audio file.

    Parameters:
    file_path (str): Path where the audio file will be saved.
    sample_rate (int): Sample rate of the audio.
    audio (numpy array): Audio data to be saved.
    """
    sf.write(file_path, audio, sample_rate)


def find_peak_frequency(audio, sample_rate):
    """
    Find the frequency with the highest magnitude in the audio signal's Fourier Transform.

    Parameters:
    sample_rate (int): Sample rate of the audio.
    audio (numpy array): Audio data.

    Returns:
    float: Frequency of the peak magnitude.
    """
    fft_result = np.fft.fft(audio)
    frequencies = np.fft.fftfreq(len(audio), d=1 / sample_rate)
    fft_result_shifted = np.fft.fftshift(fft_result)
    frequencies_shifted = np.fft.fftshift(frequencies)

    peak_index = np.argmax(np.abs(fft_result_shifted))
    peak_frequency = frequencies_shifted[peak_index]
    return abs(peak_frequency)


def apply_peak_denoising_filter(audio):
    """
    Apply a denoising filter to the audio by zeroing out the peak frequency and its negative counterpart.

    Parameters:
    audio (numpy array): Audio data to be denoised.

    Returns:
    numpy array: Denoised audio data.
    """
    fft_result = np.fft.fft(audio)
    peak_index = np.argmax(np.abs(fft_result))

    # Apply the filter
    fft_result[peak_index] = 0
    fft_result[-peak_index] = 0  # The negative peak is the same distance from the end of the array as the positive peak
    denoised_audio = np.fft.ifft(fft_result).real  # When working with images when returning to the standard base the
    # complex should be ignored

    return denoised_audio


def exploration_of_the_audio(audio, sample_rate, title):
    """
    Explore and display the characteristics of an audio signal.

    Parameters:
    audio (numpy array): Audio data.
    sample_rate (int): Sample rate of the audio.
    title_prefix (str): Prefix for the plot title.
    """
    generate_and_display_spectrogram(audio, sample_rate, title)
    plot_centered_fft_magnitude(sample_rate, audio)
    peak_freq = find_peak_frequency(audio, sample_rate)
    print(f"The frequency containing the peak is: {peak_freq:.2f} Hz")


def save_audio_file(audio_path, denoised_audio, sample_rate, file_name):
    """
    Save a denoised audio file.

    Parameters:
    audio_path (str): Original path of the audio file.
    denoised_audio (numpy array): Denoised audio data.
    sample_rate (int): Sample rate of the audio.
    file_name (str): Suffix for the new file name.

    Returns:
    str: Path of the saved denoised audio file.
    """
    output_path = audio_path.replace('.wav', file_name)
    save_audio(output_path, sample_rate, denoised_audio)
    return output_path


def show_mean_amplitude_in_range(frequencies, times, Zxx, sample_rate):
    """
    Plot the mean amplitude in the 550-650 Hz frequency range for each window.

    :param frequencies: Frequencies array from STFT.
    :param times: Times array from STFT.
    :param Zxx: STFT result.
    :param sample_rate: Sample rate of the audio.
    """
    # Identify frequency indices in the 550-650 Hz range
    freq_indices = np.where((frequencies >= 550) & (frequencies <= 650))[0]

    # Compute mean amplitude for the specified frequency range across all time windows
    mean_amplitude = np.mean(np.abs(Zxx[freq_indices, :]), axis=0)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(times, mean_amplitude)
    plt.title('Mean Amplitude in 550-650 Hz Range')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Amplitude')
    plt.show()


def show_amplitude_variance_in_range(frequencies, times, Zxx, sample_rate):
    """
    Plot the amplitude variance in the 550-650 Hz frequency range for each window.

    :param frequencies: Frequencies array from STFT.
    :param times: Times array from STFT.
    :param Zxx: STFT result.
    :param sample_rate: Sample rate of the audio.
    """
    # Identify frequency indices in the 550-650 Hz range
    freq_indices = np.where((frequencies >= 550) & (frequencies <= 650))[0]

    # Compute variance in amplitude for the specified frequency range across all time windows
    amplitude_variance = np.var(np.abs(Zxx[freq_indices, :]), axis=0)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(times, amplitude_variance)
    plt.title('Amplitude Variance in 550-650 Hz Range')
    plt.xlabel('Time (s)')
    plt.ylabel('Variance')
    plt.show()


def q1(audio_path) -> np.array:
    """
    Process, denoise, and return the denoised version of audio Q1 based on the Fourier Transform algorithm.
    Zero out the peak frequency and its negative counterpart to remove the noise.
    :param audio_path: path to q1 audio file
    :return: return q1 denoised version
    """

    sample_rate, audio = load_audio(audio_path)

    # exploration_of_the_audio(audio, sample_rate, title='Q1')

    denoised_audio = apply_peak_denoising_filter(audio)
    # save_audio_file(audio_path, denoised_audio, sample_rate, '_denoised.wav')

    # Check the denoised audio for verification
    # exploration_of_the_audio(denoised_audio, sample_rate, 'Denoised Q')
    return denoised_audio


def q2(audio_path) -> np.array:
    """
    Process, denoise, and return the denoised version of audio Q2 based on the STFT algorithm.
    Only apply denoising in the 550-650 Hz range for time windows between 1.5 and 4 seconds to remove the noise.

    :param audio_path: path to q2 audio file
    :return: return q2 denoised version
    """

    sample_rate, audio = load_audio(audio_path)
    # generate_and_display_spectrogram(audio, sample_rate, 'Q2 spectogram')

    # Compute the STFT of the audio
    frequencies, times, Zxx = scipy.signal.stft(audio, fs=sample_rate, nperseg=Q2_WINDOW_SIZE)

    # Identify the time windows between 1.5 and 4 seconds
    time_indices = np.argwhere((times >= 1.5) & (times <= 4)).flatten()

    # Suppress noise in the 550-650 Hz range only for the selected time windows
    noise_band = np.argwhere((frequencies >= 550) & (frequencies <= 650)).flatten()

    # show_amplitude_variance_in_range(frequencies, times, Zxx, sample_rate)
    # show_mean_amplitude_in_range(frequencies, times, Zxx, sample_rate)

    for time_index in time_indices:
        Zxx[noise_band, time_index] = 0

    # Compute the inverse STFT to get the denoised audio
    _, denoised_audio = scipy.signal.istft(Zxx, fs=sample_rate)

    # save_audio_file(audio_path, denoised_audio, sample_rate, '_denoised.wav')

    # Display the spectrogram of the denoised audio for verification
    # generate_and_display_spectrogram(denoised_audio, sample_rate, 'Denoised Q2')

    return denoised_audio
