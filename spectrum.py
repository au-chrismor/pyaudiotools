import argparse
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt

def analyse_audio_spectrum(filename, min_freq=0, max_freq=10000):
    """
    Performs spectrum analysis on a WAV file and displays the magnitude 
    spectrum in a specified frequency range (min_freq_limit to max_freq Hz)
    using Matplotlib.

    Args:
        filename (str): The path to the input WAV file.
        min_freq_limit (int): The minimum frequency (in Hz) to display on the plot.
    """
    try:
        # 1. Read the WAV file
        with wave.open(filename, 'rb') as wf:
            n_channels = wf.getnchannels()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read all audio frames as byte data
            audio_data = wf.readframes(n_frames)
        
    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
        sys.exit(1)
    except wave.Error as e:
        print(f"Error reading WAV file: {e}")
        sys.exit(1)

    # 2. Convert byte data to a NumPy array of signed 16-bit integers (typical for WAV)
    # np.int16 assumes 2 bytes per sample
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # 3. If the audio is stereo (or multi-channel), take only the first channel
    if n_channels > 1:
        # Reshape array to [frames, channels] and select the first column (channel 0)
        audio_array = audio_array.reshape(-1, n_channels)[:, 0]

    N = len(audio_array) # Number of samples for FFT
    
    # 4. Perform the Fast Fourier Transform (FFT)
    # The FFT converts the time-domain signal to the frequency domain
    fft_result = np.fft.fft(audio_array)
    
    # 5. Calculate the single-sided magnitude spectrum
    # We use N/2 + 1 elements because the FFT is symmetric, and we only need the positive frequencies
    P2 = np.abs(fft_result / N) # Normalise by the total number of samples
    P1 = P2[:N // 2 + 1]        # Take the first half (positive frequencies)
    P1[1:-1] = 2 * P1[1:-1]     # Double the magnitude for all non-DC, non-Nyquist frequencies
    
    # 6. Generate the corresponding frequency axis
    freq_positive = np.fft.fftfreq(N, d=1.0/frame_rate)[:N // 2 + 1]

    # 7. Filter the spectrum using both minimum and maximum limits
    
    # Find the index of the first frequency component >= the minimum limit
    min_index = np.argmax(freq_positive >= min_freq)

    # Find the index of the first frequency component > the maximum limit
    max_index = np.argmax(freq_positive > max_freq)
    
    # If all frequencies are below the max limit, set max_index to the end of the array
    if max_index == 0 and freq_positive[-1] > freq_positive[0]:
         max_index = len(freq_positive)

    # Slice the frequency and magnitude arrays
    freq_filtered = freq_positive[min_index:max_index]
    magnitude_filtered = P1[min_index:max_index]
    
    # 8. Plot the spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(freq_filtered, magnitude_filtered, color='#007ACC', linewidth=1.5)
    
    # Styling the plot
    plt.title(f"Audio Spectrum Analysis ({min_freq} - {max_freq} Hz)\nFile: {filename}", fontsize=16, fontweight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Magnitude (Linear Scale)", fontsize=14)
    
    # Ensure the X-axis is explicitly limited by the min and max limits
    plt.xlim(min_freq, max_freq) 
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().ticklabel_format(style='plain', axis='x') # Prevent scientific notation on x-axis
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An Audio Spectrum Analysis Tool")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input file"
    )

    parser.add_argument(
        "--minfrequency",
        "-m",
        type=int,
        default=0,
        help="Minimum Frequency (Hz)"
    )

    parser.add_argument(
        "--maxfrequency",
        "-x",
        type=int,
        default=3000,
        help="Maximum Frequency (Hz)"
    )

    args = parser.parse_args()

    analyse_audio_spectrum(filename=args.input, min_freq=args.minfrequency, max_freq=args.maxfrequency)
