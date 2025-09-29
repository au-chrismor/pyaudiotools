import argparse
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal # Import scipy for signal processing (filtering)

def apply_bandpass_filter(input, output, low, high):
    """
    Reads a WAV file, applies a bandpass filter between the specified 
    cutoff frequencies, saves the result to a new WAV file, and displays 
    the filtered audio stream (waveform) using Matplotlib. Assumes mono signal.

    Args:
        input_filename (str): Path to the input WAV file.
        output_filename (str): Path to the output WAV file.
        low_cutoff (float): Lower frequency cutoff (in Hz).
        high_cutoff (float): Upper frequency cutoff (in Hz).
    """
    # 1. Read the WAV file
    try:
        with wave.open(input, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read all audio frames as byte data
            audio_data = wf.readframes(n_frames)
        
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input}'")
        sys.exit(1)
    except wave.Error as e:
        print(f"Error reading WAV file: {e}")
        sys.exit(1)

    # 2. Convert byte data to a NumPy array of signed 16-bit integers (typical for WAV)
    # Assumes 16-bit audio (standard for many WAVs); adjustment needed for other sample widths
    if sample_width != 2:
        print(f"Error: This script is optimized for 16-bit audio (2 bytes/sample). Found {sample_width} bytes/sample.")
        sys.exit(1)
        
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # 3. Handle channels (assume mono for processing/filtering)
    if n_channels > 1:
        # Reshape array to [frames, channels] and select the first column (channel 0)
        audio_array = audio_array.reshape(-1, n_channels)[:, 0]

    # --- Bandpass Filter Implementation ---
    nyquist = frame_rate / 2.0
    
    # Check frequency boundaries
    if high >= nyquist:
        print(f"Error: High cutoff frequency ({high} Hz) must be less than the Nyquist frequency ({nyquist} Hz).")
        sys.exit(1)
        
    if low >= high:
        print("Error: Low cutoff frequency must be less than the high cutoff frequency.")
        sys.exit(1)

    # Calculate normalized cutoff frequencies (0 to 1, where 1 is the Nyquist frequency)
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    # 4. Design the Butterworth Bandpass filter (using Order 5 for a sharp cutoff)
    N = 5 
    b, a = signal.butter(N, [low_norm, high_norm], btype='band', analog=False)
    
    # 5. Apply the filter to the signal
    # lfilter uses the filter coefficients (b and a) to filter the array
    filtered_array_float = signal.lfilter(b, a, audio_array)

    # 6. Prepare filtered data for WAV writing (clip and convert back to 16-bit integer)
    # Clipping prevents overflow errors when saving to 16-bit
    max_val = np.iinfo(np.int16).max
    min_val = np.iinfo(np.int16).min
    
    # Clip the float data to the 16-bit range and cast it back to the required format
    filtered_array_int16 = np.clip(filtered_array_float, min_val, max_val).astype(np.int16)


    # 7. Write the filtered array to the output WAV file
    try:
        with wave.open(output, 'wb') as owf:
            # Use original file parameters (sample rate, width, and channels)
            owf.setnchannels(1) # Write as mono since we processed only one channel
            owf.setsampwidth(sample_width) 
            owf.setframerate(frame_rate)
            # Write the byte data
            owf.writeframes(filtered_array_int16.tobytes())
            print(f"Successfully saved filtered audio to: {output}")
            
    except Exception as e:
        print(f"Error writing output WAV file: {e}")
        sys.exit(1)

    # 8. Plot the filtered waveform
    N = len(filtered_array_int16) # Number of samples
    time_axis = np.linspace(0, N / frame_rate, N, endpoint=False)

    plt.figure(figsize=(12, 6))
    
    # Plot amplitude vs time
    plt.plot(time_axis, filtered_array_int16, color='#007ACC', linewidth=0.5)
    
    # Styling the plot
    plt.title(f"Filtered Waveform ({low} Hz to {high} Hz)\nInput: {input} | Output: {output}", fontsize=16, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel("Amplitude (16-bit PCM)", fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python audio_spectrum_analyzer.py <input_file.wav> <output_file.wav> <low_cutoff_Hz> <high_cutoff_Hz>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        low_cut = float(sys.argv[3])
        high_cut = float(sys.argv[4])
    except ValueError:
        print("Error: Cutoff frequencies must be numeric values.")
        sys.exit(1)

    if low_cut < 0 or high_cut < 0:
        print("Error: Cutoff frequencies cannot be negative.")
        sys.exit(1)
            
    apply_bandpass_filter(input_file, output_file, low_cut, high_cut)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An Audio Bandpass Filter Tool")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input file"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file"
    )

    parser.add_argument(
        "--low",
        "-l",
        type=int,
        default=0,
        help="Minimum Frequency (Hz)"
    )

    parser.add_argument(
        "--high",
        "-h",
        type=int,
        default=10000,
        help="Maximum Frequency (Hz)"
    )

    args = parser.parse_args()

    apply_bandpass_filter(input=args.input, output=args.output, low=args.low, high=args.high)
