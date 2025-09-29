import argparse
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt

def apply_noise_limit(input, output, limit_threshold_percent):
    """
    Reads a WAV file, applies amplitude noise limiting (clipping) based 
    on a percentage threshold, saves the result to a new WAV file, and 
    displays the limited audio stream (waveform) using Matplotlib. Assumes mono signal.

    Args:
        input (str): Path to the input WAV file.
        output (str): Path to the output WAV file.
        limit_threshold_percent (float): The maximum allowed amplitude 
                                         as a percentage (e.g., 50.0 for 50%).
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
    if sample_width != 2:
        print(f"Error: This script is optimized for 16-bit audio (2 bytes/sample). Found {sample_width} bytes/sample.")
        sys.exit(1)
        
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # 3. Handle channels (assume mono for processing/limiting)
    if n_channels > 1:
        # Reshape array to [frames, channels] and select the first column (channel 0)
        audio_array = audio_array.reshape(-1, n_channels)[:, 0]

    # --- Noise Limiting (Clipping) Implementation ---
    
    # Get the maximum possible 16-bit integer value (32767)
    max_possible_amplitude = np.iinfo(np.int16).max
    
    # 4. Calculate the clipping value based on the threshold percentage
    # If limit_threshold_percent is 80, clip_value will be 0.8 * 32767
    clip_value = max_possible_amplitude * (limit_threshold_percent / 100.0)
    
    # 5. Apply clipping to limit the amplitude
    # Values > clip_value are set to clip_value
    # Values < -clip_value are set to -clip_value
    limited_array_int16 = np.clip(audio_array, -clip_value, clip_value).astype(np.int16)


    # 6. Write the limited array to the output WAV file
    try:
        with wave.open(output, 'wb') as owf:
            # Write as mono (1 channel)
            owf.setnchannels(1) 
            owf.setsampwidth(sample_width) 
            owf.setframerate(frame_rate)
            # Write the byte data
            owf.writeframes(limited_array_int16.tobytes())
            print(f"Successfully saved noise-limited audio to: {output}")
            
    except Exception as e:
        print(f"Error writing output WAV file: {e}")
        sys.exit(1)

    # 7. Plot the limited waveform
    N = len(limited_array_int16) # Number of samples
    time_axis = np.linspace(0, N / frame_rate, N, endpoint=False)

    plt.figure(figsize=(12, 6))
    
    # Plot amplitude vs time
    plt.plot(time_axis, limited_array_int16, color='#DC3545', linewidth=0.5)
    
    # Styling the plot
    plt.title(f"Noise-Limited Waveform (Max Amplitude set to {limit_threshold_percent:.1f}%)\nInput: {input} | Output: {output}", fontsize=16, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel("Amplitude (16-bit PCM)", fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An Audio Limiting Tool")
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
        "--threshold",
        "-t",
        type=float,
        default=0,
        help="Noise cutoff threshold (%)"
    )

    parser.add_argument(
        "--high",
        "-h",
        type=int,
        default=10000,
        help="Maximum Frequency (Hz)"
    )

    args = parser.parse_args()

    apply_noise_limit(input=args.input, output=args.output, limit_threshold_percent=args.threshold)
