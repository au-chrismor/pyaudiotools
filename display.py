import argparse
import sys
import wave
import numpy as np
import matplotlib.pyplot as plt

def analyse_audio_stream(filename):
    """
    Reads a WAV file and displays the audio stream (waveform) 
    in the time domain using Matplotlib. Assumes mono signal for plotting.

    Args:
        filename (str): The path to the input WAV file.
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
    # This aligns with the request to assume a mono signal for display.
    if n_channels > 1:
        # Reshape array to [frames, channels] and select the first column (channel 0)
        audio_array = audio_array.reshape(-1, n_channels)[:, 0]

    N = len(audio_array) # Number of samples
    
    # 4. Generate the time axis
    # The time is calculated as N / frame_rate (total duration in seconds)
    time_axis = np.linspace(0, N / frame_rate, N, endpoint=False)

    # 5. Plot the waveform
    plt.figure(figsize=(12, 6))
    
    # Plot amplitude vs time
    plt.plot(time_axis, audio_array, color='#28A745', linewidth=0.5)
    
    # Styling the plot
    plt.title(f"Audio Stream Waveform Analysis\nFile: {filename}", fontsize=16, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel("Amplitude (16-bit PCM)", fontsize=14)
    
    # Optional: Limit the view to a short segment for better detail (e.g., first 2 seconds)
    # if N / frame_rate > 2:
    #     plt.xlim(0, 2) 
        
    plt.grid(True, linestyle='--', alpha=0.7)
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

    args = parser.parse_args()
    analyse_audio_stream(filename=args.input)
