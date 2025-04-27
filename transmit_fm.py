#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transmit an audio file as analog FM using an ADALM-PLUTO SDR.

WARNING: Transmitting radio signals requires adherence to local regulations
         and potentially a license. Ensure you are operating legally and
         responsibly. Start with minimum transmit power.
"""

import adi
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import time
import math

# --- Parameters ---
AUDIO_FILE = "output.wav"
SDR_URI = "ip:192.168.2.1"  # Default Pluto IP. Change if needed.
CENTER_FREQ = 440.135e6  # Target frequency (440.135 MHz)
SAMPLE_RATE = 1e6        # SDR sample rate (e.g., 1 MHz - must be >= 521e3)
TX_GAIN = -10            # TX gain in dB (Start low, e.g., -30 to -10)
FM_DEVIATION = 5e3       # FM deviation (e.g., 5 kHz for narrowband FM)
AUDIO_TARGET_RATE = 48e3 # Intermediate audio rate before final resampling
CHUNK_SIZE = 8192        # Number of samples per transmission chunk

# --- Transmission Function ---
def transmit_audio(audio_file, sdr_uri=SDR_URI, center_freq=CENTER_FREQ,
                   sample_rate=SAMPLE_RATE, tx_gain=TX_GAIN,
                   fm_deviation=FM_DEVIATION, audio_target_rate=AUDIO_TARGET_RATE,
                   chunk_size=CHUNK_SIZE):
    """
    Reads an audio file, performs FM modulation, and transmits it using PlutoSDR.

    Args:
        audio_file (str): Path to the input WAV file.
        sdr_uri (str): URI of the PlutoSDR device.
        center_freq (float): Center frequency for transmission in Hz.
        sample_rate (float): Sample rate for the SDR in Hz.
        tx_gain (int): Transmission gain in dB.
        fm_deviation (float): FM frequency deviation in Hz.
        audio_target_rate (float): Intermediate sample rate for audio before SDR resampling.
        chunk_size (int): Number of samples per transmission chunk.

    Returns:
        bool: True if transmission was successful (or finished), False otherwise.
    """
    print("--- FM Voice Transmission ---")
    print(f"Audio file: {audio_file}")
    print(f"Frequency: {center_freq / 1e6:.3f} MHz")
    print(f"Sample Rate: {sample_rate / 1e3:.0f} kHz")
    print(f"FM Deviation: {fm_deviation / 1e3:.1f} kHz")
    print(f"TX Gain: {tx_gain} dB")
    print(f"Sample Rate: {SAMPLE_RATE / 1e3:.0f} kHz")
    print(f"FM Deviation: {FM_DEVIATION / 1e3:.1f} kHz")
    print(f"TX Gain: {TX_GAIN} dB")
    print("--------------------------")
    print("WARNING: Ensure you comply with local radio regulations.")

    # 1. Read Audio File
    try:
        fs_audio, audio_data = wavfile.read(audio_file)
        print(f"Read audio file: Sample rate {fs_audio} Hz, Duration {len(audio_data)/fs_audio:.2f}s")
    except FileNotFoundError:
        print(f"Error: Audio file '{audio_file}' not found.")
        return False
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return False

    # 2. Preprocess Audio
    # Convert to mono if stereo
    if audio_data.ndim > 1:
        print("Audio is stereo, converting to mono.")
        audio_data = audio_data.mean(axis=1)

    # Convert to float [-1.0, 1.0]
    if np.issubdtype(audio_data.dtype, np.integer):
        # Find the maximum value for the integer type
        try:
            max_val = np.iinfo(audio_data.dtype).max
            min_val = np.iinfo(audio_data.dtype).min
            # Use the larger absolute value for scaling
            scale = max(abs(max_val), abs(min_val))
        except ValueError: # Handle cases like uint64 where iinfo might behave differently or for unsupported types
             print(f"Warning: Could not determine exact range for dtype {audio_data.dtype}. Using max value.")
             scale = np.max(np.abs(audio_data))
             if scale == 0: scale = 1.0 # Avoid division by zero

        audio_data = audio_data.astype(np.float32) / scale
    elif np.issubdtype(audio_data.dtype, np.floating):
         # Assume float is already in a reasonable range, but normalize just in case
         max_abs_val = np.max(np.abs(audio_data))
         if max_abs_val > 0:
             audio_data = audio_data.astype(np.float32) / max_abs_val
         else:
             audio_data = audio_data.astype(np.float32) # Already zero
    else:
        print(f"Warning: Unsupported audio data type {audio_data.dtype}. Attempting direct conversion to float.")
        # Attempt conversion, might fail or be inaccurate
        try:
            audio_data = audio_data.astype(np.float32)
            # Basic normalization if conversion succeeded
            max_abs_val = np.max(np.abs(audio_data))
            if max_abs_val > 0:
                 audio_data = audio_data / max_abs_val
        except (TypeError, ValueError) as e:
            print(f"Error: Could not convert audio data type {audio_data.dtype} to float: {e}")
            return


    # Resample audio to the target rate for modulation
    # Use integer up/down factors for resample_poly for better quality
    gcd = math.gcd(int(audio_target_rate), int(fs_audio))
    up = int(audio_target_rate) // gcd
    down = int(fs_audio) // gcd
    print(f"Resampling audio from {fs_audio} Hz to {audio_target_rate} Hz (up={up}, down={down})...")
    try:
        audio_resampled = resample_poly(audio_data, up, down)
    except Exception as e:
        print(f"Error during audio resampling: {e}")
        return False
    fs_resampled = audio_target_rate
    print("Audio resampling complete.")

    # Normalize again after resampling
    max_abs_resampled = np.max(np.abs(audio_resampled))
    if max_abs_resampled > 0:
        audio_resampled = audio_resampled / max_abs_resampled * 0.95 # Keep headroom
    else:
        audio_resampled = audio_resampled # Already zero

    # 3. FM Modulation
    print("Performing FM modulation...")
    # Calculate sensitivity factor (radians per sample per unit of input)
    sensitivity = 2.0 * np.pi * fm_deviation / fs_resampled
    # Integrate audio signal for phase modulation
    phase = sensitivity * np.cumsum(audio_resampled)
    # Generate complex FM signal e^(j*phase)
    fm_signal = np.exp(1j * phase).astype(np.complex64)
    print("FM modulation complete.")

    # 4. Resample FM signal to SDR sample rate
    gcd_sdr = math.gcd(int(sample_rate), int(fs_resampled))
    up_sdr = int(sample_rate) // gcd_sdr
    down_sdr = int(fs_resampled) // gcd_sdr
    print(f"Resampling FM signal from {fs_resampled} Hz to {sample_rate} Hz (up={up_sdr}, down={down_sdr})...")
    try:
        fm_signal_sdr = resample_poly(fm_signal, up_sdr, down_sdr)
    except Exception as e:
        print(f"Error during FM signal resampling: {e}")
        return False
    print("FM signal resampling complete.")

    # Scale signal amplitude for SDR DAC
    # PlutoSDR expects I/Q samples in the range [-2^15, 2^15-1].
    # pyadi-iio handles scaling, but it's good practice to provide
    # signals roughly in the range [-1, 1] scaled by 2**15.
    # We scale by 0.5 * 2**15 to leave some headroom.
    fm_signal_sdr *= 0.5 * (2**15)

    # 5. Initialize and Configure SDR
    sdr = None # Initialize sdr to None for cleanup check
    transmission_successful = False
    try:
        print(f"Connecting to PlutoSDR at {sdr_uri}...")
        sdr = adi.Pluto(uri=sdr_uri)
        print("PlutoSDR connected.")

        # Configure SDR parameters
        sdr.sample_rate = int(sample_rate)
        sdr.tx_lo = int(center_freq)
        sdr.tx_hardwaregain_chan0 = tx_gain
        sdr.tx_cyclic_buffer = False # We'll send chunks manually

        # Ensure buffer is destroyed before potential use (might not exist)
        try:
             sdr.tx_destroy_buffer()
        except Exception as buf_e:
             print(f"Note: Could not destroy buffer (may not exist yet): {buf_e}")


        print("SDR configured:")
        print(f"  TX LO Freq: {sdr.tx_lo / 1e6:.3f} MHz")
        print(f"  Sample Rate: {sdr.sample_rate / 1e3:.0f} kHz")
        print(f"  TX Gain: {sdr.tx_hardwaregain_chan0} dB")

    except Exception as e:
        print(f"Error initializing SDR: {e}")
        print("Ensure the PlutoSDR is connected and accessible via USB.")
        print("Verify the IP address if using network mode.")
        print("Check if libiio and pyadi-iio drivers/libraries are correctly installed.")
        if sdr: del sdr # Attempt cleanup if object exists partially
        return False # Indicate failure

    # 6. Transmit Data
    total_samples = len(fm_signal_sdr)
    num_chunks = math.ceil(total_samples / chunk_size)
    print(f"Starting transmission of {total_samples} samples in {num_chunks} chunks...")
    print(f"Chunk size: {chunk_size} samples")

    try:
        # Enable TX channel 0
        sdr.tx_enabled_channels = [0]

        start_time = time.time()
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)
            chunk = fm_signal_sdr[start_idx:end_idx]

            # The pyadi library usually handles padding the last chunk if needed,
            # but ensure chunk_size is adequate (e.g., > 1024 and multiple of 4).
            if len(chunk) == 0:
                print("Warning: Generated empty chunk, skipping.")
                continue

            try:
                sdr.tx(chunk)
                # Optional: Add a small sleep if experiencing buffer issues/underruns
                # time.sleep(0.001)
            except Exception as tx_e:
                print(f"Error sending chunk {i+1}/{num_chunks}: {tx_e}")
                # Decide whether to break or continue
                break # Stop transmission on error

            if (i + 1) % 10 == 0 or i == num_chunks - 1: # Print progress periodically
                 print(f"  Sent chunk {i+1}/{num_chunks} ({len(chunk)} samples)")


        # Wait a short moment to ensure the last buffer is transmitted
        time.sleep(0.5)
        end_time = time.time()
        print(f"Transmission finished in {end_time - start_time:.2f} seconds.")
        transmission_successful = True # Mark as finished

    except Exception as e:
        print(f"Error during transmission loop: {e}")
        transmission_successful = False # Mark as failed
    finally:
        # 7. Cleanup
        if sdr:
            print("Cleaning up SDR resources...")
            try:
                # Disable TX first
                sdr.tx_enabled_channels = []
                print("TX disabled.")
                # Destroy buffer
                sdr.tx_destroy_buffer()
                print("TX buffer destroyed.")
            except Exception as cleanup_e:
                print(f"Error during SDR cleanup: {cleanup_e}")
            finally:
                 del sdr # Release SDR object
                 print("SDR object released.")
        else:
            print("SDR not initialized, no cleanup needed.")
        return transmission_successful # Return status

# --- Main Execution Logic (when run as script) ---
def main():
    """Parses arguments and runs the transmission when script is executed directly."""
    # Basic execution using defaults defined at the top of the file
    # More complex argparse could be added here if needed for direct execution flexibility
    print("--- Running FM Transmitter Script ---")
    if transmit_audio(AUDIO_FILE):
         print("Script finished successfully.")
    else:
         print("Script finished with errors.")
         import sys
         sys.exit(1)


if __name__ == "__main__":
    main()
