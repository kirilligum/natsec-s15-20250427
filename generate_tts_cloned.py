#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uses Coqui TTS to generate a WAV audio file from text, cloning the voice
from a reference audio file.
"""

import os
import argparse
import torch
from TTS.api import TTS

# --- Parameters ---
DEFAULT_TEXT = "saying this text in a voice"
DEFAULT_OUTPUT_FILE = "output.wav"
DEFAULT_REFERENCE_VOICE = "kirill_sample_voice.wav"
# Model known for good zero-shot voice cloning
# See available models: https://github.com/coqui-ai/TTS/blob/dev/TTS/server/model_manager.py
# Or run `tts --list_models` in your terminal after installing TTS
DEFAULT_MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"

def generate_cloned_tts(text_to_speak, output_filename, reference_wav, model_name):
    """
    Generates speech from text using a cloned voice and saves it to a WAV file.

    Args:
        text_to_speak (str): The text to convert to speech.
        output_filename (str): The path to save the output WAV file.
        reference_wav (str): Path to the reference WAV file for voice cloning.
        model_name (str): The Coqui TTS model name to use.
    """
    print("--- TTS Voice Cloning ---")
    print(f"Text: '{text_to_speak}'")
    print(f"Reference Voice: {reference_wav}")
    print(f"Output File: {output_filename}")
    print(f"TTS Model: {model_name}")
    print("-------------------------")

    # Check if reference voice file exists
    if not os.path.exists(reference_wav):
        print(f"Error: Reference voice file not found at '{reference_wav}'")
        return False

    # Check if GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: TTS processing will be slower on CPU.")

    try:
        print("Initializing TTS model (may download model on first run)...")
        # Init TTS
        tts = TTS(model_name=model_name, progress_bar=True).to(device)
        print("TTS model initialized.")

        print("Generating speech (this may take a moment)...")
        # Run TTS
        # Assumes English ('en') for the text. Change if needed for multilingual models.
        tts.tts_to_file(
            text=text_to_speak,
            speaker_wav=reference_wav,
            language="en", # Specify language for multilingual models
            file_path=output_filename
        )
        print(f"Speech successfully generated and saved to '{output_filename}'")
        return True

    except FileNotFoundError as e:
         print(f"Error: Model file not found. It might need to be downloaded.")
         print(f"Check your internet connection and TTS installation. Details: {e}")
         return False
    except RuntimeError as e:
        if "module 'torchaudio.functional' has no attribute 'compute_kaldi_pitch'" in str(e):
             print(f"Error: Missing torchaudio dependency or version mismatch. {e}")
             print("Try reinstalling torchaudio or ensuring versions are compatible.")
        elif "CUDA out of memory" in str(e):
             print(f"Error: CUDA out of memory. Try using a smaller model or running on CPU.")
        else:
             print(f"Runtime Error during TTS generation: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate a WAV file from text using voice cloning.")
    parser.add_argument("text",
                        help="Text to convert to speech (use quotes if it contains spaces).")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILE,
                        help=f"Output WAV file name (default: '{DEFAULT_OUTPUT_FILE}')")
    parser.add_argument("-r", "--reference", default=DEFAULT_REFERENCE_VOICE,
                        help=f"Reference WAV file for voice cloning (default: '{DEFAULT_REFERENCE_VOICE}')")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_NAME,
                        help=f"Coqui TTS model name to use (default: '{DEFAULT_MODEL_NAME}')")
    args = parser.parse_args()

    if not os.path.exists(args.reference):
         print(f"Error: Cannot find reference voice file: {args.reference}")
         print("Please ensure the file exists and the path is correct.")
         return # Exit early if reference doesn't exist

    if generate_cloned_tts(args.text, args.output, args.reference, args.model):
        print(f"Successfully created '{args.output}' with cloned voice.")
    else:
        print(f"Failed to create '{args.output}'")

if __name__ == "__main__":
    main()
