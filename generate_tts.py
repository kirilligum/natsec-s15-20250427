#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uses pyttsx3 to generate a WAV audio file from text.
"""

import pyttsx3
import argparse
import os

# --- Parameters ---
DEFAULT_TEXT = "saying this text in a voice"
DEFAULT_OUTPUT_FILE = "output.wav"

def generate_tts(text_to_speak, output_filename):
    """
    Generates speech from text and saves it to a WAV file.

    Args:
        text_to_speak (str): The text to convert to speech.
        output_filename (str): The path to save the output WAV file.
    """
    print(f"Initializing TTS engine...")
    try:
        engine = pyttsx3.init()
    except Exception as e:
        print(f"Error initializing pyttsx3 engine: {e}")
        print("Ensure you have the necessary TTS engine backends installed (e.g., espeak, nsss, sapi5).")
        print("On Debian/Ubuntu, try: sudo apt-get update && sudo apt-get install espeak")
        print("On Fedora, try: sudo dnf install espeak")
        return False

    # --- Engine Configuration (Optional) ---
    # Rate
    rate = engine.getProperty('rate')   # getting details of current speaking rate
    # print (f"Default rate: {rate}")
    engine.setProperty('rate', 150)     # setting up new voice rate (adjust as needed)

    # Volume
    volume = engine.getProperty('volume') # getting to know current volume level (min=0 max=1)
    # print (f"Default volume: {volume}")
    engine.setProperty('volume', 1.0)    # setting up volume level between 0 and 1

    # Voice (Optional: List available voices and select one)
    # voices = engine.getProperty('voices')
    # for voice in voices:
    #     print(f"Voice ID: {voice.id}, Name: {voice.name}, Lang: {voice.languages}")
    # Set a specific voice ID if desired
    # engine.setProperty('voice', voices[0].id) # Example: Set to the first available voice

    print(f"Generating speech for: '{text_to_speak}'")
    print(f"Saving audio to: {output_filename}")

    try:
        # Ensure the directory exists if the path includes one
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # Save to file
        engine.save_to_file(text_to_speak, output_filename)

        # Wait for the speech synthesis to complete
        engine.runAndWait()
        engine.stop()
        print("Speech generation complete.")
        return True

    except Exception as e:
        print(f"Error during speech generation or saving: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate a WAV file from text using TTS.")
    parser.add_argument("-t", "--text", default=DEFAULT_TEXT,
                        help=f"Text to convert to speech (default: '{DEFAULT_TEXT}')")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_FILE,
                        help=f"Output WAV file name (default: '{DEFAULT_OUTPUT_FILE}')")
    args = parser.parse_args()

    if generate_tts(args.text, args.output):
        print(f"Successfully created '{args.output}'")
    else:
        print(f"Failed to create '{args.output}'")

if __name__ == "__main__":
    main()
