#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main workflow script:
1. Takes a text prompt.
2. Queries Ollama for a response.
3. Generates speech from the response using Coqui TTS with voice cloning.
4. Transmits the generated speech using ADALM-PLUTO SDR.
"""

import argparse
import sys
import os

# Import functions from other scripts
from query_ollama import query_ollama, DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL
from generate_tts_cloned import generate_cloned_tts, DEFAULT_REFERENCE_VOICE, DEFAULT_MODEL_NAME as TTS_DEFAULT_MODEL
from transmit_fm import transmit_audio, SDR_URI, CENTER_FREQ, SAMPLE_RATE, TX_GAIN, FM_DEVIATION, AUDIO_TARGET_RATE, CHUNK_SIZE

# --- Workflow Parameters ---
DEFAULT_OUTPUT_WAV = "workflow_output.wav" # Use a different name to avoid conflict if run separately

def run_workflow(prompt, ollama_model, tts_reference_voice, tts_model, output_wav, sdr_params):
    """
    Executes the full workflow: Ollama -> TTS -> SDR Transmission.

    Args:
        prompt (str): The initial text prompt for Ollama.
        ollama_model (str): The Ollama model to use.
        tts_reference_voice (str): Path to the reference WAV for voice cloning.
        tts_model (str): The Coqui TTS model to use.
        output_wav (str): Path to save the intermediate TTS audio file.
        sdr_params (dict): Dictionary containing parameters for transmit_audio.

    Returns:
        bool: True if the entire workflow completed successfully, False otherwise.
    """
    print("--- Starting Main Workflow ---")
    print(f"Input Prompt: '{prompt}'")

    # Step 1: Query Ollama
    print("\n[Step 1/3] Querying Ollama...")
    ollama_response = query_ollama(prompt=prompt, model_name=ollama_model)

    if not ollama_response:
        print("Workflow failed at Ollama step.")
        return False
    print(f"Ollama Response: '{ollama_response}'")

    # Step 2: Generate TTS with Cloned Voice
    print("\n[Step 2/3] Generating Cloned TTS...")
    if not os.path.exists(tts_reference_voice):
        print(f"Error: TTS Reference voice file not found: {tts_reference_voice}")
        print("Workflow failed at TTS step.")
        return False

    tts_success = generate_cloned_tts(
        text_to_speak=ollama_response,
        output_filename=output_wav,
        reference_wav=tts_reference_voice,
        model_name=tts_model
    )

    if not tts_success:
        print("Workflow failed at TTS step.")
        return False
    print(f"TTS audio generated: '{output_wav}'")

    # Step 3: Transmit Audio via SDR
    print("\n[Step 3/3] Transmitting Audio via SDR...")
    transmit_success = transmit_audio(
        audio_file=output_wav,
        sdr_uri=sdr_params['uri'],
        center_freq=sdr_params['freq'],
        sample_rate=sdr_params['rate'],
        tx_gain=sdr_params['gain'],
        fm_deviation=sdr_params['deviation'],
        audio_target_rate=sdr_params['audio_rate'],
        chunk_size=sdr_params['chunk']
    )

    if not transmit_success:
        print("Workflow failed at SDR transmission step.")
        return False

    print("\n--- Workflow Completed Successfully ---")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the Ollama -> TTS -> SDR transmission workflow.")
    parser.add_argument("prompt",
                        help="The initial text prompt to send to Ollama (use quotes).")
    parser.add_argument("--ollama-model", default=OLLAMA_DEFAULT_MODEL,
                        help=f"Ollama model name (default: {OLLAMA_DEFAULT_MODEL})")
    parser.add_argument("--tts-reference", default=DEFAULT_REFERENCE_VOICE,
                        help=f"Reference WAV for voice cloning (default: {DEFAULT_REFERENCE_VOICE})")
    parser.add_argument("--tts-model", default=TTS_DEFAULT_MODEL,
                        help=f"Coqui TTS model name (default: {TTS_DEFAULT_MODEL})")
    parser.add_argument("--output-wav", default=DEFAULT_OUTPUT_WAV,
                        help=f"Intermediate WAV file name (default: {DEFAULT_OUTPUT_WAV})")
    # Add arguments for SDR parameters if needed, otherwise use defaults from transmit_fm
    parser.add_argument("--sdr-uri", default=SDR_URI, help="SDR device URI")
    parser.add_argument("--sdr-freq", type=float, default=CENTER_FREQ, help="SDR frequency (Hz)")
    parser.add_argument("--sdr-rate", type=float, default=SAMPLE_RATE, help="SDR sample rate (Hz)")
    parser.add_argument("--sdr-gain", type=int, default=TX_GAIN, help="SDR TX gain (dB)")
    parser.add_argument("--sdr-deviation", type=float, default=FM_DEVIATION, help="FM deviation (Hz)")


    args = parser.parse_args()

    sdr_parameters = {
        'uri': args.sdr_uri,
        'freq': args.sdr_freq,
        'rate': args.sdr_rate,
        'gain': args.sdr_gain,
        'deviation': args.sdr_deviation,
        'audio_rate': AUDIO_TARGET_RATE, # Keep using the default from transmit_fm for now
        'chunk': CHUNK_SIZE             # Keep using the default from transmit_fm for now
    }

    if run_workflow(args.prompt, args.ollama_model, args.tts_reference, args.tts_model, args.output_wav, sdr_parameters):
        print("Main script finished.")
    else:
        print("Main script finished with errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
