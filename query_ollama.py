#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Queries a local Ollama instance with a given prompt and model.
"""

import ollama
import argparse
import sys

# --- Parameters ---
DEFAULT_MODEL = "gemma3:1b-it-qat"
DEFAULT_PROMPT = "tell me a haiku about national security"

def query_ollama(prompt, model_name):
    """
    Sends a prompt to the specified Ollama model and returns the response.

    Args:
        prompt (str): The input prompt for the model.
        model_name (str): The name of the Ollama model to use.

    Returns:
        str: The content of the model's response, or None if an error occurs.
    """
    print(f"--- Querying Ollama ---")
    print(f"Model: {model_name}")
    print(f"Prompt: '{prompt}'")
    print("-----------------------")

    try:
        # Use ollama.chat for instruction-following/chat models
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'user', 'content': prompt},
            ]
        )
        # The response object is a dictionary, we want the message content
        if response and 'message' in response and 'content' in response['message']:
            return response['message']['content'].strip()
        else:
            print("Error: Received unexpected response format from Ollama.")
            print(f"Full response: {response}")
            return None

    except ollama.ResponseError as e:
        print(f"Error interacting with Ollama model '{model_name}': {e}")
        if "model not found" in str(e):
            print(f"Ensure the model '{model_name}' is available locally. You might need to run: ollama pull {model_name}")
        else:
            print(f"Error details: {e.error}")
        return None
    except Exception as e:
        # Catch potential connection errors etc.
        print(f"An unexpected error occurred: {e}")
        print("Ensure the Ollama service is running locally.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Query a local Ollama model.")
    parser.add_argument("prompt", nargs='?', default=DEFAULT_PROMPT,
                        help=f"The prompt to send to the model (default: '{DEFAULT_PROMPT}')")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"The Ollama model name to use (default: '{DEFAULT_MODEL}')")
    args = parser.parse_args()

    result = query_ollama(args.prompt, args.model)

    if result:
        print("\n--- Ollama Response ---")
        print(result)
        print("-----------------------")
    else:
        print("\nFailed to get response from Ollama.")
        sys.exit(1) # Exit with error code if failed

if __name__ == "__main__":
    main()
