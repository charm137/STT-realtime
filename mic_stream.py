#----------------------------------------------------------
#  STT-realtime
#  Copyright (c) 2025 Akshay Mishra
#  This file is part of the Hikari project. You may modify and distribute this file 
#  under the terms of the MIT license. See LICENSE.md in the project root for license information.
#----------------------------------------------------------
import pyaudio
import platform
import torch
import numpy as np
import threading
import queue
from collections import deque
import requests
import time
import datetime
import json
import os
import sys

# --- Audio Streaming Configuration ---
# You may need to adjust these values based on your microphone.
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16384

# LM Studio API Configuration
# By default, LM Studio runs a local server on port 1234.
# Make sure LM Studio is running and the Gemma model is loaded.
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# The hop size: how often the transcription process is triggered.
PROCESSING_HOP_SECONDS = 3
# The window size: the amount of audio (in seconds) that Whisper processes.
WHISPER_WINDOW_SECONDS = 4
# Set the no speech threshold to reduce hallucinations (e.g., "Thank you for watching.")
# The default value for this is 0.6. A higher value requires more confidence to output text.
WHISPER_NO_SPEECH_THRESHOLD = 0.6
# This threshold filters out transcribed segments based on their average log probability. 
# Lower values allow more uncertain transcriptions, while higher values enforce stricter filtering.
WHISPER_LOG_PROB_THRESHOLD = -0.4

# --- Model Configuration ---
# Load the Whisper large-v3 model for transcription.
# This model is more accurate than "base" but requires more resources.
WHISPER_MODEL = "large-v3"
WHISPER_MODEL_MLX = "mlx-community/whisper-large-v3-mlx"

# --- Model Configuration ---
# Define a custom directory to load/save the Whisper models.
# If this directory is empty or does not exist, the model will be downloaded here.
# You can set this to `None` to use the default cache location.
WHISPER_MODEL_DIR = "./whisper_models"

if WHISPER_MODEL_DIR and not os.path.exists(WHISPER_MODEL_DIR):
    os.makedirs(WHISPER_MODEL_DIR)
    print(f"Created custom Whisper model directory: {WHISPER_MODEL_DIR}")

# --- Dynamic Library Loading based on Device ---
# Prioritize CUDA, then MPS, and finally default to CPU.
if torch.cuda.is_available():
    device = "cuda"
    print("CUDA device detected. Using the original whisper module.")
    use_mlx_whisper = False
elif torch.backends.mps.is_available():
    device = "mps"
    try:
        import mlx_whisper
        import mlx.core as mx
        print("MPS device detected. Using mlx-whisper for optimized performance.")
        use_mlx_whisper = True
    except ImportError:
        print("mlx-whisper not found. Falling back to the original whisper module.")
        import whisper
        use_mlx_whisper = False
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = "xpu"
    print("Intel GPU (XPU) device detected. Using the original whisper module.")
    use_mlx_whisper = False
else:
    device = "cpu"
    print("No CUDA, MPS, or XPU device detected. Using the original whisper module on CPU.")
    use_mlx_whisper = False


# Conditionally import the standard whisper module if not using mlx-whisper
if not use_mlx_whisper:
    import whisper


# Load the Whisper model for transcription onto the selected device and with float16.
try:
    if use_mlx_whisper:
        # For mlx-whisper, we pass the path directly to the transcribe function.
        # It handles downloading the model to this path if it doesn't exist.
        whisper_model = WHISPER_MODEL_MLX
        #print(f"MLX Whisper large-v3 model prepared successfully.")
    else:
        # Fallback to the original whisper module for other devices or if mlx-whisper is not installed.
        print(f"Loading Whisper {WHISPER_MODEL}...")
        whisper_model = whisper.load_model(WHISPER_MODEL, device=device, download_root=WHISPER_MODEL_DIR)
        print(f"Original Whisper {WHISPER_MODEL} loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    exit()

# --- Shared Data and Threading ---
audio_queue = queue.Queue()
audio_frames = deque()
is_recording = True
# Track the last processed position to manage the hop
last_processed_position = 0
# Global variable for the log file
log_file = None

# --- PyAudio Callback Function ---
def audio_callback(in_data, frame_count, time_info, status):
    """
    This function is called continuously by PyAudio in a separate thread.
    It adds incoming audio data to a queue.
    """
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

# --- Processing Thread Function ---
def process_audio():
    """
    This function runs in a separate thread, pulling audio data
    from the queue, transcribing, and translating.
    """
    global last_processed_position
    global log_file
    
    last_japanese_text = ""

    while is_recording or not audio_queue.empty():
        # Check if we have enough new data to trigger a new processing step
        current_audio_length = len(audio_frames) * CHUNK
        if current_audio_length - last_processed_position >= RATE * PROCESSING_HOP_SECONDS:            
            
            # Combine the frames into a single NumPy array
            audio_data = b''.join(list(audio_frames))
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # --- Transcription with Whisper ---
            try:
                # Get the current date and time for the output
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Determine the end of the window
                end_of_window = len(audio_array)
                # Determine the start of the window, ensuring it doesn't go below 0
                start_of_window = max(0, end_of_window - RATE * WHISPER_WINDOW_SECONDS)
                
                # This is the 5-second sliding window that gets transcribed
                segment_to_transcribe = audio_array[start_of_window:end_of_window]
                # IMPORTANT: Convert the numpy array to a float32 PyTorch tensor on the correct device.
                audio_tensor = torch.from_numpy(segment_to_transcribe).to(device=device, dtype=torch.float32)
                

                if use_mlx_whisper:
                    # Transcribe using mlx-whisper. Note the API differences.
                    # path_or_hf_repo can be a local path or a Hugging Face model ID.
                    result = mlx_whisper.transcribe(segment_to_transcribe, path_or_hf_repo=whisper_model, language="ja")
                else:
                    # Transcribe using original whisper model             
                    result = whisper_model.transcribe(audio_tensor, language="ja"
                                                  , no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD
                                                  , logprob_threshold=WHISPER_LOG_PROB_THRESHOLD
                                                  #, initial_prompt="Meeting transcript:"
                                                  )                
                japanese_text = result["text"].strip()
                
                if japanese_text and japanese_text != last_japanese_text and japanese_text != "ご視聴ありがとうございました":
                    japanese_output = f"[{current_time}] JP: {japanese_text}"
                    print(japanese_output)
                    if log_file:
                        log_file.write(japanese_output + "\n")
                    last_japanese_text = japanese_text
                    last_japanese_text = japanese_text

                    # --- Translation with LM Studio ---
                    #print("Translating with LM Studio...")
                    try:
                        headers = {"Content-Type": "application/json"}
                        payload = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant that translates Japanese text to English."
                                },
                                {
                                    "role": "user",
                                    "content": f"Translate the following Japanese text to English. Do not provide any explanation or any other text: {japanese_text}"
                                }
                            ],
                            "temperature": 0.6,
                            "max_tokens": 100
                        }
                        
                        response = requests.post(LM_STUDIO_URL, headers=headers, json=payload)
                        response.raise_for_status()
                        
                        response_data = response.json()
                        translated_text = response_data['choices'][0]['message']['content'].strip()
                        
                        english_output = f"[{current_time}] ENG: {translated_text}"
                        print(english_output)
                        if log_file:
                            log_file.write(english_output + "\n")
                    except requests.exceptions.RequestException as e:
                        print(f"Error communicating with LM Studio: {e}")
                    except (KeyError, IndexError) as e:
                        print(f"Error parsing LM Studio response: {e}")
            
            except Exception as e:
                print(f"Error during Whisper transcription: {e}")
            
            # Update the last processed position by the hop size
            last_processed_position += RATE * PROCESSING_HOP_SECONDS
            
        # Append new data from the queue to our frames buffer
        try:
            new_data = audio_queue.get_nowait()
            audio_frames.append(new_data)
        except queue.Empty:
            # Sleep briefly to prevent the loop from consuming too much CPU
            time.sleep(0.1)

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the 'requests' library is installed
    try:
        import requests
    except ImportError:
        print("The 'requests' library is not installed. Please install it by running:")
        print("pip install requests")
        exit()

    p = pyaudio.PyAudio()
    
    # --- Command-line argument parsing for audio source ---
    device_index = None
    input_device_name = "Default Microphone"
    # Check for a command-line argument for the audio source
    capture_loopback_device = False
    if "--loopback" in sys.argv:        
        print("Attempting to use the default loopback audio device.")
        input_device_name = "Default Loopback"
        
        try:
            # First, find the default output device
            default_output_device = p.get_default_output_device_info()
            default_output_name = default_output_device["name"]
            print(f"Default output device: {default_output_name}")
            
            # Then, iterate through all devices to find a matching loopback input device
            found_loopback = False
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                print(dev_info["name"])
                if sys.platform == "linux" and "sysdefault" in dev_info["name"]:
                    device_index = i
                    found_loopback = True
                    capture_loopback_device = True                                    
                    break
                if sys.platform == "darwin" and "BlackHole 2ch" in dev_info["name"]:
                    device_index = i
                    found_loopback = True
                    capture_loopback_device = True                                    
                    break


            if not found_loopback:
                print("Warning: Default loopback device not found. Falling back to the default microphone.")
                device_index = None
                input_device_name = "Default Microphone"

        except IOError:
            print("Warning: Could not get default output device info. Falling back to the default microphone.")
            device_index = None
            input_device_name = "Default Microphone"

    # Open the audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_index,
                    stream_callback=audio_callback)
    
    # Create the 'transcripts' directory if it doesn't exist
    transcripts_dir = "transcripts"
    if not os.path.exists(transcripts_dir):
        os.makedirs(transcripts_dir)
        print(f"Created directory: {transcripts_dir}")

    # Check for a command-line argument for the filename prefix.
    # If no argument is provided, default to "Transcript-JA-EN".
    filename_prefix = "Transcript-JA-EN"
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            cmd_arg = sys.argv[i]
            if (cmd_arg != "--loopback"):
                filename_prefix = cmd_arg
                break      

    # Generate a timestamped log file name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = os.path.join(transcripts_dir, f"{filename_prefix}-{timestamp}.txt")
    try:
        # Open the log file in append mode
        log_file = open(log_filename, "a", encoding="utf-8")
        print(f"Logging output to {log_filename}")
    except IOError as e:
        print(f"Error opening log file: {e}")
        log_file = None
    
    # Start the audio processing thread
    processor_thread = threading.Thread(target=process_audio)
    processor_thread.daemon = True
    processor_thread.start()

    print("--- Starting audio capture. Press Ctrl+C to stop. ---")
    
    try:
        # Keep the main thread alive while other threads are running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n--- Stopping audio capture. Please wait for the final processing to complete. ---")
    finally:
        # Stop the recording flag and wait for the processing thread to finish
        is_recording = False
        processor_thread.join()
        
        # Stop and close the audio stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Close the log file
        if log_file:
            log_file.close()
            print(f"Log file '{log_filename}' closed.")
        
    print("Application closed.")
