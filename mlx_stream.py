#----------------------------------------------------------
#  STT-realtime
#  Copyright (c) 2026 Akshay Mishra
#  You may modify and distribute this file under the terms 
#  of the MIT license. See LICENSE.md in the project root 
#  for license information.
#----------------------------------------------------------
import pyaudio
import numpy as np
import threading
import queue
from collections import deque
import requests
import time
import datetime
import os
import sys
import tempfile
import soundfile as sf
from termcolor import colored

# --- Audio Streaming Configuration ---
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
# ASR standard sample rate
RATE = 16000

# LM Studio API Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

PROCESSING_HOP_SECONDS = 3
ASR_WINDOW_SECONDS = 4

# --- Model Configuration ---
# Load the ASR model
ASR_MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"

if "--voxtral" in sys.argv:
    ASR_MODEL_ID = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
elif "--model" in sys.argv:
    model_idx = sys.argv.index("--model")
    if model_idx + 1 < len(sys.argv):
        ASR_MODEL_ID = sys.argv[model_idx + 1]

SOURCE_LANGUAGE = "ja" # Original script target language

asr_model = None
asr_processor = None
use_mlx_vlm = "gemma-4" in ASR_MODEL_ID.lower() or "vlm" in ASR_MODEL_ID.lower()

try:
    if use_mlx_vlm:
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        print(f"Loading Multimodal model: {ASR_MODEL_ID}...")
        asr_model, asr_processor = load(ASR_MODEL_ID)
        print(f"Model {ASR_MODEL_ID} loaded successfully via mlx-vlm.")
    else:
        from mlx_audio.stt import load
        print(f"Loading ASR model: {ASR_MODEL_ID}...")
        asr_model = load(ASR_MODEL_ID)
        print(f"ASR model {ASR_MODEL_ID} loaded successfully via mlx-audio.")
except ImportError:
    print(f"Error: {'mlx-vlm' if use_mlx_vlm else 'mlx-audio'} is not installed. Please install it using:")
    print(f"pip install {'mlx-vlm' if use_mlx_vlm else 'mlx-audio'}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- Shared Data and Threading ---
audio_queue = queue.Queue()
audio_frames = deque()
is_recording = True
last_processed_position = 0
log_file = None

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

def process_audio():
    global last_processed_position
    global log_file
    
    last_transcribed_text = ""

    while is_recording or not audio_queue.empty():
        current_audio_length = len(audio_frames) * CHUNK
        if current_audio_length - last_processed_position >= RATE * PROCESSING_HOP_SECONDS:            
            
            # Combine the frames into a single NumPy array
            audio_data = b''.join(list(audio_frames))
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # --- Transcription with ASR model ---
            try:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                end_of_window = len(audio_array)
                start_of_window = max(0, end_of_window - RATE * ASR_WINDOW_SECONDS)
                
                segment_to_transcribe = audio_array[start_of_window:end_of_window]
                
                # Save array chunk to a temporary audio file so mlx-audio can process it
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    temp_filename = tmp_file.name
                
                sf.write(temp_filename, segment_to_transcribe, RATE)

                # Transcribe
                if use_mlx_vlm:
                    prompt = f"Transcribe the following {SOURCE_LANGUAGE} speech exactly as spoken:"
                    formatted_prompt = apply_chat_template(
                        asr_processor, asr_model.config, prompt, num_audios=1
                    )
                    result = generate(asr_model, asr_processor, prompt, audio=[temp_filename], max_tokens=100, verbose=False, temperature=0.0)
                else:
                    result = asr_model.generate(temp_filename, language=SOURCE_LANGUAGE)
                
                # Parse output format reliably
                transcribed_text = result.text.strip() if hasattr(result, 'text') else str(result).strip()

                # Cleanup temporary fragment
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
                if transcribed_text and transcribed_text != last_transcribed_text:
                    output_text = f"[{current_time}] ASR: {transcribed_text}"
                    print(colored(output_text, "light_red", "on_light_grey"))
                    if log_file:
                        log_file.write(output_text + "\n")
                        log_file.flush()
                    last_transcribed_text = transcribed_text

                    # --- Translation with LM Studio ---
                    if not use_mlx_vlm:
                        try:
                            headers = {"Content-Type": "application/json"}
                            payload = {
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": f"You are a helpful assistant that translates {SOURCE_LANGUAGE} text to English."
                                    },
                                    {
                                        "role": "user",
                                        "content": f"Translate the following text to English. Do not provide any explanation or any other text: {transcribed_text}"
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
                            print(colored(english_output, "cyan", attrs=["bold"]))
                            if log_file:
                                log_file.write(english_output + "\n")
                                log_file.flush()
                        except requests.exceptions.RequestException as e:
                            print(f"Error communicating with LM Studio: {e}")
                        except (KeyError, IndexError) as e:
                            print(f"Error parsing LM Studio response: {e}")
            
            except Exception as e:
                print(f"Error during ASR transcription: {e}")
            
            last_processed_position += RATE * PROCESSING_HOP_SECONDS
            
        try:
            new_data = audio_queue.get_nowait()
            audio_frames.append(new_data)
        except queue.Empty:
            time.sleep(0.1)

if __name__ == "__main__":
    # Ensure the 'soundfile' library is installed to support rapid buffer translation
    try:
        import soundfile
    except ImportError:
        print("The 'soundfile' library is required. Please install it using:")
        print("pip install soundfile")
        exit()

    p = pyaudio.PyAudio()
    
    device_index = None
    input_device_name = "Default Microphone"
    capture_loopback_device = False

    if "--loopback" in sys.argv:        
        print("Attempting to use the default loopback audio device.")
        input_device_name = "Default Loopback"
        
        try:
            found_loopback = False
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                print(dev_info["name"])
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

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_index,
                    stream_callback=audio_callback)
    
    transcripts_dir = "transcripts"
    if not os.path.exists(transcripts_dir):
        os.makedirs(transcripts_dir)
        print(f"Created directory: {transcripts_dir}")

    filename_prefix = "Transcript-ASR"
    if len(sys.argv) > 1:
        skip_next = False
        for i in range(1, len(sys.argv)):
            if skip_next:
                skip_next = False
                continue
            cmd_arg = sys.argv[i]
            if cmd_arg == "--model":
                skip_next = True
                continue
            if cmd_arg not in ("--loopback", "--voxtral"):
                filename_prefix = cmd_arg
                break      

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = os.path.join(transcripts_dir, f"{filename_prefix}-{timestamp}.txt")
    try:
        log_file = open(log_filename, "a", encoding="utf-8")
        print(f"Logging output to {log_filename}")
    except IOError as e:
        print(f"Error opening log file: {e}")
        log_file = None
    
    processor_thread = threading.Thread(target=process_audio)
    processor_thread.daemon = True
    processor_thread.start()

    print("--- Starting audio capture. Press Ctrl+C to stop. ---")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n--- Stopping audio capture. Please wait for the final processing to complete. ---")
    finally:
        is_recording = False
        processor_thread.join()
        
        stream.stop_stream()
        stream.close()
        p.terminate()

        if log_file:
            log_file.close()
            print(f"Log file '{log_filename}' closed.")
        
    print("Application closed.")
