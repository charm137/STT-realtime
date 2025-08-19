# STT-realtime
Python script to transcribe streaming audio from a mic or loopback source with a very reasonable latency (3-sec lag) utilizing OpenAI's [whisper](https://github.com/openai/whisper) and [LM Studio as a local server](https://lmstudio.ai/docs/app/api/headless). The models run entirely locally (the system may try to download the Whisper model files for an initial run).
The default settings are configured to translate Japanese audio to English taking audio from the specified source (default: microphone) with a 5-sec rolling context window for Whisper (capturing most spoken sentences) and a 3-sec step-size for the context, so that overlap with the previous sentence is maintained while ensuring minimal practical latency. Both paramters are adjustable through the `WHISPER_WINDOW_SECONDS` and `PROCESSING_HOP_SECONDS` variables respectively, located at the beginning of the script.

This script has been tested on Windows, macOS and Linux (Fedora 42) using whisper-large-v3, whisper-large-v3-turbo, base, medium for transcription and then utilizing the [gemma-3-4b-qat](https://huggingface.co/collections/google/gemma-3-qat-67ee61ccacbf2be4195c265b) LLM model for translation to English. For Windows and Linux, NVIDIA GPUs have been tested although it should be possible to run this on AMD or Intel GPUs with minimal changes. Playing about with the parameters and different models is recommended, depending on your machine configuration. Naturally, higher GPU VRAM (at least 8GB) or total RAM (in the case of macOS; >= 24GB recommended) would lead to better real-time performance.

## Dependencies
You will need to install torch (and CUDA, if you want to use NVIDIA GPU-acceleration), mlx-whisper (for macOS), pyaudio whisper and numpy. Use of uv is recommended instead of utilizing pip directly. Python 3.12+ should work.

Overall the following should work, in most instances (except CUDA, which you will have to install manually, if you plan to use NVIDIA GPUs):  
`uv venv`  

(Linux/macOS): 
`source .venv\bin\activate`  

(Windows), in cmd.exe  
`venv\Scripts\activate.bat`  

(Windows) or, in PowerShell  
`venv\Scripts\Activate.ps1`  

`uv pip install torch`  
`uv pip install numpy`  
`uv pip install mlx-whisper` (this may yield an error if you are not on macOS, you can ignore this, if so)  
`uv pip install pyaudio`  


It is recommended you install whisper using the latest git commit (assuming you are using uv):

`uv pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git`  


## Recommendations
1. For capturing loopback audio on macOS, you will need to install the [Blackhole 2ch](https://github.com/ExistentialAudio/BlackHole) virtual audio driver (via homebrew or manually). 
2. Use a venv to ensure no version/dependency conflicts with system libraries
3. Install the latest torch nightly build if you are using macOS, so that MPS-based GPU acceleration can be utilized
4. The context window for the LLM hosted on LM Studio can be pretty short, to help with resource usage: 512 tokens or so should suffice

## Usage
Run:
`python mic_stream.py`

Make sure your LM Studio server is running and has loaded up an LLM model.

You will be able to see the transcribed text in the console. By default, a transcript file would be created within a "transcripts" directory within the current working directory.

Add a `<filename_prefix>` to the command-line arguments to change the transcript filename prefix.

Add a `--loopback` command line argument to capture audio from a playback device. For Windows, you will need to change the default input device to "Stereo Mix" or something similar to capture the audio. See requirements for macOS above. For Linux, the script latches onto the "sysdefault" ALSA audio sink but this may or may not be correct for your system - modify the script to suit your needs.


