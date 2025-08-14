# STT-realtime
Python script to transcribe audio from mic or loopback utilizing whisper and LM Studio. The models run entirely locally (the system may try to download the Whisper model files for an initial run).
The default settings are configured to translate Japanese audio to English.

This script has been tested on Windows, macOS and Linux (Fedora 42) using whisper-large-v3, whisper-large-v3-turbo, base, medium for transcription and then utilizing the gemma-3-4b-qat LLM for translation to English. For Windows and Linux, NVIDIA GPUs have been tested although it should be possible to run this on AMD or Intel GPUs with minimal changes. Playing about with the parameters and different models is recommended, depending on your machine configuration. Naturally, higher GPU VRAM (at least 8GB) or total RAM (in the case of macOS; >= 24GB recommended) would lead to better results.

## Dependencies
You need to install torch (and CUDA, if you want to use NVIDIA GPU-accelration), mlx-whisper, whisper and numpy. Use of uv is recommended instead of utilizing pip directly. Python 3.12+ should work.


## Recommendations
1. For capturing loopback audio on macOS, you will need to install the Blackhole 2ch virtual audio driver (via homebrew or manually). 
2. Use of a venv is recommended
3. It is also recommended to install the latest torch nightly build if you are using macOS, so that MPS-based GPU acceleration can be utilized

## Usage
Run:
`python mic_stream.py`

You will be able to see the transcribed text in the console. By default, a transcript file would be created within a "transcripts" directory within the current working directory.

Add a `<filename_prefix>` to the command-line arguments to change the transcript filename prefix.

Add a `--loopback` command line argument to capture audio from a playback device. For Windows, you will need to change the default input device to "Stereo Mix" or something similar to capture the audio. See requirements for macOS above. The script latches onto the "sysdefault" sudio sink but this may or may not be correct for your system - modify the script to suit your needs.


