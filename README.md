# STT-realtime
Python script to transcribe audio from mic or loopback utilizing whisper and LM Studio. The models run entirely locally (the system may try to download the Whisper model files for an initial run).
The default settings are configured to translate Japanese audio to English.

This script has been tested on Windows, macOS and Linux (Fedora 42) using whisper-large-v3, whisper-large-v3-turbo, base, medium for transcription and then utilizing the gemma-3-4b-qat LLM for translation to English. For Windows and Linux, NVIDIA GPUs have been tested although it should be possible to run this on AMD or Intel GPUs with minimal changes. Playing about with the parameters and different models is recommended, depending on your machine configuration. Naturally, higher GPU VRAM (at least 8GB) or total RAM (in the case of macOS; >= 24GB recommended) would lead to better results.

==Recommendations==
1. For capturing loopback audio on macOS, you will need to install the Blackhole 2ch virtual audio driver (via homebrew or manually). 
2. Use of a venv is recommended
3. It is also recommended to install the latest torch nightly build if you are using macOS, so that MPS-based GPU acceleration can be utilized
