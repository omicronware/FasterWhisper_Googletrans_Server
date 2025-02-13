# Faster-Whisper GoogleTrans Server

This repository contains a transcription and translation server based on **faster-whisper** and **googletrans**. It is designed for use with **PVT Leaf clients** and runs on **CUDA-enabled Windows servers**. The server supports **multi-client connections** over **HTTP/HTTPS** and provides real-time transcription and translation capabilities.

## Features

- **Fast and accurate speech recognition** using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Automatic language detection and translation** using [googletrans](https://github.com/ssut/py-googletrans)
- **Supports multi-client requests** with **Flask** and **gevent** for concurrency
- **CUDA acceleration** for enhanced performance on Windows [NVIDIA Driver](https://www.nvidia.com/drivers/) [CUDA12](https://developer.nvidia.com/cuda-toolkit) [cudnn9.6](https://developer.nvidia.com/cudnn)
- **Transcription and translation APIs** accessible via HTTP/HTTPS

## Installation

Ensure you have Python 3.8 or later installed. Then, install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

Run the server with the following command:

```sh
python fasterwhisper_googletrans_server.py
```

The server will start with:

- **HTTP** on port `9000`
- **HTTPS** on port `9443` (if SSL certificates are provided)

### API Endpoints

#### Health Check (GET)

```http
GET /transcribe
```

Response:

```json
{"status": "ok"}
```

#### Transcription and Translation (POST)

```http
POST /transcribe
```

**Form Data:**

- `audio_file` (required): The audio file (e.g., MP3)
- `from_language` (optional): Language for transcription (auto-detect if not specified)
- `to_language` (optional): Target translation language (e.g., `en`, `ja`)

**Response Example:**

```json
{
  "transcript_text": "Hello, how are you?",
  "translated_text": "こんにちは、元気ですか？",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "Hello, how are you?"}
  ],
  "language": "en"
}
```

## Server Details (For Advanced Users)

This server is designed to be used as a **PVT Leaf Server** for Omicronware applications, running on a Windows-based CUDA-enabled server. The implementation ensures **low-latency**, **high-accuracy speech recognition**, and **scalable multi-client support**.

More details can be found in the **PVT Leaf documentation**: [Omicronware PVT Leaf Usage](https://www.omicronware.com/home/pvtleaf-usage/)

## License

This project follows the licensing agreements of the included libraries. Refer to each library's repository for specific details:

- Faster-Whisper: [GitHub](https://github.com/SYSTRAN/faster-whisper)
- GoogleTrans: [GitHub](https://github.com/ssut/py-googletrans)

## Author

Copyright (C) Omicronware.

