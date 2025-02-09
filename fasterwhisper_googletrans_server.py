#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This sample code is the server model for the PVT Leaf client.
# It is specifically created to target a server with CUDA on Windows. 
# Multi-client and HTTP/HTTPS are supported. Some error messages are output as is.
# Please use this server model in accordance with each LICENSE.
# Copyright(C) Omicronware.
import os
import sys
import tempfile
import traceback
import asyncio
import ctypes

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel, BatchedInferencePipeline

# If using the asynchronous version of googletrans
# pip install git+https://github.com/ssut/py-googletrans.git@master
from googletrans import Translator

# WSGI server using gevent
# pip install gevent
from gevent.pywsgi import WSGIServer
import gevent
from datetime import datetime

app = Flask(__name__)

def is_cuda_available():
    """Sample implementation to check CUDA availability based on Windows + nvcuda.dll"""
    if sys.platform != "win32":
        return False
    try:
        return bool(ctypes.windll.kernel32.LoadLibraryW("nvcuda.dll"))
    except OSError:
        return False

MODEL_NAME = "large-v3-turbo" if is_cuda_available() else "tiny"
DEVICE = "cuda" if is_cuda_available() else "cpu"
COMPUTE_TYPE = "auto"  # Example: setting to "float16" can reduce GPU memory usage

try:
    print(f"Loading faster-whisper model '{MODEL_NAME}' on {DEVICE} ({COMPUTE_TYPE}) ...")
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    batched_model = BatchedInferencePipeline(model=model)
    print("Batched Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e, file=sys.stderr)
    sys.exit(1)

# Asynchronous translation function
async def Translate_Google(lang_from, lang_to, text):
    async with Translator() as translator:
        translated = await translator.translate(text, src=lang_from, dest=lang_to)
    return translated.text


@app.errorhandler(Exception)
def handle_exception(e):
    """Catch all unhandled exceptions in Flask"""
    tb = traceback.format_exc()
    return jsonify({
        "error": "An internal server error has occurred.",
        "details": str(e),
        "trace": tb
    }), 500

@app.route('/transcribe', methods=['GET', 'POST'])
def transcribe():
    """
    GET:
      - Returns { "status": "ok" } for health check (works with both HTTP/HTTPS)

    POST:
      - Form data:
        - audio_file (required): Audio file such as mp3
        - from_language (optional): Language for Faster-Whisper transcription (auto-detect if not specified)
        - to_language (optional): Target language for translation (e.g., 'en', 'ja')
      - Returns transcription and translation results in JSON format
    """
    if request.method == 'GET':
        return jsonify({"status": "ok"}), 200

    if 'audio_file' not in request.files:
        return jsonify({"error": "audio_file is not included in the request"}), 400

    audio_file = request.files['audio_file']
    from_language = request.form.get("from_language", None)
    to_language   = request.form.get("to_language", None)

    # Save audio file to a temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_file.save(tmp)
            tmp_filename = tmp.name
    except Exception as e:
        return jsonify({
            "error": "Failed to temporarily save the audio file",
            "details": str(e)
        }), 500

    try:
        # Transcription
        segments, info = batched_model.transcribe(tmp_filename, language=from_language)
        full_text = "".join(segment.text for segment in segments)
        segments_list = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ]

        detected_language = getattr(info, "language", from_language)

        # Perform translation if required
        translated_text = None
        if to_language:
            from_lang_google = detected_language if detected_language else "auto"
            translated_text = asyncio.run(
                Translate_Google(from_lang_google, to_language, full_text)
            )
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({
            "error": "Failed to process transcription",
            "details": str(e),
            "trace": tb
        }), 500
    finally:
        # Delete temporary file
        try:
            os.remove(tmp_filename)
        except Exception:
            pass

    # Return JSON
    result = {
        "transcript_text": full_text,
        "translated_text": translated_text,
        "segments": segments_list,
        "language": detected_language
    }
    return jsonify(result), 200


def custom_exception_handler(exc_type, exc_value, exc_traceback):
    # Append error details to a log file with timestamp
    
    ErrorLogFile = './error.log'
    with open(ErrorLogFile, "a") as log_file:
        head = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Uncaught exception:\n"
        log_file.write("\n")
        log_file.write(head)
        print("\n"+head)
        
        for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
                log_file.write(line)
                print(line)
    
    # Terminate the script
    sys.exit(1)  # Return error exit code 1


# Override default exception handler
sys.excepthook = custom_exception_handler

if __name__ == '__main__':
    # Certificate files (self-signed or official certificates, etc.)
    SSL_CERT_FILE = "server.crt"
    SSL_KEY_FILE = "server.key"

    # HTTP WSGIServer (0.0.0.0:9000)
    http_server = WSGIServer(('0.0.0.0', 9000), app)

    # HTTPS WSGIServer (0.0.0.0:9443)
    # Specify certfile/keyfile
    try:
      https_server = WSGIServer(
          ('0.0.0.0', 9443), 
          app,
          keyfile=SSL_KEY_FILE,
          certfile=SSL_CERT_FILE
      )

      print("Starting HTTP server on port 9000")
      http_server.start()

      print("Starting HTTPS server on port 9443")
      https_server.start()

      print("Servers are running. Press Ctrl+C to stop.")
      
    except ssl.SSLError as e:
        print(f"SSL Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred while starting the server: {e}", file=sys.stderr)
        traceback.print_exc()


    try:
        # Run concurrently with gevent (modified to stop with Ctrl+C)
        gevent.wait()
    except KeyboardInterrupt:
        print("\nShutdown requested. Stopping servers...")
        http_server.stop()
        https_server.stop()
        print("Servers stopped successfully.")
        sys.exit(0)
