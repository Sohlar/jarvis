from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from collections import deque
import pyaudio
import torch
import queue
import threading
import numpy as np
import soundfile as sf
import time

# Parameters
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLERATE = 16000
RECORD_SECONDS = 5
BUFFER_SIZE = 5
MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"  # pretrained 1.26GB

ACTIVATION_KEYWORD = "Jarvis"
TERMINATION_KEYWORD = "terminate program"
ACTIVATION_THRESHOLD = 0.7  # Confidence level required to activate recording
KEYWORD_MODEL_NAME = "facebook/wav2vec2-base-960h"  # Faster mode


class AudioTranscriber:
    def __init__(self):
        self.q = queue.Queue()
        self.transcription_q = queue.Queue()
        self.transcribe_event = threading.Event()

        # Double Buffering Initialization
        self.current_deque = deque(maxlen=int(SAMPLERATE / CHUNK * RECORD_SECONDS))
        self.processing_deque = deque(maxlen=int(SAMPLERATE / CHUNK * RECORD_SECONDS))

        # Initialize model, tokenizer, and feature extractor
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

        self.t = threading.Thread(target=self.transcribe_audio)
        self.t.start()

        # Additional initialization for keyword spotting
        self.keyword_model = Wav2Vec2ForCTC.from_pretrained(KEYWORD_MODEL_NAME).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.keyword_tokenizer = Wav2Vec2Processor.from_pretrained(KEYWORD_MODEL_NAME)

        # Initialize PyAudio and stream within the class
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLERATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

    def detect_phrase(self, audio_data, model, tokenizer, phrase):
        input_values = tokenizer(
            audio_data, return_tensors="pt", sampling_rate=SAMPLERATE
        ).input_values.float()

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.decode(predicted_ids[0])

        # If keyword detected, return True
        return phrase.lower() in transcription.lower()

    def transcribe_audio(self):
        while True:
            audio_frames = self.q.get()
            if audio_frames == "STOP":
                break

            # Combine frames
            audio_data = np.frombuffer(b"".join(audio_frames), dtype=np.int16)

            # Guard against empty or inappropriate-sized audio data
            if audio_data.size == 0 or audio_data.size < BUFFER_SIZE:
                self.transcription_q.put("Invalid audio chunk.")
                continue

            input_values = self.tokenizer(
                audio_data, return_tensors="pt", sampling_rate=SAMPLERATE
            ).input_values.float()
            with torch.no_grad():
                logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.tokenizer.decode(predicted_ids[0])
            self.transcription_q.put(transcription)

    def record_and_transcribe(self, stream):
        print("Listening for keywords...")
        while True:
            # Continuously append audio data to current_deque
            data = stream.read(CHUNK, exception_on_overflow=False)
            self.current_deque.append(data)

            # When the current_deque is full, check for the activation keyword
            if len(self.current_deque) == int(SAMPLERATE / CHUNK * RECORD_SECONDS):
                audio_data = np.frombuffer(
                    b"".join(list(self.current_deque)), dtype=np.int16
                )

                # Check for the termination keyword
                if self.detect_phrase(
                    audio_data,
                    self.keyword_model,
                    self.keyword_tokenizer,
                    TERMINATION_KEYWORD,
                ):
                    print("Termination keyword detected!")
                    print("Terminating the program.")
                    return

                # Check for the activation keyword
                if self.detect_phrase(
                    audio_data,
                    self.keyword_model,
                    self.keyword_tokenizer,
                    ACTIVATION_KEYWORD,
                ):
                    print("Activation keyword detected! Recording next chunk...")

                    # Clear the current_deque for the new recording
                    self.current_deque.clear()

                    # Record for the specified duration
                    for _ in range(0, int(SAMPLERATE / CHUNK * RECORD_SECONDS)):
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        self.current_deque.append(data)

                    print("Recording ended.")

                    # Switch the buffers for transcription
                    self.processing_deque, self.current_deque = (
                        self.current_deque,
                        self.processing_deque,
                    )

                    audio_frames = list(self.processing_deque)
                    self.q.put(audio_frames)

                    # Transcribe the audio
                    recorded_audio = b"".join(list(self.current_deque))
                    self.q.put(list(self.current_deque))
                    transcription = self.transcription_q.get()
                    print(f"Transcribed: {transcription}")

                    # Save the recorded chunk to a file
                    filename = f"recorded_chunk_{int(time.time())}.wav"
                    sf.write(
                        filename,
                        np.frombuffer(recorded_audio, dtype=np.int16),
                        SAMPLERATE,
                    )
                    print(f"Saved the recorded chunk to '{filename}'.")
                    print("Listening for keywords...")

                # Clear the current_deque for fresh keyword listening
                self.current_deque.clear()

    def start(self):
        self.record_and_transcribe(self.stream)

    def terminate(self):
        # Close and terminate the audio stream
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        # Terminate the transcription thread
        self.q.put("STOP")
        self.t.join()


# Usage
transcriber = AudioTranscriber()

# MAIN
try:
    transcriber.start()
finally:
    # This ensures that even if there's an error or interruption,
    # the terminate method will be called to clean up resources
    transcriber.terminate()
