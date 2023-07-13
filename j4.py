import pyaudio
import wave
import io
import numpy as np
import threading
import queue
from transformers import pipeline

# Parameters
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
BUFFER_SIZE = 5

# Global variable to record if the system should record or not
record_command = False


# Function to transcribe audio
def transcribe_audio(q):
    global record_command
    asr_pipeline = pipeline(
        "automatic-speech-recognition", model="facebook/wav2vec2-base-960h"
    )
    buffer = []
    while True:
        audio_np_array = q.get()  # Get next item from the queue
        # print(f" Current working item: {audio_np_array}")
        buffer.append(audio_np_array)
        print(f"Buffer: {len(buffer)}")
        if len(buffer) >= BUFFER_SIZE:
            print(f"Buffer Loaded {buffer}")
            foo = np.concatenate(buffer)
            transcription = asr_pipeline(foo)
            print(transcription)
            if "Big cock" in transcription:
                print("* Initiating Boot")
                record_command = True
            elif record_command:
                print(f"Command received: {transcription}")
                record_command = False
            print("Buffer Reset")
            buffer = []


# Start transcription thread
q = queue.Queue()
threading.Thread(target=transcribe_audio, args=(q,)).start()

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

# Main loop
print("* Active listening started.")
frames = []
with io.BytesIO() as buffer:
    while True:
        buffer.write(stream.read(CHUNK))
        audio_float32 = (np.frombuffer(buffer.getvalue(), dtype=np.int16)) / np.iinfo(
            np.int16
        ).max
        q.put(audio_float32)
        buffer.seek(0)
        buffer.truncate()

        if record_command:
            print("* Activation phrase detected. Starting to record.")
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("* Done recording.")
            record_command = False
            # Save recording
            wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
            wf.close()
            frames = []

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
