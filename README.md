# PyAudio Speech-to-Text and Activation Phrase Detection

This project uses the power of the `transformers` library in combination with PyAudio to transcribe audio from your microphone and perform certain actions based on the detected activation phrase.

## Overview

The main function of the code is to listen for audio from the system's microphone and transcribe it into text. When a specific activation phrase is detected, the code starts recording audio for a predefined amount of time and saves it as a WAV file.

## Requirements

- Python 3.7 or higher
- `pyaudio`
- `numpy`
- `threading`
- `queue`
- `wave`
- `transformers` library (HuggingFace)

You can install the necessary dependencies with pip:

```bash
pip install pyaudio numpy wave transformers
```

## Usage
To run the script, simply clone the repository and run the script using Python:

```bash
git clone <repository_link>
cd <repository_name>
python main.py
```
Replace <repository_link> and <repository_name> with the appropriate details.

## Customization
Activation Phrase: You can customize the activation phrase by changing the string in the condition if "Big cock" in transcription:. Replace "Big cock" with your own activation phrase.
Audio Recording Duration: You can change the recording duration by modifying the RECORD_SECONDS constant.
Buffer Size: The buffer size can be adjusted by changing the BUFFER_SIZE constant.
ASR Model: You can change the ASR (Automatic Speech Recognition) model used by the transformers pipeline by modifying the string in model="facebook/wav2vec2-base-960h". Replace it with the model of your choice from the Hugging Face Model Hub.
Limitations
As with any ASR model, the transcription accuracy can vary depending on the quality of the audio input and the accent of the speaker. Be aware that false positives or negatives can occur when detecting the activation phrase.

## License
This project is licensed under the terms of the MIT license.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
