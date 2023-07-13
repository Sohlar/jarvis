from transformers import pipeline

gen = pipeline(task="automatic-speech-recognition")
gen("output.wav")
