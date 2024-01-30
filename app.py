from flask import Flask
import whisper
import gradio as gr
import time
import torch

app = Flask(__name__)

model = whisper.load_model("base")

def SpeechToText(audio):
    time.sleep(1)

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    return result.text

# def transcribe_audio(file_path):
#     #  time.sleep(1)
#     # Read the entire audio file
#     full_audio = whisper.load_audio(file_path)

#     # Preprocess the full audio
#     full_audio = whisper.pad_or_trim(full_audio)
#     mel = whisper.log_mel_spectrogram(full_audio).to(model.device)

#     # Decode the full audio
#     options = whisper.DecodingOptions(fp16=False)
#     result = whisper.decode(model, mel, options)

#     return result.text

try:
    print("Starting the Gradio Web UI")
    gr.Interface(
        title='Speech Transcription',
        fn=SpeechToText,
        inputs=[
            gr.Audio(type="filepath", label="Upload Audio File")
        ],
        outputs=[
            "textbox",
        ]
    ).launch(
        debug=False,
        share=False
    )
except Exception as e:
    print(f"An error occurred: {str(e)}")
