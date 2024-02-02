from flask import Flask, request, jsonify
import whisper
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

@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    try:
        audio_file = request.files['audio']
        transcription = SpeechToText(audio_file)
        return jsonify({'transcription': transcription})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # This block is for running with the built-in Flask server
    # Use Gunicorn for production deployments
    app.run(debug=True, port=8000)

