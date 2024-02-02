import streamlit as st
import requests

def streamlit_ui():
    st.title('Speech Transcription')
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Send audio file to Flask endpoint
        files = {'audio': uploaded_file}
        response = requests.post('http://localhost:8000/transcribe', files=files)

        if response.status_code == 200:
            transcription = response.json()['transcription']
            st.text("Transcription:")
            st.text(transcription)
        else:
            st.text("Error during transcription")

if __name__ == '__main__':
    streamlit_ui()
