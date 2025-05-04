import ssl
import json
import os
import tempfile
import whisper
import numpy as np
import speech_recognition as sr
import pyttsx3
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from gtts import gTTS
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Disable SSL verification (for downloading Whisper model)
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize FastAPI app
app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")

class TTSRequest(BaseModel):
    text: str

# Define the request body model
class InterviewRequest(BaseModel):
    questions: List[str]

# Function to conduct the interview
def conduct_interview(questions):
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)  # Adjust speech rate
    
    # Set a modern, natural-sounding voice
    voice_id = "com.apple.eloquence.en-US.Reed"  # Choose from available voices
    engine.setProperty('voice', voice_id)
    
    answers = {}
    
    with sr.Microphone() as source:
        print("\nCalibrating for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        
        for question in questions:
            print(f"\nQuestion: {question}")
            engine.say(question)
            engine.runAndWait()
            
            print("Listening for your answer...")
            try:
                audio = recognizer.listen(source, timeout=10)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    temp_audio.write(audio.get_wav_data())
                    temp_audio_path = temp_audio.name
                
                try:
                    result = model.transcribe(temp_audio_path)
                    answer = result["text"].strip()
                    print(f"You said: {answer}")
                    answers[question] = answer
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    answers[question] = "Error in processing"
                
                os.unlink(temp_audio_path)
                
            except sr.WaitTimeoutError:
                print("No speech detected within the timeout period")
                answers[question] = "No answer provided"
    
    return answers
def listen_and_transcribe():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("\nCalibrating for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        
        print("Listening... (speak now)")
        try:
            audio = recognizer.listen(source)  # Removed timeout for natural speech

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio.get_wav_data())
                temp_audio_path = temp_audio.name

            try:
                result = model.transcribe(temp_audio_path)
                transcription = result["text"].strip()
                print(f"Transcription: {transcription}")
                os.unlink(temp_audio_path)  # Clean up temp file
                return transcription
            except Exception as e:
                print(f"Error during transcription: {e}")
                os.unlink(temp_audio_path)
                return "Error during transcription"
                
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected.")
            return "No speech detected"

# Define the API endpoints
@app.post("/interview/")
async def start_interview(request: InterviewRequest):
    print("Interview started...")
    answers = conduct_interview(request.questions)
    return {"answers": answers}

@app.post("/generate-tts/")
async def generate_tts(request: TTSRequest):
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)  # Adjust speech speed
    
    # Set voice (optional)
    voice_id = "com.apple.speech.synthesis.voice.reed"  # You can change this to match your preferred voice
    engine.setProperty('voice', voice_id)
    
    engine.say(request.text)
    engine.runAndWait()
    
    return {"message": "Speech played"}

# Update the API endpoint to use the listening behavior
@app.post("/generate-stt/")
async def generate_stt():
    transcription = listen_and_transcribe()
    return {"transcription": transcription}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)