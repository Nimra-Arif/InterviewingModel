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

# Disable SSL verification (for downloading Whisper model)
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize FastAPI app
app = FastAPI()

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")

# Define the request body model
class InterviewRequest(BaseModel):
    questions: List[str]

# Function to conduct the interview
def conduct_interview(questions):
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  

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

# Define the API endpoint
@app.post("/interview/")
async def start_interview(request: InterviewRequest):
    print("Interview started...")
    answers = conduct_interview(request.questions)
    return {"answers": answers}
