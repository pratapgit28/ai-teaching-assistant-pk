import whisper
import numpy as np

def transcribe_audio(audio_input):
    try:
        if isinstance(audio_input, np.ndarray):
            audio_array = audio_input
        elif isinstance(audio_input, str):
            audio_array, _ = librosa.load(audio_input, sr=16000)
        else:
            raise ValueError("Unsupported audio input type")
        
        if len(audio_array) == 0:
            raise ValueError("Audio input is empty")
        
        model = whisper.load_model("base")
        result = model.transcribe(audio_array)
        return result["text"]
    except Exception as e:
        raise ValueError(f"Error in transcribing audio: {str(e)}")

# Implement Deepgram as an alternative
from deepgram import Deepgram

def transcribe_audio_deepgram(audio_file, api_key):
    dg = Deepgram(api_key)
    with open(audio_file, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': 'audio/wav'}
        response = dg.transcription.sync_prerecorded(source, {'punctuate': True})
    return response['results']['channels'][0]['alternatives'][0]['transcript']

# Create a function to generate structured notes using an LLM

import openai
import os

# Set your OpenAI API key here
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

def generate_notes(transcript, lesson_plan):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use a supported model
            messages=[
                {"role": "user", "content": f"Generate structured notes from this transcript:\n{transcript}\n\nLesson plan:\n{lesson_plan}"}
            ],
            max_tokens=500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error in generate_notes: {str(e)}")
        return "Failed to generate notes"

# Implement timestamp linking
def link_timestamps(notes, transcript):
    # Implement logic to link key topics with timestamps
    # This will require natural language processing to match topics with transcript segments
    pass

# Create a function to generate quiz questions

def generate_quiz(transcript):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use a supported model
            messages=[
                {"role": "user", "content": f"Generate 5 multiple-choice questions based on this lecture transcript:\n{transcript}"}
            ],
            max_tokens=500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error in generate_quiz: {str(e)}")
        return "Failed to generate quiz"

# Implement YouTube video download and audio extraction

import youtube_dl
from pydub import AudioSegment

import numpy as np
import librosa

def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': 'temp_audio.%(ext)s'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Load the downloaded audio file into a numpy array
    audio_array, sample_rate = librosa.load('temp_audio.wav', sr=16000)
    return audio_array
    # The audio file will be saved in the current directory
    # You'll need to implement logic to get the filenamec

# Develop the main application flow

def main():
    # Get input (audio file or YouTube URL)
    input_source = input("Enter audio file path or YouTube URL: ")
    
    # If YouTube URL, download audio
    if "youtube.com" in input_source or "youtu.be" in input_source:
        input_source = download_youtube_audio(input_source)
    
    # Transcribe audio
    transcript = transcribe_audio(input_source)
    
    # Generate notes
    lesson_plan = input("Enter lesson plan: ")
    notes = generate_notes(transcript, lesson_plan)
    notes_with_timestamps = link_timestamps(notes, transcript)
    
    # Generate quiz
    quiz = generate_quiz(transcript)
    
    # Output results
    print("Lecture Notes:")
    print(notes_with_timestamps)
    print("\nQuiz Questions:")
    print(quiz)

# Add this new function before the Gradio interface
import time
import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf

def process_input(audio_file, youtube_url, lesson_plan):
    try:
        print("Starting process_input function")
        start_time = time.time()
        
        # Debug print
        print(f"Audio file input type: {type(audio_file)}, content: {audio_file}")
        
        
        # Determine input source
        if audio_file is not None:
            print(f"Audio file processed. Time taken: {time.time() - start_time:.2f} seconds")
            
            # Handle Gradio audio input (which comes as a tuple)
            if isinstance(audio_file, tuple):
                audio_path = audio_file[0]  # First element is the file path
            else:
                audio_path = audio_file
            
            # Convert MP3 to WAV if necessary
            if audio_path.lower().endswith('.mp3'):
                # Use pydub to convert MP3 to WAV
                audio = AudioSegment.from_mp3(audio_path)
                wav_path = audio_path.replace('.mp3', '.wav')
                audio.export(wav_path, format="wav")
                audio_path = wav_path
            
            # Load audio using librosa
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Debug print
            print(f"Audio array shape: {audio_array.shape}, type: {type(audio_array)}")
        
        elif youtube_url:
            audio_array = download_youtube_audio(youtube_url)
            print(f"YouTube audio array shape: {audio_array.shape}, type: {type(audio_array)}")
        else:
            return "Please provide either an audio file or YouTube URL"
        
        # Check if audio_array is empty
        if len(audio_array) == 0:
            return "The audio file is empty or could not be processed."
        
        # Transcribe audio
        print("Starting transcription")
        
        transcript = transcribe_audio(audio_array)
        
        print(f"Transcription completed. Time taken: {time.time() - start_time:.2f} seconds")
        
        # Generate notes
        print("Generating notes")
        
        notes = generate_notes(transcript, lesson_plan)

        print(f"Notes generated. Time taken: {time.time() - start_time:.2f} seconds")

        print("Linking timestamps")
        
        notes_with_timestamps = link_timestamps(notes, transcript)

        print(f"Timestamps linked. Time taken: {time.time() - start_time:.2f} seconds")

        
        
        # Generate quiz
        print("Generating quiz")
        
        quiz = generate_quiz(transcript)

        print(f"Quiz generated. Time taken: {time.time() - start_time:.2f} seconds")
        
        # Format output
        output = f"""
        Lecture Notes:
        {notes_with_timestamps}
        
        Quiz Questions:
        {quiz}
        """
        return output
    
    except Exception as e:
        # More detailed error logging
        print(f"Full error details: {str(e)}")
        return f"An error occurred: {str(e)}"

# Create Gradio interface
import gradio as gr

iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Textbox(label="Or Enter YouTube URL"),
        gr.Textbox(label="Enter Lesson Plan", lines=3)
    ],
    outputs=gr.Textbox(label="Generated Content", lines=10),
    title="AI Teaching Assistant",
    description="Upload an audio file or provide a YouTube URL to generate lecture notes and quiz questions.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    iface.launch()
