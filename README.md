# AI Teaching Assistant

This application leverages advanced AI models to transcribe audio files, generate structured lecture notes, and create quiz questions based on the provided content. It supports audio input from both local files and YouTube URLs.

## Features

- **Audio Transcription**: Convert audio files into text using the Whisper model.
- **Note Generation**: Generate structured notes from transcripts using OpenAI's GPT-3.5 Turbo model.
- **Quiz Creation**: Automatically create multiple-choice questions based on lecture transcripts.
- **YouTube Support**: Download and transcribe audio directly from YouTube videos.

## Requirements

- Python 3.7 or higher
- Libraries:
  - `whisper`
  - `numpy`
  - `openai`
  - `librosa`
  - `pydub`
  - `youtube_dl`
  - `gradio`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-teaching-assistant.git
   cd ai-teaching-assistant

2. Install the required packages:
   pip install -r requirements.txt

3. Set your OpenAI API key on HuggigFace
   OPENAI_API_KEY="your_api_key_here"

## Usage

To run the application, execute the following command:

python app.py

## Input Options

Audio File: Upload a local audio file (WAV or MP3 format).
YouTube URL: Provide a link to a YouTube video containing the lecture audio.
Lesson Plan: Enter a brief lesson plan to guide note generation.

## Acknowledgments

OpenAI for providing powerful language models.
Whisper for robust audio transcription capabilities.
Gradio for creating user-friendly interfaces.
