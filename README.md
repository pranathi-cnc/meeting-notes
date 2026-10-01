# Meeting Notes Generator

A simple local AI application that converts meeting transcripts or audio recordings into structured meeting notes.

## Features

- Generate meeting summary using Gemma 4
- Extract explicitly stated decisions
- Extract action items:
  - Task
  - Owner
  - Due date
- Audio transcription using faster-whisper
- Pydantic validation for structured output
- Human review and correction using Gradio
- Save notes as JSON and TXT files
- Runs locally using Ollama

## Workflow

```text
Audio / Transcript
       ↓
faster-whisper
       ↓
Transcript
       ↓
Gemma 4
       ↓
Summary + Decisions + Action Items
       ↓
Pydantic Validation
       ↓
Human Review
       ↓
JSON + TXT

Project Structure
meeting-notes/
├── data/
│   ├── meeting_transcript.txt
│   ├── meeting_notes.json
│   └── meeting_notes.txt
├── src/
│   └── app.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
Setup

Create and activate the virtual environment:

python -m venv .venv
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Create .env:

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:latest

Make sure Ollama is running and Gemma 4 is available.

Run
python src\app.py

Open:

http://127.0.0.1:7860
Important Rule

The model must use only information stated in the meeting.

No invented decisions
No invented owners
No invented due dates
Missing information is kept empty

Compatibility Issue

While adding audio transcription, a compatibility issue occurred between faster-whisper and PyAV.

faster-whisper was using the metadata_errors argument, which was not supported by PyAV 19.0.0.

The issue was resolved by using a compatible PyAV version:

faster-whisper: 1.2.1
PyAV: 16.0.1
Output

The application saves:

data/meeting_notes.json
data/meeting_notes.txt

The generated notes can be reviewed and corrected in the Gradio UI before saving.