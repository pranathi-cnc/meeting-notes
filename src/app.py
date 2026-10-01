import json
import os
from pathlib import Path
from faster_whisper import WhisperModel

import gradio as gr
from dotenv import load_dotenv
from ollama import Client
from pydantic import BaseModel


load_dotenv()

client = Client(host=os.getenv("OLLAMA_URL"))
MODEL = os.getenv("OLLAMA_MODEL")


class ActionItem(BaseModel):
    task: str
    owner: str
    due_date: str


class MeetingNotes(BaseModel):
    summary: str
    decisions: list[str]
    action_items: list[ActionItem]


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPT_PATH = DATA_DIR / "meeting_transcript.txt"
JSON_PATH = DATA_DIR / "meeting_notes.json"
TXT_PATH = DATA_DIR / "meeting_notes.txt"


def read_transcript():
    with open(TRANSCRIPT_PATH, encoding="utf-8") as file:
        return file.read()


def generate_notes(transcript):
    prompt = f"""
You are a meeting notes assistant.

Read the meeting transcript below and extract:

1. A short summary.
2. All decisions explicitly stated in the meeting.
3. All action items.

For every action item provide:
- task
- owner
- due_date

IMPORTANT RULES:
- Use ONLY information explicitly stated in the transcript.
- NEVER invent a decision.
- NEVER invent an owner.
- NEVER invent a due date.
- If the owner is not stated, use an empty string.
- If the due date is not stated, use an empty string.
- If there are no decisions, return an empty decisions list.
- Return ONLY valid JSON.

Required JSON format:

{{
    "summary": "short meeting summary",
    "decisions": [],
    "action_items": [
        {{
            "task": "",
            "owner": "",
            "due_date": ""
        }}
    ]
}}

Meeting transcript:

{transcript}
"""

    response = client.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
    )

    return response.message.content


def save_notes(notes):
    DATA_DIR.mkdir(exist_ok=True)

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(notes.model_dump(), file, indent=4)

    with open(TXT_PATH, "w", encoding="utf-8") as file:
        file.write("MEETING NOTES\n")
        file.write("=" * 40 + "\n\n")

        file.write("Summary:\n")
        file.write(notes.summary + "\n\n")

        file.write("Decisions:\n")

        if notes.decisions:
            for decision in notes.decisions:
                file.write(f"- {decision}\n")
        else:
            file.write("- No decisions recorded.\n")

        file.write("\nAction Items:\n")

        if notes.action_items:
            for index, action in enumerate(notes.action_items, start=1):
                file.write(f"\n{index}. {action.task}\n")
                file.write(f"   Owner: {action.owner}\n")
                file.write(f"   Due Date: {action.due_date}\n")
        else:
            file.write("- No action items recorded.\n")


def transcribe_audio(audio_path):
    model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )

    segments, info = model.transcribe(audio_path)

    transcript = ""

    for segment in segments:
        transcript += segment.text.strip() + " "

    return transcript.strip()

def generate_for_ui(transcript):
    

    raw_response = generate_notes(transcript)

    data = json.loads(raw_response)

    notes = MeetingNotes.model_validate(data)

    decisions = "\n".join(notes.decisions)

    action_text = ""

    for action in notes.action_items:
        action_text += (
            f"Task: {action.task}\n"
            f"Owner: {action.owner}\n"
            f"Due Date: {action.due_date}\n\n"
        )

    return (
        transcript,
        notes.summary,
        decisions,
        action_text.strip()
    )


def save_from_ui(summary, decisions, actions):
    decision_list = [
        item.strip()
        for item in decisions.split("\n")
        if item.strip()
    ]

    action_items = []

    blocks = actions.split("\n\n")

    for block in blocks:
        lines = block.strip().splitlines()

        task = ""
        owner = ""
        due_date = ""

        for line in lines:
            if line.startswith("Task:"):
                task = line.replace("Task:", "", 1).strip()

            elif line.startswith("Owner:"):
                owner = line.replace("Owner:", "", 1).strip()

            elif line.startswith("Due Date:"):
                due_date = line.replace("Due Date:", "", 1).strip()

        if task:
            action_items.append(
                ActionItem(
                    task=task,
                    owner=owner,
                    due_date=due_date
                )
            )

    notes = MeetingNotes(
        summary=summary.strip(),
        decisions=decision_list,
        action_items=action_items
    )

    save_notes(notes)

    return (
        f"Saved successfully!\n\n"
        f"JSON: {JSON_PATH}\n"
        f"Text: {TXT_PATH}"
    )


with gr.Blocks(title="Meeting Notes Generator") as demo:

    gr.Markdown("# Meeting Notes Generator")
    gr.Markdown(
        "Generate meeting notes using Gemma 4 and review them before saving."
    )

    audio_input = gr.Audio(
    label="Upload Meeting Audio",
    type="filepath"
)

    transcribe_button = gr.Button("Transcribe Audio")

    transcript_box = gr.Textbox(
        label="Meeting Transcript",
        lines=10
    )
    transcribe_button.click(
    fn=transcribe_audio,
    inputs=audio_input,
    outputs=transcript_box
)

    generate_button = gr.Button("Generate Meeting Notes")

    summary_box = gr.Textbox(
        label="Summary",
        lines=5
    )

    decisions_box = gr.Textbox(
        label="Decisions (one per line)",
        lines=5
    )

    actions_box = gr.Textbox(
        label="Action Items",
        lines=12,
        placeholder=(
            "Task: Send the revised design\n"
            "Owner: Priya\n"
            "Due Date: Friday\n\n"
            "Task: Test the website before the launch\n"
            "Owner: Rahul\n"
            "Due Date:"
        )
    )

    save_button = gr.Button("Save Corrected Notes")

    status_box = gr.Textbox(
        label="Status",
        lines=4
    )
    

    generate_button.click(
    fn=generate_for_ui,
    inputs=transcript_box,
    outputs=[
        transcript_box,
        summary_box,
        decisions_box,
        actions_box
    ]
)

    save_button.click(
        fn=save_from_ui,
        inputs=[
            summary_box,
            decisions_box,
            actions_box
        ],
        outputs=status_box
    )


if __name__ == "__main__":
    demo.launch()