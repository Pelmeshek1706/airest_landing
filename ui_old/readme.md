# AIREST Mental Health Test

A browser-based mental health assessment app that records your voice during each task, displays a live waveform and timer, and uploads the session data to a Flask backend.

## Features

- Four placeholder tasks (read sentence, describe image, count, free response)  
- Audio recording with MediaRecorder & WebAudio API  
- Live waveform visualization and elapsed timer  
- Tasks loaded from JSON (`tasks_text.json`, `tasks_image.json`), sorted by ID  
- UI in Ukrainian with Tailwind CSS  
- Saves recording in MP4/WebM and event log to server  

## Prerequisites

- Git  
- Python 3.10+  
- (Optional) Conda or built-in `venv` for virtual environments  

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/poluidol2/airest_ui
   
2. **Create & activate a Python environment**
    ```bash    
    conda create -n airest-env python=3.10
    conda activate airest-env
   
3. **Install dependencies**
    ```bash   
   pip install -r requirements.txt 
   
## Running
   ```bash  
    python app.py
