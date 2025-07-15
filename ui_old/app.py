import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # Serve the UI only
    return render_template('frontend.html')

@app.route('/upload', methods=['POST'])
def upload():
    session_id = request.form.get('session_id')
    events_json = request.form.get('events', '[]')
    video_file  = request.files.get('video_blob')

    if not session_id or not video_file:
        return jsonify({'error': 'Missing session_id or video_blob'}), 400

    # Save video file
    video_filename = secure_filename(f"{session_id}.webm")
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    video_file.save(video_path)

    # Save events JSON
    events_filename = secure_filename(f"{session_id}_events.json")
    events_path = os.path.join(app.config['UPLOAD_FOLDER'], events_filename)
    with open(events_path, 'w') as f:
        f.write(events_json)

    # Placeholder: feature extraction & DB logic goes here later

    return jsonify({'success': True, 'video_path': video_path, 'events_path': events_path}), 200

if __name__ == '__main__':
    app.run(debug=True)
