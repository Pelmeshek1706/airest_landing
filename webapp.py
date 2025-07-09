import cv2
import time
import json
from flask import Flask, jsonify, request, render_template
from threading import Lock
from src.api import GazeTrackerAPI

app = Flask(__name__, static_folder='web/static', template_folder='web/templates')

api = None
cap = None
calibration_params = {
    'instruction_frames': 50,
    'fixation_frames': 15,
    'calibration_frames': 30,
    'test_frames': 30,
    'circle_radius': 25,
    'run_test_stage': True,
}
current_slide = 0
recording = False
gaze_data = []
lock = Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_test', methods=['POST'])
def start_test():
    global api, cap
    dims = request.get_json() or {}
    width = int(dims.get('width', 1280))
    height = int(dims.get('height', 720))
    api = GazeTrackerAPI(screen_width=width, screen_height=height)
    api.start()
    cap = cv2.VideoCapture(0)
    api.start_calibration(calibration_params)
    return jsonify({'status': 'started'})

@app.route('/calibration_step')
def calibration_step():
    ret, frame = cap.read()
    if not ret:
        return jsonify({'status': 'error', 'display_info': {'type':'message','text':'Camera error'}})
    frame = cv2.flip(frame,1)
    status = api.process_calibration_step(frame)
    return jsonify(status)

@app.route('/start_slides')
def start_slides():
    global recording, gaze_data
    recording = True
    gaze_data = []
    return jsonify({'status':'ok'})

@app.route('/set_slide/<int:idx>')
def set_slide(idx):
    global current_slide
    current_slide = idx
    return jsonify({'status':'ok'})

@app.route('/record_gaze')
def record_gaze():
    global gaze_data
    if not recording:
        return jsonify({'status':'not_recording'})
    ret, frame = cap.read()
    if not ret:
        return jsonify({'status':'camera_error'})
    frame = cv2.flip(frame,1)
    gaze = api.get_gaze(frame)
    timestamp = time.time()
    with lock:
        gaze_data.append({'time': timestamp, 'slide': current_slide, 'x': gaze[0], 'y': gaze[1]})
    return jsonify({'status':'ok'})

@app.route('/finish')
def finish():
    global recording
    recording = False
    if cap:
        cap.release()
    if api:
        api.stop()
    # save gaze data
    with open('api_test_results/web_gaze_data.json','w') as f:
        json.dump(gaze_data,f)
    return jsonify({'status':'done'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

