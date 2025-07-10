import socket
import cv2
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, jsonify, request, render_template
from threading import Lock
from src.api import GazeTrackerAPI

app = Flask(__name__, static_folder='web/static', template_folder='web/templates')

# configure logging to a file and reduce console noise
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler('debug.log', maxBytes=1024 * 1024 * 5, backupCount=2)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
if app.logger.handlers:
    app.logger.handlers.clear()
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
logging.getLogger('werkzeug').setLevel(logging.ERROR)


api = None
cap = None
calibration_params = {
    'instruction_frames': 50,
    'calibration_frames': 60,
    'test_frames': 30,
    'circle_radius': 20,
    'outer_circle_initial_radius': 80,
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
    with lock:
        dims = request.get_json() or {}
        width = int(dims.get('width', 1280))
        height = int(dims.get('height', 720))
        
        app.logger.info(f"Starting test with screen dimensions {width}x{height}")
        api = GazeTrackerAPI(screen_width=width, screen_height=height)
        api.start()
        
        cap = cv2.VideoCapture(0)
        api.start_calibration(calibration_params)
    
    return jsonify({'status': 'started'})

@app.route('/calibration_step')
def calibration_step():
    ret, frame = cap.read()
    if not ret:
        app.logger.error("Failed to read frame from camera")
        return jsonify({'status': 'error', 'display_info': {'type': 'message', 'text': 'Camera error'}})
    
    frame = cv2.flip(frame, 1)
    
    with lock:
        if not api or not api.is_calibrating:
            return jsonify({'status': 'not_calibrating'})

        space_down = request.args.get('space_down', '0') == '1'
        user_input = {'space_down': space_down}
        status = api.process_calibration_step(frame, user_input=user_input)
    
    return jsonify(status)

@app.route('/start_slides')
def start_slides():
    global recording, gaze_data
    with lock:
        app.logger.info("Starting slide presentation and gaze recording")
        recording = True
        gaze_data = []
    return jsonify({'status': 'ok'})

@app.route('/set_slide/<int:idx>')
def set_slide(idx):
    global current_slide
    with lock:
        current_slide = idx
    return jsonify({'status': 'ok'})

@app.route('/record_gaze')
def record_gaze():
    global gaze_data
    if not recording:
        return jsonify({'status': 'not_recording'})
    
    ret, frame = cap.read()
    if not ret:
        return jsonify({'status': 'camera_error'})
    
    frame = cv2.flip(frame, 1)
    gaze = api.get_gaze(frame)
    timestamp = time.time()
    
    with lock:
        gaze_data.append({'time': timestamp, 'slide': current_slide, 'x': gaze[0], 'y': gaze[1]})
    
    return jsonify({'status': 'ok'})

@app.route('/finish')
def finish():
    global recording, api, cap
    with lock:
        app.logger.info("Finishing test session")
        recording = False
        if cap:
            cap.release()
            cap = None
        if api:
            api.stop()
            api = None

        with open('api_test_results/web_gaze_data.json', 'w') as f:
            json.dump(gaze_data, f)
        app.logger.info("Gaze data saved to api_test_results/web_gaze_data.json")

    return jsonify({'status': 'done'})


if __name__ == '__main__':
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    port = find_free_port()
    app.logger.info(f"Starting server on port {port}. Open http://127.0.0.1:{port} in a browser")
    app.run(host='0.0.0.0', port=port, debug=False)
    