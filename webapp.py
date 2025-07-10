import eventlet
eventlet.monkey_patch(thread=False)

import socket
import cv2
import time
import json
import logging
from collections import deque
from logging.handlers import RotatingFileHandler
from eventlet.semaphore import BoundedSemaphore
from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO, emit
from src.api import GazeTrackerAPI


# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler('debug.log', maxBytes=1024 * 1024 * 5, backupCount=2)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)


app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
socketio = SocketIO(app, async_mode='eventlet')

# --- Global State ---
api = None
cap = None
# use the eventlet-safe semaphore
lock = BoundedSemaphore(1)
calibration_active = False

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


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_test')
def handle_start_test(data):
    global api, cap, calibration_active
    with lock:
        width = int(data.get('width', 1280))
        height = int(data.get('height', 720))
        
        logging.info(f"Starting test with screen dimensions {width}x{height}")
        api = GazeTrackerAPI(screen_width=width, screen_height=height)
        api.start()
        
        cap = cv2.VideoCapture(0)
        api.start_calibration(calibration_params)
        calibration_active = True
        
        socketio.start_background_task(target=calibration_stream_task)
    emit('test_started')

def calibration_stream_task():
    global api, cap, calibration_active
    fps_tracker = deque(maxlen=20)

    while calibration_active:
        with lock:
            if not api or not api.is_calibrating:
                calibration_active = False
                break
            
            is_space_down = user_input_state.get('space_down', False)

        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read frame from camera")
            socketio.emit('calibration_error', {'message': 'Camera error'})
            break
        
        frame = cv2.flip(frame, 1)

        with lock:
            status = api.process_calibration_step(frame, user_input={'space_down': is_space_down})
            
            fps_tracker.append(current_time)
            fps = 0
            if len(fps_tracker) > 1:
                time_span = fps_tracker[-1] - fps_tracker[0]
                if time_span > 0:
                    fps = (len(fps_tracker) - 1) / time_span
            status['fps'] = fps

        socketio.emit('calibration_update', status)
        
        if status['status'] == 'finished_all':
            calibration_active = False
            
        socketio.sleep(0.01)
    
    logging.info("Calibration stream has ended.")


user_input_state = {'space_down': False}
@socketio.on('user_input')
def handle_user_input(data):
    global user_input_state
    user_input_state['space_down'] = data.get('space_down', False)


@app.route('/start_slides')
def start_slides():
    global recording, gaze_data
    with lock:
        logging.info("Starting slide presentation and gaze recording")
        recording = True
        gaze_data = []
    return jsonify({'status':'ok'})

@app.route('/set_slide/<int:idx>')
def set_slide_route(idx):
    global current_slide
    with lock:
        current_slide = idx
    return jsonify({'status':'ok'})

@app.route('/record_gaze')
def record_gaze_route():
    global gaze_data
    if not recording:
        return jsonify({'status':'not_recording'})
    
    ret, frame = cap.read()
    if not ret:
        return jsonify({'status':'camera_error'})
    
    frame = cv2.flip(frame, 1)
    gaze = api.get_gaze(frame)
    timestamp = time.time()
    
    with lock:
        gaze_data.append({'time': timestamp, 'slide': current_slide, 'x': gaze[0], 'y': gaze[1]})
    
    return jsonify({'status':'ok'})

@app.route('/finish')
def finish():
    global recording, api, cap, calibration_active
    with lock:
        logging.info("Finishing test session")
        recording = False
        calibration_active = False
        if cap:
            cap.release()
            cap = None
        if api:
            api.stop()
            api = None

        with open('api_test_results/web_gaze_data.json', 'w') as f:
            json.dump(gaze_data, f)
        logging.info("Gaze data saved to api_test_results/web_gaze_data.json")

    return jsonify({'status':'done'})


if __name__ == '__main__':
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    port = find_free_port()
    logging.info(f"Starting server on port {port}. Open http://127.0.0.1:{port} in a browser")
    socketio.run(app, host='0.0.0.0', port=port)