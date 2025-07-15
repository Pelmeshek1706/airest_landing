import os
import asyncio
import socket
import cv2
import time
import json
import logging
from collections import deque
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, Request, UploadFile, File, Form
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from werkzeug.utils import secure_filename
import socketio

from src.api import GazeTrackerAPI

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler('debug.log', maxBytes=1024 * 1024 * 5, backupCount=2)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)
logging.getLogger('socketio').setLevel(logging.ERROR)

# --- FastAPI and Socket.IO Setup ---
app = FastAPI()
app.mount('/static', StaticFiles(directory='web/static'), name='static')
templates = Jinja2Templates(directory='web/templates')

sio = socketio.AsyncServer(async_mode='asgi')
app_sio = socketio.ASGIApp(sio, other_asgi_app=app)

# --- Global State ---
api = None
cap = None
lock = asyncio.Lock()
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

user_input_state = {'space_down': False}

# --- Routes ---
@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/attention', response_class=HTMLResponse)
async def attention(request: Request):
    return templates.TemplateResponse('attention.html', {'request': request})

@app.get('/speech', response_class=HTMLResponse)
async def speech(request: Request):
    return templates.TemplateResponse('speech.html', {'request': request})
@app.get('/start_slides')
async def start_slides():
    global recording, gaze_data
    async with lock:
        logging.info('Starting slide presentation and gaze recording')
        recording = True
        gaze_data = []
    return JSONResponse({'status': 'ok'})

@app.get('/set_slide/{idx}')
async def set_slide_route(idx: int):
    global current_slide
    async with lock:
        current_slide = idx
    return JSONResponse({'status': 'ok'})

@app.get('/record_gaze')
async def record_gaze_route(slide: int = 0):
    global gaze_data
    if not recording:
        return JSONResponse({'status': 'not_recording'})

    ret, frame = await asyncio.to_thread(cap.read)
    if not ret:
        return JSONResponse({'status': 'camera_error'})

    frame = cv2.flip(frame, 1)
    gaze = await asyncio.to_thread(api.get_gaze, frame)
    timestamp = time.time()

    async with lock:
        gaze_data.append({'time': timestamp, 'slide': current_slide, 'x': gaze[0], 'y': gaze[1]})

    return JSONResponse({'status': 'ok'})

@app.get('/finish')
async def finish():
    global recording, api, cap, calibration_active
    async with lock:
        logging.info('Finishing test session')
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
        logging.info('Gaze data saved to api_test_results/web_gaze_data.json')
    return JSONResponse({'status': 'done'})

@app.post('/upload')
async def upload(session_id: str = Form(...), events: str = Form('[]'), video_blob: UploadFile = File(...)):
    video_ext = '.webm' if video_blob.filename.endswith('.webm') else '.mp4'
    video_path = os.path.join(UPLOAD_FOLDER, secure_filename(session_id + video_ext))
    with open(video_path, 'wb') as f:
        f.write(await video_blob.read())
    events_path = os.path.join(UPLOAD_FOLDER, secure_filename(f'{session_id}_events.json'))
    with open(events_path, 'w') as f:
        f.write(events)
    return JSONResponse({'success': True, 'video_path': video_path, 'events_path': events_path})
# --- Socket.IO Events ---
@sio.event
async def connect(sid, environ):
    logging.info(f'Client connected: {sid}')

@sio.on('start_test')
async def handle_start_test(sid, data):
    global api, cap, calibration_active
    async with lock:
        width = int(data.get('width', 1280))
        height = int(data.get('height', 720))
        logging.info(f'Starting test with screen dimensions {width}x{height}')
        api = GazeTrackerAPI(screen_width=width, screen_height=height)
        api.start()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        api.start_calibration(calibration_params)
        calibration_active = True
        sio.start_background_task(calibration_stream_task, sid)
    await sio.emit('test_started', to=sid)

async def calibration_stream_task(sid):
    global api, cap, calibration_active
    fps_tracker = deque(maxlen=20)
    target_frame_time = 1.0 / 30.0
    while calibration_active:
        loop_start = time.time()
        async with lock:
            if not api or not api.is_calibrating:
                calibration_active = False
                break
            is_space_down = user_input_state.get('space_down', False)
        current_time = time.time()
        ret, frame = await asyncio.to_thread(cap.read)
        if not ret:
            logging.error('Failed to read frame from camera')
            await sio.emit('calibration_error', {'message': 'Camera error'}, to=sid)
            break
        frame = cv2.flip(frame, 1)
        status = await asyncio.to_thread(
            api.process_calibration_step,
            frame,
            user_input={'space_down': is_space_down},
        )
        async with lock:
            fps_tracker.append(current_time)
            fps = 0
            if len(fps_tracker) > 1:
                time_span = fps_tracker[-1] - fps_tracker[0]
                if time_span > 0:
                    fps = (len(fps_tracker) - 1) / time_span
            status['fps'] = fps
        await sio.emit('calibration_update', status, to=sid)
        if status['status'] == 'finished_all':
            calibration_active = False
        elapsed = time.time() - loop_start
        await asyncio.sleep(max(0, target_frame_time - elapsed))
    logging.info('Calibration stream has ended.')

@sio.on('user_input')
async def handle_user_input(sid, data):
    user_input_state['space_down'] = data.get('space_down', False)

if __name__ == '__main__':
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    port = find_free_port()
    logging.info(f'Starting ASGI server on port {port}. Open http://127.0.0.1:{port} in a browser')
    import uvicorn
    uvicorn.run(app_sio, host='0.0.0.0', port=port)
