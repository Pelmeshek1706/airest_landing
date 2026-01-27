import os
import asyncio
import socket
import time
import json
import logging
from collections import deque
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, Request, UploadFile, File, Form

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'api_test_results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from werkzeug.utils import secure_filename
import socketio


# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler('debug.log', maxBytes=1024 * 1024 * 5, backupCount=2)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)
logging.getLogger('socketio').setLevel(logging.ERROR)

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app):
    yield
    logging.info('Server shutdown: cleaning up resources')
    await cleanup_resources()

# --- FastAPI and Socket.IO Setup ---
app = FastAPI(lifespan=lifespan)
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


async def cleanup_resources():
    global api, cap, calibration_active, recording
    async with lock:
        local_api = api
        local_cap = cap
        api = None
        cap = None
        calibration_active = False
        recording = False
        user_input_state['space_down'] = False
    if local_cap:
        local_cap.release()
    if local_api:
        local_api.stop()


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

@app.get('/test', response_class=HTMLResponse)
async def test(request: Request):
    return templates.TemplateResponse('test.html', {'request': request})

@app.get('/analytics_summary', response_class=HTMLResponse)
async def analytics_summary(request: Request):
    return templates.TemplateResponse('analytics_summary.html', {'request': request})

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
    async with lock:
        if not recording or api is None or cap is None:
            return JSONResponse({'status': 'not_recording'})
        api_local = api
        cap_local = cap
        slide_idx = current_slide

    import cv2
    ret, frame = await asyncio.to_thread(cap_local.read)
    if not ret:
        return JSONResponse({'status': 'camera_error'})

    frame = cv2.flip(frame, 1)
    gaze = await asyncio.to_thread(api_local.get_gaze, frame)
    timestamp = time.time()

    async with lock:
        gaze_data.append({'time': timestamp, 'slide': slide_idx, 'x': gaze[0], 'y': gaze[1]})

    return JSONResponse({'status': 'ok'})

@app.get('/finish')
async def finish():
    logging.info('Finishing test session')
    async with lock:
        snapshot = list(gaze_data)
    with open(os.path.join(RESULTS_DIR, 'web_gaze_data.json'), 'w') as f:
        json.dump(snapshot, f)
    logging.info('Gaze data saved to api_test_results/web_gaze_data.json')
    await cleanup_resources()
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

@sio.event
async def disconnect(sid):
    logging.info(f'Client disconnected: {sid}')
    await cleanup_resources()

@sio.on('start_test')
async def handle_start_test(sid, data):
    global api, cap, calibration_active
    try:
        import cv2
        from src.api import GazeTrackerAPI
    except Exception as e:
        logging.error('Gaze tracking dependencies missing: %s', e)
        await sio.emit('calibration_error', {'message': 'Server missing OpenCV'}, to=sid)
        return
    async with lock:
        width = int(data.get('width', 1280))
        height = int(data.get('height', 720))
        logging.info(f'Starting test with screen dimensions {width}x{height}')
        api_local = GazeTrackerAPI(screen_width=width, screen_height=height)
        api_local.start()
        cap_local = cv2.VideoCapture(0)
        if not cap_local.isOpened():
            api_local.stop()
            await sio.emit('calibration_error', {'message': 'Camera not available'}, to=sid)
            return
        cap_local.set(cv2.CAP_PROP_FPS, 30)
        api_local.start_calibration(calibration_params)
        api = api_local
        cap = cap_local
        calibration_active = True
        sio.start_background_task(calibration_stream_task, sid)
    await sio.emit('test_started', to=sid)

async def calibration_stream_task(sid):
    global api, cap, calibration_active
    import cv2
    fps_tracker = deque(maxlen=20)
    target_frame_time = 1.0 / 30.0
    while calibration_active:
        loop_start = time.time()
        async with lock:
            api_local = api
            cap_local = cap
            if not api_local or not api_local.is_calibrating:
                calibration_active = False
                break
            is_space_down = user_input_state.get('space_down', False)
        current_time = time.time()
        ret, frame = await asyncio.to_thread(cap_local.read)
        if not ret:
            logging.error('Failed to read frame from camera')
            await sio.emit('calibration_error', {'message': 'Camera error'}, to=sid)
            await cleanup_resources()
            break
        frame = cv2.flip(frame, 1)
        status = await asyncio.to_thread(
            api_local.process_calibration_step,
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
    logging.info(f'Starting ASGI server on port {port}. Open http://localhost:{port} in a browser')
    import uvicorn
    uvicorn.run(app_sio, host='127.0.0.1', port=port)
