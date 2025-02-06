import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from calibration.epog_analyzer import EPOGAnalyzer
from processing.video_processing import Video_processor
from utils.utils import prepare_directories

# define paths to assets and output directories
VIDEO_PATHS = ['images/test2.mp4']
OUTPUT_DIR = 'output_videos'         # directory to save recorded videos
ANALYSIS_DIR = 'analysis_results'    # directory to save analysis results
EPOG_TEST_ERROR_DIR = 'epog_test_errors'  # directory to save epog test errors

# create required output directories if they do not exist
prepare_directories([OUTPUT_DIR, ANALYSIS_DIR, EPOG_TEST_ERROR_DIR])

def main(play_video_test=False, stabilize_gaze=True):
    """
    run the gaze calibration demo and optionally a video test.

    :param play_video_test: if true, play and record a video demonstration.
    :param stabilize_gaze: if true, enable gaze stabilization in the analyzer.
    """
    # create an epog analyzer instance with gaze stabilization enabled/disabled
    epog_analyzer = EPOGAnalyzer(stabilize=stabilize_gaze)
    
    # perform calibration using the webcam
    epog_analyzer.perform_calibration()

    # if requested, play a test video and record the output
    if play_video_test:
        recorded_video_path = os.path.join(OUTPUT_DIR, 'recorded_videos.mov')
        processor = Video_processor()
        processor.play_and_record_videos(VIDEO_PATHS, recorded_video_path, epog_analyzer)

if __name__ == "__main__":
    main()
