import os
import sys
from modules.epog_analyzer import EPOGAnalyzer
from modules.video_processing import Video_processor
from modules.utils import ensure_directories_exist

# Paths to the videos you want to show
video_paths = ['images/test2.mp4']
output_dir = 'output_videos'  # Directory to save the recorded videos
analysis_dir = 'analysis_results'  # Directory to save the analysis results
epog_test_error_dir = 'epog_test_errors'  # Directory to save EPOG test errors

# Ensure output directories exist
ensure_directories_exist([output_dir, analysis_dir, epog_test_error_dir])

def main():
    epog_analyzer = EPOGAnalyzer(epog_test_error_dir, sys.argv)
    epog_analyzer.perform_calibration()

    # recorded_video_path = os.path.join(output_dir, 'recorded_videos.mov')
    # processor = Video_processor()
    # processor.play_and_record_videos(video_paths, recorded_video_path, epog_analyzer)

if __name__ == "__main__":
    main()
    