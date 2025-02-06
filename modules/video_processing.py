import cv2
import pandas as pd
from pathlib import Path
from .visualization_test import FaceMeshVisualization
# from ..modules.epog_analyzer import EPOGAnalyzer

class Video_processor:
    def __init__(self, epog_analyzer):
        self.epog_analyzer = epog_analyzer

    def play_and_record_video(self, video_paths, output_filename):
        output_path = Path('other') / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        webcam_cap = cv2.VideoCapture(0)
        if not webcam_cap.isOpened():
            print("Error opening webcam")
            return

        ret_webcam, frame_webcam = webcam_cap.read()
        if not ret_webcam:
            print("Error reading frame from webcam")
            return

        webcam_frame_width = frame_webcam.shape[1]
        webcam_frame_height = frame_webcam.shape[0]

        fps = 30
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (webcam_frame_width, webcam_frame_height))

        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('Video', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

        for video_path in video_paths:
            video_cap = cv2.VideoCapture(video_path)
            if not video_cap.isOpened():
                print(f"Error opening video file {video_path}")
                continue

            while video_cap.isOpened():
                ret_video, frame_video = video_cap.read()
                if not ret_video:
                    break

                ret_webcam, frame_webcam = webcam_cap.read()
                if not ret_webcam:
                    print("Error reading frame from webcam")
                    break

                cv2.imshow('Video', frame_video)
                out.write(frame_webcam)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_cap.release()
            cv2.destroyAllWindows()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam_cap.release()
        out.release()

    def estimate_gaze_from_video(self, video_filename, fps):
        video_path = Path('other') / video_filename
        video_cap = cv2.VideoCapture(str(video_path))
        if not video_cap.isOpened():
            print(f"Error opening video file {video_path}")
            return pd.DataFrame()

        gaze_data = []

        frame_count = 0
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break

            frame_time = frame_count / fps
            screen_x, screen_y = self.epog_analyzer.analyze(frame)
            gaze_data.append((frame_time, screen_x, screen_y))

            frame_count += 1

        video_cap.release()
        cv2.destroyAllWindows()

        return pd.DataFrame(gaze_data, columns=['time', 'screen_x', 'screen_y'])

    def create_annotated_video(self, video_paths, gaze_df, annotated_output_filename, fps=30):
        annotated_output_path = Path('output_videos') / annotated_output_filename
        annotated_output_path.parent.mkdir(parents=True, exist_ok=True)

        first_video = cv2.VideoCapture(video_paths[0])
        if not first_video.isOpened():
            print(f"Error opening video file {video_paths[0]}")
            return

        frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_video.release()

        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out_annotated = cv2.VideoWriter(str(annotated_output_path), fourcc, fps, (frame_width, frame_height))

        for video_path in video_paths:
            video_cap = cv2.VideoCapture(video_path)
            if not video_cap.isOpened():
                print(f"Error opening video file {video_path}")
                continue

            frame_count = 0
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break

                current_time = frame_count / fps
                closest_gaze = gaze_df.iloc[(gaze_df['time'] - current_time).abs().argsort()[:1]]
                annotated_frame = frame.copy()
                if not closest_gaze.empty:
                    screen_x = closest_gaze['screen_x'].values[0]
                    screen_y = closest_gaze['screen_y'].values[0]
                    if pd.notna(screen_x) and pd.notna(screen_y):
                        cv2.circle(annotated_frame, (int(screen_x), int(screen_y)), 10, (0, 0, 255), -1)

                out_annotated.write(annotated_frame)
                frame_count += 1

            video_cap.release()

        out_annotated.release()
        print("Starting fase_mesh...")
        input_video_path = 'other/output_videos/recorded_videos.mov'  # Path to your input video
        output_video_path = 'output_videos/output_landmark_visualization.mp4'  # Path to save the output
        face_mesh_vis = FaceMeshVisualization(input_video_path, output_video_path)
        face_mesh_vis.run()
        print("Finished fase_mesh...")


    def func(self):
        pass

    def run_processor(self, video_paths, output_filename):
        print("Playing and recording video...")
        self.play_and_record_video(video_paths, output_filename)

        print("\nYou can rest now. Do not close any window.\n\nEstimating gaze...")
        gaze_df = self.estimate_gaze_from_video(output_filename, fps=30)

        gaze_csv_path = Path('other') / output_filename.replace('.mov', '_gaze_data.csv')
        gaze_df.to_csv(gaze_csv_path, index=False)
        print(f"Gaze data saved to {gaze_csv_path}")

        print("Creating annotated video...")
        annotated_output_filename = output_filename.replace('.mov', '_annotated.mov')
        self.create_annotated_video(video_paths, gaze_df, annotated_output_filename)
        print(f"Annotated video saved to {annotated_output_filename}")

# Usage example:
# epog_analyzer = EpogAnalyzer()  # Assuming you have this class defined elsewhere
# processor = Video_processor(epog_analyzer)
# video_paths = ['video1.mp4', 'video2.mp4']
# output_filename = 'output.mov'
# processor.run_processor(video_paths, output_filename)