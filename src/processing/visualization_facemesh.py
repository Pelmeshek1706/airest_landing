import cv2
import mediapipe as mp
import numpy as np

class FaceMeshVisualization:
    def __init__(self, input_video_path, output_video_path):
        # Initialize MediaPipe FaceMesh with refined landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Enables refined landmarks, including pupils
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Video input and output paths
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        # VideoWriter object will be initialized later
        self.out_video = None
        self.fps = None
        self.frame_width = None
        self.frame_height = None

    def initialize_video_writer(self, frame):
        """
        Initialize the VideoWriter with the frame dimensions.
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Codec for saving the video
        self.out_video = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))

    def run(self):
        # Open the input video file
        cap = cv2.VideoCapture(self.input_video_path)

        # Check if the video file is opened correctly
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.input_video_path}")
            return

        # Get FPS, width, and height from the input video
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {self.fps}")

        # Loop through frames from the video
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("End of video file or error.")
                break

            # Convert the frame color space from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Initialize VideoWriter if it hasn't been done yet
            if self.out_video is None:
                self.initialize_video_writer(frame)

            # Process the frame and detect the face mesh landmarks
            results = self.face_mesh.process(rgb_frame)

            # Create a white mask for annotation
            mask = np.ones_like(frame) * 255

            # If landmarks are found, draw them on the white mask
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the face landmarks and connections (on the white mask)
                    self.mp_drawing.draw_landmarks(
                        image=mask,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,  # Show connections
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),  # Green landmarks
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)  # Red connections
                    )

                    # Define pupil indices for both eyes (refined landmarks for the rhombus)
                    right_pupil_indices = [469, 470, 471, 472]  # Right pupil
                    left_pupil_indices = [474, 475, 476, 477]   # Left pupil

                    # Check if the number of landmarks is enough to include the pupil indices
                    if len(face_landmarks.landmark) > max(left_pupil_indices):
                        # Function to draw rhombus for a set of landmarks
                        def draw_rhombus(indices):
                            pupil_coords = []
                            for idx in indices:
                                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                                pupil_coords.append((x, y))
                                # Draw pupils in blue
                                cv2.circle(mask, (x, y), 2, (0, 0, 255), -1)  # Blue dots for pupils

                            # Draw rhombus shape by connecting consecutive points
                            for i in range(len(pupil_coords)):
                                next_point = pupil_coords[(i + 1) % len(pupil_coords)]  # Wrap around to form a closed shape
                                cv2.line(mask, pupil_coords[i], next_point, (0, 0, 255), 2)  # Red lines connecting pupils in a rhombus

                        # Draw rhombus around right and left pupils
                        draw_rhombus(right_pupil_indices)
                        draw_rhombus(left_pupil_indices)

            # Write the annotated frame to the video
            self.out_video.write(mask)

        # Release video resources
        cap.release()
        if self.out_video is not None:
            self.out_video.release()

# Set the input and output paths and run the visualization
input_video_path = 'output_videos/recorded_videos.mov'  # Path to your input video
output_video_path = 'output_videos/output_landmark_visualization.mp4'  # Path to save the output
face_mesh_vis = FaceMeshVisualization(input_video_path, output_video_path)
face_mesh_vis.run()