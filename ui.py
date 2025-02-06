import sys
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from qt_material import apply_stylesheet
from modules.epog_analyzer import EPOGAnalyzer
from modules.video_processing import Video_processor
from modules.utils import ensure_directories_exist

import pygame  # for audio
import os
import pyaudio
import wave
import threading
from PyQt5 import QtWidgets, QtGui, QtCore, QtMultimedia
# from pyvidplayer2 import VideoPlayer, Video

# Create a custom stylesheet to redefine fonts
custom_stylesheet = """
    QLabel {
        font-family: 'Roboto';
        font-size: 72px;
        font-weight: bold;
        color: #424242;
    }
    QMessageBox QLabel {
        font-size: 24px;
        font-weight: normal;
        color: #424242;
    }
    #textLabel {
        font-size: 50px;
    }
    #audioLabel {
        font-size: 36px;
    }
    #recommendationLabel {
        font-size: 24px;
        font-weight: normal;
        line-height: 0.8;
    }
    QMessageBox QPushButton {
        height: 30px;
        width: 100px;
        font-size: 24px;
    }
    QPushButton {
        background-color: #ffffff;
        font-family: 'Roboto';
        font-size: 32px;
        font-weight: bold;
        height: 100px;
    }
    QPushButton:hover {
        background-color: #2979ff;
        border: 2px solid #2979ff;
        color: #ffffff;
    }
    QLineEdit {
        font-family: 'Roboto';
        font-size: 32px;
        font-weight: normal;
    }
    QRadioButton {
        padding: 5px;
        font-size: 24px;
        font-weight: normal;
    }
"""

# Paths to the videos you want to show
video_paths = ['videos/test2.mp4']
output_dir = 'output_videos'  # Directory to save the recorded videos
other_dir = 'other'
analysis_dir = 'analysis_results'  # Directory to save the analysis results
epog_test_error_dir = 'epog_test_errors'  # Directory to save EPOG test errors

# Ensure output directories exist
ensure_directories_exist([output_dir, analysis_dir, epog_test_error_dir])
recorded_video_path = os.path.join(output_dir, 'recorded_videos.mov')
tests = {
    "Do you often experience unwanted memories or flashbacks related to the traumatic event?": [
        "Never (0): You don't experience unwanted memories or flashbacks.",
        "Rarely (1): You occasionally experience unwanted memories or flashbacks, but they are infrequent.",
        "Sometimes (2): You experience unwanted memories or flashbacks regularly, but they do not significantly interfere with your life.",
        "Often (3): You frequently experience unwanted memories or flashbacks, and they cause noticeable distress.",
        "Very Often (4): You are constantly overwhelmed by unwanted memories or flashbacks, and they severely disrupt your daily life."
    ],
    "Do you try to avoid situations, places, or people that remind you of the traumatic event?": [
        "Never (0): You do not avoid reminders of the traumatic event.",
        "Rarely (1): You sometimes avoid situations, places, or people that remind you of the trauma.",
        "Sometimes (2): You regularly avoid reminders, and it has a moderate impact on your life.",
        "Often (3): You frequently avoid many situations, places, or people that remind you of the trauma, and it causes significant difficulties.",
        "Very Often (4): You go to great lengths to avoid any reminders of the traumatic event, and it severely restricts your daily functioning."
    ],
    "Do you feel constantly anxious, tense, or easily startled since the traumatic event?": [
        "Never (0): You do not feel increased anxiety or tension.",
        "Rarely (1): You occasionally feel anxious, tense, or easily startled, but it does not affect your life much.",
        "Sometimes (2): You experience anxiety or tension regularly, which sometimes affects your ability to relax.",
        "Often (3): You frequently feel anxious, tense, or easily startled, and it affects your quality of life.",
        "Very Often (4): You feel constantly anxious and on edge, and it severely impacts your well-being."
    ],
    "Do you feel emotionally numb or detached, as if you can’t enjoy things that used to bring you pleasure?": [
        "Never (0): You do not feel emotionally numb or detached.",
        "Rarely (1): You occasionally feel emotionally detached, but it does not significantly affect your enjoyment of life.",
        "Sometimes (2): You regularly feel emotionally numb, and it affects your ability to experience pleasure in some activities.",
        "Often (3): You frequently feel emotionally detached or numb, and it interferes with your relationships or daily activities.",
        "Very Often (4): You feel completely emotionally numb or detached, and it significantly limits your ability to feel joy or connection."
    ],
    "Do you feel guilt or shame about what happened, even if it was beyond your control?": [
        "Never (0): You do not feel guilt or shame related to the traumatic event.",
        "Rarely (1): You occasionally feel guilt or shame, but it does not affect your daily life.",
        "Sometimes (2): You regularly feel guilt or shame, and it affects your self-esteem or mood.",
        "Often (3): You frequently feel guilty or ashamed, and it impacts your ability to engage in normal activities.",
        "Very Often (4): You feel overwhelming guilt or shame, and it severely disrupts your daily functioning and emotional health."
    ]
}

recommendations = """
a) You are doing a fantastic job of taking the initiative to understand yourself better. 
We have noticed that you have some noticeable anxiety and anhedonia. 
It is common for people to experience problems with stress tolerance during these challenging times and it is completely normal.
Talking to a mental health professional can be incredibly helpful. 
They can provide individualized support and practical strategies 
to help you overcome these difficulties and improve your overall well-being. 
Reaching out for support is a sign of strength and a positive step towards feeling more confident and empowered.

Here are the contacts of professionals who will be happy to help...

b) You are doing a fantastic job of taking the initiative to understand yourself better. 
We admire your resilience, keep it up! By taking this proactive step, 
you are investing in your future development and overall health. Keep up the good work and remember 
that preventive visits to mental health professionals can support you on your way to a healthy and happy life.

Here are the contacts of specialists who will be happy to help you if you need it
"""

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.epog_analyzer = None
        self.duration_test = 15  # seconds
        self.video_processor = None
        self.current_question_index = 0  # Track current question index
        self.audio_questions = []
        self.current_audio_index = 0
        pygame.mixer.init()

    def init_ui(self):
        self.setWindowTitle("Testing")

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.setSpacing(40)

        self.name_label = QtWidgets.QLabel("Enter your name")
        self.name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.name_label.setFixedWidth(600)
        self.layout.addWidget(self.name_label)

        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setAlignment(QtCore.Qt.AlignLeft)
        self.name_input.setStyleSheet("border: 2px solid #2979ff; color: #424242; background-color: white")
        self.name_input.setFixedWidth(600)
        self.name_input.setFixedHeight(50)
        self.layout.addWidget(self.name_input)

        self.start_button = QtWidgets.QPushButton("Start textual testing")
        self.start_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.start_button.setFixedWidth(600)
        self.layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_test)

        self.start_audio_button = QtWidgets.QPushButton("Start audio testing")
        self.start_audio_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.start_audio_button.setFixedWidth(600)
        self.layout.addWidget(self.start_audio_button)
        self.start_audio_button.clicked.connect(self.start_audio_test)

        self.setLayout(self.layout)
        self.showMaximized()

    def start_audio_test(self):
        name = self.name_input.text()
        if name.strip() == "":
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter your name")
            return

        # Hide the input fields and start the audio test
        self.name_label.hide()
        self.name_input.hide()
        self.start_button.hide()
        self.start_audio_button.hide()

        # Audio settings
        self.chunk = 1024  # Record in chunks of 1024 samples
        self.sample_format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 1
        self.fs = 44100  # Record at 44100 samples per second
        self.frames = []  # Initialize array to store frames
        self.is_recording = False
        self.p = pyaudio.PyAudio()

        # Load audio questions
        self.load_audio_questions()
        self.show_audio_question()

    def load_audio_questions(self):
        audio_dir = "audio_test/questions"
        self.audio_questions = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.mp3')]
        self.audio_questions.sort()  # Ensure questions are in order

    def show_audio_question(self):
        if self.current_audio_index < len(self.audio_questions):
            # Display the query
            question_keys = list(tests.keys())
            current_question = question_keys[self.current_audio_index]
            self.question_label = QtWidgets.QLabel(current_question)
            self.question_label.setObjectName("audioLabel")
            self.question_label.setAlignment(QtCore.Qt.AlignCenter)
            self.question_label.setWordWrap(True)
            self.layout.addWidget(self.question_label)

            # Play audio
            pygame.mixer.music.load(self.audio_questions[self.current_audio_index])
            pygame.mixer.music.play()

            # Create buttons
            self.start_speaking_button = QtWidgets.QPushButton("Start to talk")
            self.start_speaking_button.setFixedWidth(400)
            self.stop_speaking_button = QtWidgets.QPushButton("Stop talking")
            self.stop_speaking_button.setFixedWidth(400)
            self.start_speaking_button.clicked.connect(self.start_speaking)
            self.stop_speaking_button.clicked.connect(self.stop_speaking)
            self.stop_speaking_button.setEnabled(False)

            # Create horizontal layout for buttons
            self.button_layout = QtWidgets.QHBoxLayout()
            self.button_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
            self.button_layout.addWidget(self.start_speaking_button)
            self.button_layout.addWidget(self.stop_speaking_button)

            # Make sure widgets stretch proportionally and do not exceed screen size
            for i in range(self.button_layout.count()):
                item = self.button_layout.itemAt(i)
                widget = item.widget()
                if widget:
                    widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

            # Add video and buttons to layout
            self.layout.addLayout(self.button_layout)

        else:
            self.start_testing_button = QtWidgets.QPushButton("Start visual testing")
            self.start_testing_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self.start_testing_button.clicked.connect(self.start_testing)
            self.layout.addWidget(self.start_testing_button)

    def start_speaking(self):
        self.start_speaking_button.setEnabled(False)
        self.stop_speaking_button.setEnabled(True)
        self.start_recording()

    def start_recording(self):
        # Clear previous frames
        self.frames = []
        
        # Start recording in a new thread to avoid blocking the UI
        self.is_recording = True
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.start()

    def record_audio(self):
        # Get the default input device
        default_device_info = self.p.get_default_input_device_info()
        default_device_index = default_device_info['index']
        print(f"Using default device: {default_device_info['name']} (Index: {default_device_index})")

        stream = self.p.open(format=self.sample_format,
                            channels=self.channels,
                            rate=self.fs,
                            frames_per_buffer=self.chunk,
                            input=True,
                            input_device_index=default_device_index)

        while self.is_recording:
            data = stream.read(self.chunk)
            self.frames.append(data)
        
        # Stop the stream properly when recording finishes
        stream.stop_stream()
        stream.close()

    def stop_speaking(self):
        self.start_speaking_button.setEnabled(True)
        self.stop_speaking_button.setEnabled(False)
        # Stop the recording
        self.is_recording = False
        self.record_thread.join()  # Wait for the recording thread to finish

        # Save the recorded data as a WAV file
        path_answear = 'audio_test/answers'
        if not os.path.exists(path_answear):
            os.makedirs(path_answear)
        filename = f"{path_answear}/question_{self.current_audio_index}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Move to next question
        self.current_audio_index += 1
        self.clear_audio_question()
        self.show_audio_question()

    def clear_audio_question(self):
        # Remove buttons from layout
        self.layout.removeWidget(self.start_speaking_button)
        self.layout.removeWidget(self.stop_speaking_button)
        self.layout.removeWidget(self.question_label)
        self.start_speaking_button.deleteLater()
        self.stop_speaking_button.deleteLater()
        self.question_label.deleteLater()

    def start_test(self):
        #start test default
        name = self.name_input.text()
        if name.strip() == "":
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter your name")
            return

        # Hide the input fields and start the test questions
        self.name_label.hide()
        self.name_input.hide()
        self.start_button.hide()
        self.start_audio_button.hide()

        self.show_test_question()

    def show_test_question(self):
        # Show the current question and its options
        if self.current_question_index < len(tests):
            question = list(tests.keys())[self.current_question_index]
            answers = tests[question]

            # Create question label
            self.question_label = QtWidgets.QLabel(question)
            self.question_label.setObjectName("textLabel")
            self.question_label.setAlignment(QtCore.Qt.AlignLeft)
            self.question_label.setWordWrap(True)
            self.layout.addWidget(self.question_label)

            # Create radio buttons for each answer
            self.answer_group = QtWidgets.QButtonGroup(self)
            for answer in answers:
                radio_button = QtWidgets.QRadioButton(answer)
                radio_button.setFont(QtGui.QFont("Arial", 16))
                self.answer_group.addButton(radio_button)
                self.layout.addWidget(radio_button)

            # Create a button to submit the answer
            self.submit_button = QtWidgets.QPushButton("Next")
            self.submit_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self.submit_button.clicked.connect(self.submit_answer)
            self.layout.addWidget(self.submit_button)
        else:
            self.start_testing_button = QtWidgets.QPushButton("Start visual testing")
            self.start_testing_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self.start_testing_button.clicked.connect(self.start_testing)
            self.layout.addWidget(self.start_testing_button)

    def submit_answer(self):
        selected_button = self.answer_group.checkedButton()
        if selected_button:
            selected_answer = selected_button.text()
            print(f"Question: {self.question_label.text()}")
            print(f"Selected Answer: {selected_answer}")
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select an answer")
            return

        # Clear the previous question and answers
        self.question_label.deleteLater()
        for button in self.answer_group.buttons():
            self.layout.removeWidget(button)
            button.deleteLater()
        self.submit_button.deleteLater()

        # Move to the next question
        self.current_question_index += 1
        self.show_test_question()

    def start_testing(self):
        self.hide()
        self.calibration_screen()

    def calibration_screen(self):
        self.epog_analyzer = EPOGAnalyzer(epog_test_error_dir, sys.argv)
        self.epog_analyzer.perform_calibration()

        self.run_test()

    def run_test(self):
        self.video_processor = Video_processor(self.epog_analyzer)
        self.video_processor.play_and_record_video(video_paths, recorded_video_path)

        self.process_data()

    def process_data(self):
        self.processing_window = QtWidgets.QWidget()
        self.processing_window.setWindowTitle("Data processing")
        self.processing_window.setGeometry(100, 100, 300, 200)  # Restore original window size

        self.processing_label = QtWidgets.QLabel("Thanks for passing, now we are processing your data")
        self.processing_label.setObjectName("recommendationLabel")
        self.processing_label.setAlignment(QtCore.Qt.AlignLeft)
        self.processing_label.setFont(QtGui.QFont("Arial", 16))
        self.processing_window.setLayout(QtWidgets.QVBoxLayout())
        self.processing_window.layout().addWidget(self.processing_label)
        self.processing_window.show()

        self.data_processed()

    def data_processed(self):
        self.processing_label.setText("Thanks for passing, now we are processing your data")
        self.processing_label.setText(recommendations)

        self.finish_button = QtWidgets.QPushButton("Back to home screen")
        self.processing_label.setWordWrap(True)
        self.finish_button.setEnabled(False)

        self.processing_window.layout().addWidget(self.finish_button)
        self.processing_window.show()

        QtCore.QTimer.singleShot(10, self.process_data_in_background)
        self.finish_button.clicked.connect(self.finish_test)

    def process_data_in_background(self):
        gaze_df = self.video_processor.estimate_gaze_from_video(recorded_video_path, fps=30)
        gaze_csv_path = recorded_video_path.replace('.mov', '_gaze_data.csv')
        gaze_df.to_csv(gaze_csv_path, index=False)

        annotated_output_path = recorded_video_path.replace('.mov', '_annotated.mov')
        self.video_processor.create_annotated_video(video_paths, gaze_df, annotated_output_path)

        self.finish_button.setEnabled(True)

    def finish_test(self):
        # Reset the UI to the initial state
        self.processing_window.close()
        self.current_question_index = 0  # Reset question index
        self.name_input.clear()  # Clear the name input field

        # Show the name input screen again
        self.name_label.show()
        self.name_input.show()
        self.start_button.show()

        # Clear any remaining test widgets
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    apply_stylesheet(app, theme='light_blue.xml')
    app.setStyleSheet(app.styleSheet() + custom_stylesheet)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())