import os

def ensure_directories_exist(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)