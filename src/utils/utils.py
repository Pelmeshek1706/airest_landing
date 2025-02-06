"""
utils.py

This module provides general utilities, including:
  - a function to retrieve the screen resolution (supports Windows, MacOS, and Linux),
  - and a function to ensure that directories exist.
"""

import os
import sys
import platform

def prepare_directories(directories):
    """
    Create directories if they do not already exist.

    :param directories: list of directory paths to be created.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_screen_size():
    """
    Retrieve the screen resolution in pixels.

    Supports Windows, MacOS, and Linux.

    :return: dictionary with keys 'width' and 'height'.
    """
    try:
        if platform.system() == 'Windows':
            from win32api import GetSystemMetrics
            import win32con
            width_px = GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height_px = GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            return {'width': int(width_px), 'height': int(height_px)}
        elif platform.system() == 'Darwin':
            import AppKit
            screen = AppKit.NSScreen.mainScreen().visibleFrame()
            width_px = screen.size.width
            height_px = screen.size.height
            return {'width': int(width_px), 'height': int(height_px)}
        elif platform.system() == 'Linux':
            import Xlib.display
            resolution = Xlib.display.Display().screen().root.get_geometry()
            return {'width': int(resolution.width), 'height': int(resolution.height)}
    except Exception as e:
        sys.exit("Error obtaining screen size: " + str(e))
