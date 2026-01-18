"""
NP ˚❀༉‧ Preprocessing video data for CNN input.
This module handles receiving video input from the frontend, cleaning the video frames, and formatting them into a structure suitable for CNN models.
"""
# ~~~~ Necessary Imports ~~~~
import cv2  # OpenCV (Open Source Computer Vision Library)
import numpy as np
from typing import Optional

class VideoProcessor:
    """Class Function to process and format video input for a CNN model."""
    def __init__(self, target_size: tuple = (224, 224), fps: int = 30):
      """
      Initialize the video input, process by:
          target_size: Target frame dimensions (height, width)
          fps: Target frames per second
      """
      self.target_size = target_size
      self.fps = fps

    def recieve(self, video_path: str) -> Optional[csv.VideoCapture]:
        """This function receives video from the frontend, the csv
        library allows optional live streaming. Requires a video path.
        Returns the captured video as an object"""
        try:
            capture = csv.VideoCapture(video_path)
            if not capture.isOpened():
                raise ValueError(f"Cannot open video from: {video_path}")
            return capture
        except Exception as e:
            print(f"Error loading video: {e}")
            return None
