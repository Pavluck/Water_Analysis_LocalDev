"""
Tests for the Video Data 
Needs to recieve video
Then it needs to Process/Normalize correctly
TODO: Test with direct livestream to a body of water
TODO: Test with CNN model
TODO: Test with another livestream to a body of water
"""

# ~~~ Imports ~~~
from videodata import VideoProcessor
import cv2
import numpy as np
import subprocess

# Test with live stream video URL, make sure it can recieve a livestream
test_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
processedvid = VideoProcessor(target_size=(224, 224), fps=30)
frames = processedvid.process_video(test_url)
print("Testing... Processed frames shape:", frames.shape if frames is not None else "Test failed: No frames processed"
# Test should succeed with output: 
# Testing...Processed frames shape: (14315, 224, 224, 3)
