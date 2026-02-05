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
