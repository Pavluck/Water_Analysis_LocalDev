"""
✧ Natasha Pavelek
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
import csv
import tempfile
from pathlib import Path
import torch
from torchvision import transforms
import requests
import yt_dlp
# ~~~ Import CNN from Same Folder ~~~
import sys
sys.path.insert(0, str(Filepath))
try:
    from CNNtraining import WaterCNN
except ImportError:
    print("Oppsie Daisy~ Could not import the model")
    exit(1)

# ~~~ Test 0 ~~~
# Test with live stream video URL, make sure it can recieve a livestream
test_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
processedvid = VideoProcessor(target_size=(224, 224), fps=30)
frames = processedvid.process_video(test_url)
print("Testing... Processed frames shape:", frames.shape if frames is not None else "Test failed: No frames processed"
# Test should succeed with output: 
# Testing...Processed frames shape: (14315, 224, 224, 3)

# ~~~ Test Setup ~~~
Filepath = Path(__file__).parent
CNN_Name = "water_potability_image_model.pth"      # version 2.1
CNNPath = Filepath / CNN_Name
CSVName = "water_potability_test_results.csv"
CSV = Filepath / CSVName
FrameMax = 10   # Process 10 for now TODO, update for it to process every 2 seconds or so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ~~~ Test 1 & 2 ~~~
"""Test URLS, in form {url, label, description}, {url2, label2, description2}, ..."""
TEST_URLS = [
  {
        "url": "https://youtu.be/SgEQrUIKJ6Y?si=eZL4UjAH-92QUy2W",
        "label": "NON-POTABLE",
        "description": "Test 1: Non-potable water with possible algae."
  }, {
        "url": "https://youtu.be/DPCMG7C5OLE?si=YL6dh1V8z0OONBA0",
        "label": "POTABLE",
        "description": "Test 2: River water, clear, natural. Test with waterfall"
  }
]
