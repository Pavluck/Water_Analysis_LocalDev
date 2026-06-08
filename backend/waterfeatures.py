"""
☆~ NPavelek ~☆
Show CNN features extracted from video frames 
takes the processed video data, runs it through a CNN model, and visualizes the extracted features for analysis.
Display feature map and kernels 
"""
# ~~~~ Necessary Imports ~~~~
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from testvideostream import load_water_model, test_transforms

# ~~~~ Global Functions ~~~~~
Directory = Path(__file__).resolve().parent
"""
ensures the scripts can find own directory 
regardless of from where the command is executed.
"""
if str(Directory) not in sys.path:
  # safegaurd to prevent duplicate entries in path
  sys.path.insert(0, str(Directory)

# ~~~~ Class Functions ~~~~~                 
class Visuals:
  """Loads the CNN, hooks to a layer to see different levels and extract visual features"""

  def __init__(self, model:nn.Module):
    """Initializes the CNN to be ready for feature exraction"""
    self.model = model
    self.features = None
    self.hook = None
    
  # hook setup (features, registers, removal) 
  def hook_register(self, layer_name:str):
    """
    Hook a layer to see different levels of feature extraction
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    """
    for name, module in self.model.named_modules():
      if name == layer_name:
        self.hook = module.register_forward_hook(self._hook_fn)
        print(f"Hooked to layer: {layer_name}")
        return
    raise ValueError(f"Error: {layer_name} not found")
  
  def hook_remove(self):
    """Frees the model of the hook"""
    if self.hook is not None:
      self.hook.remove()
      self.hook = None
    
  # ~~~~ ✧ CNN Visuals ✧ ~~~~
  def plot_features(self, max_features: int=64, cmap:str = 'viridis'):
    """ 
    Viridis highlights different patterns and activations in the CNN features map
    To quote: "Pretty, oh so pretty"
    Reference: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    if self.features is None:
      return
    # ~~~ Setup ~~~
    num_features = self.features.shape[1]
    num_visualize = min(num_features, max_features)
    grid_size = int(np.ceil(np.sqrt(num_visualize)))
    plt.feature(figsize=(grid_size*2, grid_size*2))
    # ~~~ Plot loop (plot features one by one) ~~~
    for i in range(num_visualize):
      plt.subplot(grid_size, gride_size, i+1)
      # use imshow to display data as a 2D image
      plt.imshow(self.features[0,i],cmap=cmap)
      plt.axis('off')
    plt.subtitle("Water Analysis ~✧~ Feature Map")
    plt.tight_layout()
    plt.show()
    
    
  def plot_kernels(model: nn.Module, max_kernels: int=16):
    """
    Each kernel is visualized as an image, where the intensity of each pixel corresponds to the weight value of that kernel. 
    Reference: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
    """
    # ~~~ sanity check ~~~
    if not hasattr(model, 'backbone'):
      return
    conv_layer = getattr(model.backbone, 'conv1', None)
    if conv_layer is None:
      return
    weights = conv_layer.weight.detach().cpu().numpy()
    num_kernels = min)weights.shape[0], max_kernels)
    grid_size = int(np.ceil(np.sqrt(num_kernels)))
    plt.figure(figsize=(grid_size * 2, grid_size * 2))
    for i in range(num_kernels):
      kernel = weights[i]
      # extract color channel from the final dimension
      kernel = np.transpose(kernel, (1, 2, 0))
      # normalize kernel value
      kernel = (kernel - kernel.min()) / kernel.max() - kernel.min() + 1e-9)
      plt.subplot(grid_size, grid_size, i + 1)
      plt.imshow(kernel)
      plt.axis('off')
      plt.tight_layout()
      plt.show

  
  def preprocess_frame(frame, transform):
    """
    Given a BGR frame returns a normalized tensor batch
    """
    # CNNS expect 4D tensor (Size, Channels, Height, Width)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 0th dimension to batch/stack the image
    return transform(image).unsqueeze(0)

  # Visual CNN features from video/file
  def run_stream(model: nn.Module, layer_name: str = 'backbone.layer1', source=0, max_frames: int =1):
    """
    Hook to file
    """
    feature_visualizer = FeatureVisualizer(model)
    feature_visualizer.register_hook(layer_name)
    cap = sv2.VideoCapture(source)
    # sanity check
    if not cap.isOpened():
      raise RuntimeError(f"Oopsie Daisy~ Unable to open video source: {source}")
    device = next(model.parameters).device
    frame_index = 0
    try:
      while True:
        ret, frame = cap.read()
        if not ret:
          print("End of stream, or cannot read frame.")
          break
        frame_index += 1
        key = None
        try:
          cv2.imshow("Video Stream", frame)
        except cv2.error:
          print("Problems, exiting..")
        # Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
        tensor = preprocess_frame(frame, test_transforms).to(device)
        with torch.no_grad():
          model(tensor)
        feature_visualizer.plot_features()
    finally:
      cap.release()
      cv2.destroyAllWindows()
      except cv2.error:
        pass
      feature_visualizer.remove_hook()

def parse_source(source: str):
  """
  Translates the camera index, file path, or stream URL into an integer
  """
  try:
    return int(source)
  except ValueError:
    return source

# Helper to automate visuals for multiple inputs 
def visualize_suite(model, layer):
  """
  Takes in multiple URLs for testing and visualization
  Helps the class function with hooks and cleanup
  """
  visualizer = Visuals(model)
  visualizer.hook_register(layer)
  # speedup helper
  device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")

  # Kernel Visuals
  for url_data in TEST_URLS:    # from testvideostream.py
    url = url_daya['url']
    # download, extract a few frames for the features
    # TODO: I got to find a better way (than downloading) for feature extraction without violating website security measures...
    frames = extract_video_frames(video_path, max_frames=10)
    for i, pil_img in enumerate(frames):
      # Convert the PIL to Tensor
      # Converts the image to a tensor [C, H, W], add batch dim
      tensor = test_transforms(pil_img).unsqueeze(0).to(device) 
      # Forward Pass to trigger the hook
      with torch.no_grad():
        model(tensor)
      # plot features
      print(f"Features for: {data_data['label']} - Frame {i+1}")
      visualizer.plot_features()
      # clean up
      if video_path.exists():
        video_path.unlink()
  visualizer.remove_hook()

# main function, loads and visualizes
if __name__ == "__main__":
  model = load_water_model()
  if model:
    visualize_suite(model, 'backbone.layer1')
    visualize_suite(model, 'backbone.layer4')   # maybe update this funt. for all layers
  if model is None:
    # error handling
    print("The model was not able to load, Exiting..")
    sys.exit(1)  # break here instead of proceeding
    plot_kernels(model, max_kernels=args.kernels)
    run_stream(model, layer_name=args.layer, source=parse_source(args.source))

# ~~~ EOF ~~~
