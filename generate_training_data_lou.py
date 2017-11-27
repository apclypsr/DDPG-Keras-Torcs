#outputs array of image

from europilot.screen import Box
from europilot.train import generate_training_data, Config
import numpy as np

class MyConfig(Config):
    # Screen area
    BOX = Box(60, 60, 700, 540)
    # Screen capture fps
    DEFAULT_FPS = 20
