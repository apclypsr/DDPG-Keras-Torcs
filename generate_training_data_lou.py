#outputs array of image

from europilot.screen import Box
from europilot.train import generate_training_data, Config
import numpy as np

class MyConfig(Config):
    # Screen area
    BOX = Box(60, 60, 220, 220)
    # Screen capture fps
    DEFAULT_FPS = 20

generate_training_data(MyConfig)