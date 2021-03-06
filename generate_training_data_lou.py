#outputs array of image

from europilot.screen import Box
from europilot.train import generate_training_data, generate_one_trainingdata, Config
import numpy as np

class MyConfig(Config):
    # Screen area
    BOX = Box(60, 60, 220, 220)
    # Screen capture fps
    DEFAULT_FPS = 60

for i in range(1000):
    print(i)
    print(generate_one_trainingdata(MyConfig))