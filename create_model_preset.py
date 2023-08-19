import os
from tensorflow import keras
from paths import BUILD_PATH
from train import SAVE_MODEL_PATH, build_model

name = input("Name to save model? ")

model = build_model()
model.load_weights(SAVE_MODEL_PATH)
model.save_weights(os.path.join(BUILD_PATH, f"{name}.ckpt"))