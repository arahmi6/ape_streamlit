from importlib.resources import path
import numpy as np
import pandas as pd

from tensorflow import keras
#from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image

class DenoisingAE:
  def UseModel(model, path_image):
    # convert image to numpy array
    images = keras.utils.img_to_array(path_image, dtype='float32')/255

    # expand dimension of image
    images = np.expand_dims(images, axis=0)

    prediction = model.predict(images)

    return prediction
