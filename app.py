import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv
from tensorflow import keras
from denoising import DenoisingAE as dae
from cropping1 import CropLuar as cl
from PIL import Image
from tensorflow import keras
from io import BytesIO

def intro():
    st.write("# Welcome! 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown("""### KoTA 403""")
    
    st.write("Anggota 1 : Aulia Rahmi")
    st.write("Anggota 2 : Azhar Subhan Fauzi")
    st.write("Pembimbing 1 : xx")
    st.write("Pembimbing 2 : xx")

    st.markdown("""### PENINGKATAN KUALITAS""")

def load_model(path_model):
  model = keras.models.load_model(path_model)
  return model

def upload_image():
  st.sidebar.success("Done!")
  st.title('KoTA 403')

  st.header('Step 1: Prepare Data')
  input_image = st.file_uploader("Upload an image", type=['png', 'jpg'])
  
  img = None
  cvImg = None
  check = 0

  if input_image != None:
    st.image(input_image)

    img = Image.open(input_image)
    arrImg = np.array(img)
    cvImg = cv.cvtColor(arrImg, cv.COLOR_RGB2BGR)

    image_size = img.size
    st.text(image_size)
    check = 1
    
  st.header('Step 2: Cropping image')
  if check == 1:
    crop_image = cl.getLinesAndCrop(cvImg)
    st.image(crop_image)
    
    #cv.imencode('.jpg', crop_image)

  st.header('Step 3: Deskewing image')

  st.header('Step 4: Denoising image')

  dict = {'Default': 'model_denoising/model_780550_20230508_v4.h5', 
      'P01': 'model_denoising/model_1280904_20230425_P01.h5'}
  select_model = st.selectbox('Select a model', dict)
  st.write('You selected:', select_model)

  Autoencoder = load_model(dict[select_model])
  img_AE = keras.utils.load_img(input_image, color_mode="grayscale")
  result = dae.UseModel(Autoencoder, img_AE)
  st.image(result)

def take_picture():
  st.sidebar.success("Done!")
  st.title('KoTA 403')

  st.header('Step 1: Prepare Data')
  input_image = st.camera_input("Take a picture")

  img = None
  cvImg = None
  check = 0

  if input_image != None:
    st.image(input_image)

    #img = keras.utils.load_img(input_image)
    img = Image.open(input_image)
    arrImg = np.array(img)
    cvImg = cv.cvtColor(arrImg, cv.COLOR_RGB2BGR)

    image_size = img.size
    st.text(image_size)
    check = 1
    
  st.header('Step 2: Cropping image')
  if check == 1:
    crop_image = cl.getLinesAndCrop(cvImg)
    st.image(crop_image)

  st.header('Step 3: Deskewing image')

  # DENOISING
  st.header('Step 4: Denoising image')

  dict = {'Default': 'model_denoising\model_1280904_20230425_P01.h5', 
      'P01': 'model_denoising\model_1280904_20230425_P01.h5', 
      'Class': 'First'}
  select_model = st.selectbox('Select a model', dict)
  st.write('You selected:', select_model)



  if dict['Default']:
    opt = keras.optimizers.Adam(learning_rate=0.01)
    loss='mse'
  if dict['P01']:
    opt = keras.optimizers.Adam(learning_rate=0.01)
    loss='mse'@st.cache(allow_output_mutation=True)
    


  

  result = dae.UseModel(dict[select_model], input_image, opt, loss)
  st.text(result.type())

page_names_to_funcs = {
    "—": intro,
    "With upload an image": upload_image,
    "With take a picture": take_picture
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()





# -------------------------------------------------------------
# st.title('KoTA 403')

# st.header('Step 1: Prepare Data')

# col1, col2 = st.columns(2)

# dt = 0
# with col1:
#   if st.button('Upload an image'):
#     dt = 1

# with col2:
#   if st.button('Take a picture'):
#     dt = 2

# if dt == 1: 
#   input_image = st.file_uploader("Upload an image", type=['png', 'jpg'])
# elif dt == 2:
#   input_image = st.camera_input("Take a picture")
# else:
#   input_image = None

# if input_image != None:
#   st.image(input_image)

#   img = keras.utils.load_img(input_image)
#   image_size = img.size
#   st.text(image_size)
  
# st.header('Step 2: Cropping image')

# st.header('Step 3: Deskewing image')

# st.header('Step 4: Denoising image')

# dict = {'Default': 'model_denoising\model_1280904_20230425_P01.h5', 
#      'P01': 'model_denoising\model_1280904_20230425_P01.h5', 
#      'Class': 'First'}
# select_model = st.selectbox('Select a model', dict)
# st.write('You selected:', select_model)

# if dict['Default']:
#   opt = keras.optimizers.Adam(learning_rate=0.01)
#   loss='mse'
# if dict['P01']:
#   opt = keras.optimizers.Adam(learning_rate=0.01)
#   loss='mse'

#result = dae.UseModel(dict[select_model], input_image, opt, loss)
#st.text(result.type())