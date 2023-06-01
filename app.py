import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import os
from tensorflow import keras
from PIL import Image
import cropping1 as cr
from deskewing import Deskew as dsk

def intro():
    st.write("# Welcome! 👋")

    st.markdown("""### KoTA 403""")
    
    st.write("Anggota 1 : Aulia Rahmi")
    st.write("Anggota 2 : Azhar Subhan Fauzi")
    st.write("Pembimbing 1 : Yudi Widhiyasana, S.Si., M.T.")
    st.write("Pembimbing 2 : Dr. Nurjannah Syakrani, DRA., M.T.")

    st.markdown("""### PENINGKATAN KUALITAS CITRA DOKUMEN DIGITAL""")
    st.markdown("""### MENGGUNAKAN AUTOENCODER""")
    st.markdown("""### PADA APLIKASI SIMA PT LEN INDUSTRI""")

def load_model(path_model):
  model = keras.models.load_model(path_model, compile=False)
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
  return model

def cvToImage(img_cv_type):
  # Balikin ke gambar lagi (bukan bentuk array)
  data = Image.fromarray(img_cv_type)
  path_temp = 'temp/temp.jpg'
  data.save(path_temp)
  return path_temp

def upload_image():
  st.sidebar.success("Done!")

  st.header('Step 1: Prepare Data')
  input_image = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
  path1 = None
  path2 = None
  after_denoise = False

  if input_image != None:
    st.image(input_image)
    img = Image.open(input_image) # buka image pakai PIL
    st.text(img.size)

  # Cropping ------------------------------
  st.header('Step 2: Cropping image')
  if input_image != None:
    # convert image buat buka di openCV
    arrImg = np.array(img)
    cvImg = cv.cvtColor(arrImg, cv.COLOR_RGB2BGR)

    # proses crop
    hough_image,drawed_img, drawed_img_lrtb, imgres, real = cr.crop_main(cvImg)
    colcrop1, colcrop2, colcrop3 = st.columns(3)
    colcrop1.image(drawed_img, caption = 'All Line')
    colcrop2.image(drawed_img_lrtb, caption = 'Divided Line')
    colcrop3.image(imgres, caption = 'Before Cropping')
    st.image(real, caption = 'After Cropping')
    path1 = cvToImage(real)
    # st.image(Image.open(path1))
    st.text(Image.open(path1).size)
  else:
    st.write("!!!Please upload an image!!!")

  # Deskewing ------------------------------
  st.header('Step 3: Deskewing image')

  if path1 != None:
    angle, deskew_image = dsk.correct_skew(real, 0.01, 10, 0.1)
    path2 = cvToImage(deskew_image)
    st.write("Best Angle: ", angle)
    st.image(path2)
  else:
    st.write("!!!Please upload an image!!!")

  # Denoising ------------------------------
  st.header('Step 4: Denoising image')

  if path2 != None:
    temp_image = Image.open(path2)
    resized_image = temp_image.resize((1240,1754))
    path_resize = 'temp/resize_image.jpg'
    resized_image.save(path_resize)

    Autoencoder = load_model('model_denoising/model_potrait_17541240_15_4.h5')
    img_AE = keras.utils.load_img(path_resize, color_mode="grayscale")

    # convert image to numpy array
    images = keras.utils.img_to_array(img_AE, dtype='float32')/255

    # expand dimension of image
    images = np.expand_dims(images, axis=0)

    prediction = Autoencoder.predict(images)
    st.image(prediction)

    # Mengambil gambar asli dari hasil prediksi
    image_array = keras.preprocessing.image.array_to_img(prediction[0])
    image_array.save('temp/predict.jpg')

    after_denoise = True
  else:
    st.write("!!!Please upload an image!!!")
  
  st.header('Step 5: Evaluate Metrics')
  if after_denoise:
    base_image = st.file_uploader("Upload an target image", type=['png', 'jpg', 'jpeg'])

    if base_image != None:
      base_image_toPIL = Image.open(base_image)
      base_image_toCV = cv.cvtColor(np.array(base_image_toPIL), cv.COLOR_RGB2BGR)

      pred_image_toPIL = Image.open('temp/predict.jpg')
      pred_image_toCV = cv.cvtColor(np.array(pred_image_toPIL), cv.COLOR_RGB2BGR)

      img2_asli = cv.resize(base_image_toCV, (1240,1754))
      img2_prediksi = cv.resize(pred_image_toCV, (1240,1754))

      # Hitung MSE antara gambar asli dan hasil prediksi
      mse = np.mean((img2_asli - img2_prediksi) ** 2)

      col1, col2 = st.columns(2)
      col1.image(img2_asli)
      col2.image(img2_prediksi)

      st.subheader('1. Precision and Recall')

      st.subheader('2. MSE')
      st.write('Hasil MSE:', mse)
      
      st.subheader('3. PSNR')
      # menghitung nilai PSNR
      if mse == 0:
          psnr = 100
      else:
          pixel_max = 255.0
          psnr = 20 * np.log10(pixel_max / np.sqrt(mse))

      # mencetak nilai PSNR
      st.write('PSNR:', psnr)

page_names_to_funcs = {
    "Welcome": intro,
    "Demo": upload_image
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()