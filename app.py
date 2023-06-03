import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import jaccard as jc
import pytesseract

from tensorflow import keras
from PIL import Image
import cropping1 as cr
from deskewing import Deskew as dsk

def intro():
    st.write("# Welcome! 👋")
    st.markdown("""#### PENINGKATAN KUALITAS CITRA DOKUMEN DIGITAL MENGGUNAKAN AUTOENCODER PADA APLIKASI SIMA PT. LEN INDUSTRI""")
  
    st.markdown("""##### KoTA 403""")
    st.write("- Anggota 1 : Aulia Rahmi")
    st.write("- Anggota 2 : Azhar Subhan Fauzi")
    st.write("- Pembimbing 1 : Yudi Widhiyasana, S.Si., M.T.")
    st.write("- Pembimbing 2 : Dr. Nurjannah Syakrani, DRA., M.T.")

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
    algo = st.radio(
    "Select Algorithm",
    ('Select line by its gradient', 'Select line by intersection points (Outermost)'))

    if algo == 'Select line by its gradient':
        hough_image,drawed_img, drawed_img_lrtb, imgres, real = cr.crop_main(cvImg, 0)
    elif algo == 'Select line by intersection points (Outermost)':
        _,_,_,_, real = cr.crop_main(cvImg, 1)
        hough_image,drawed_img, drawed_img_lrtb, imgres, real = cr.crop_main(real, 1)
    
    colcrop1, colcrop2, colcrop3, colcrop4 = st.columns(4)
    colcrop1.image(hough_image, caption = 'Hough Line')
    colcrop2.image(drawed_img, caption = 'All Line')
    colcrop3.image(drawed_img_lrtb, caption = 'Divided Line')
    colcrop4.image(imgres, caption = 'Before Cropping')
    st.image(real, caption = 'After Cropping')
    path1 = cvToImage(real)
    st.text(Image.open(path1).size)
  else:
    st.write(" Please upload the image first! 😄")

  # Deskewing ------------------------------
  st.header('Step 3: Deskewing image')

  if path1 != None:
    delta = st.number_input('Insert a delta', min_value=0.01, max_value=1.0, value=0.1)
    resize = st.number_input('Insert a resize', min_value=0.01, max_value=1.0, value=0.1)
    if delta != 0 and resize != 0:
      angle, deskew_image = dsk.correct_skew(real, delta, 10, resize)
      path2 = cvToImage(deskew_image)
      st.write("Best Angle: ", angle)
      st.image(path2)
  else:
    st.write(" Please upload the image first! 😄")

  # Denoising ------------------------------
  st.header('Step 4: Denoising image')

  if path2 != None:
    temp_image = Image.open(path2)
    check_size = temp_image.size
    if check_size[0] > check_size[1]:
      rotated_image = temp_image.rotate(90, expand=True)
      resized_image = rotated_image.resize((1240,1754))
      path_resize = 'temp/resize_image.jpg'
      resized_image.save(path_resize)

      Autoencoder = load_model('model_denoising/model_potrait_17541240_15_4.h5')
      img_AE = keras.utils.load_img(path_resize, color_mode="grayscale")
      images = keras.utils.img_to_array(img_AE, dtype='float32')/255
      images = np.expand_dims(images, axis=0) # expand dimension of image

      prediction = Autoencoder.predict(images)
      # st.image(prediction)

      # Mengambil gambar asli dari hasil prediksi
      image_array = keras.preprocessing.image.array_to_img(prediction[0])
      image_array.save('temp/predict.jpg')

      result = Image.open('temp/predict.jpg')
      result_rotate = result.rotate(-90, expand=True)
      result_save = result_rotate.save('temp/predict.jpg')
      st.image(result_rotate)

      read_thres = cv.imread('temp/predict.jpg',2)
      ret, thres_img = cv.threshold(read_thres,230,255,cv.THRESH_BINARY)
      st.image(thres_img, caption="Threshold")
      cvToImage(thres_img)

      after_denoise = True
    else:
      resized_image = temp_image.resize((1240,1754))
      path_resize = 'temp/resize_image.jpg'
      resized_image.save(path_resize)

      Autoencoder = load_model('model_denoising/model_potrait_17541240_15_4.h5')
      img_AE = keras.utils.load_img(path_resize, color_mode="grayscale")
      images = keras.utils.img_to_array(img_AE, dtype='float32')/255
      images = np.expand_dims(images, axis=0) # expand dimension of image

      prediction = Autoencoder.predict(images)
      st.image(prediction)

      # Mengambil gambar asli dari hasil prediksi
      image_array = keras.preprocessing.image.array_to_img(prediction[0])
      image_array.save('temp/predict.jpg')

      read_thres = cv.imread('temp/predict.jpg',2)
      ret, thres_img = cv.threshold(read_thres,230,255,cv.THRESH_BINARY)
      st.image(thres_img, caption="Threshold")
      cvToImage(thres_img)

      after_denoise = True
  else:
    st.write(" Please upload the image first! 😄")
  
  st.header('Step 5: Evaluate Metrics')
  if after_denoise:
    base_image = st.file_uploader("Upload an target image", type=['png', 'jpg', 'jpeg'])

    if base_image != None:
      base_image_toPIL = Image.open(base_image)
      base_size = base_image_toPIL.size
      base_image_toCV = cv.cvtColor(np.array(base_image_toPIL), cv.COLOR_RGB2BGR)

      pred_image_toPIL = Image.open('temp/temp.jpg')
      pred_image_toCV = cv.cvtColor(np.array(pred_image_toPIL), cv.COLOR_RGB2BGR)

      if base_size[0] > base_size[1]:
        img2_asli = cv.resize(base_image_toCV, (1754,1240))
        img2_prediksi = cv.resize(pred_image_toCV, (1754,1240))
      else:
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

      st.subheader('4. Jaccard Similarity Index')
      jacc1, jacc2 = st.columns(2)
      pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
      custom_config=r'''--oem 3 --psm 4 -c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .-,/\ 0123456789"'''

      ret, bw_asli = cv.threshold(img2_asli,230,255,cv.THRESH_BINARY)
      text_asli = pytesseract.image_to_string(bw_asli,config=custom_config)
      jacc1.text("Expected")
      jacc1.write(text_asli)

      ret, bw_pred = cv.threshold(img2_prediksi,230,255,cv.THRESH_BINARY)
      text_pred = pytesseract.image_to_string(bw_pred,config=custom_config)
      jacc2.text("Resulted")
      jacc2.write(text_pred)

      calculate = jc.Jaccard_Similarity(text_asli, text_pred)
      st.write("Accuracy Jaccard:", calculate)

  else:
    st.write(" Please upload the image first! 😄")

page_names_to_funcs = {
    "Welcome": intro,
    "Demo": upload_image
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()