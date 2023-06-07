import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import os
import jaccard as jc
import pytesseract

from tensorflow import keras
from PIL import Image
import cropping as cr
from deskewing import Deskew as dsk
import pr

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
  without_denoise = False

  if input_image != None:
    st.image(input_image)
    img = Image.open(input_image) # buka image pakai PIL
    st.text(img.size)

  # Cropping ------------------------------
  st.header('Step 2: Cropping image')
  if input_image != None:
    # convert image buat buka di openCV
    arrImg = np.array(img)
    toBGR = cv.cvtColor(arrImg, cv.COLOR_RGB2BGR)
    cvImg = cv.cvtColor(toBGR, cv.COLOR_BGR2RGB)

    # proses crop
    algo = st.radio(
    "Select Algorithm",
    ('None',
     'Select line by its gradient',
     'Select line by intersection points (Outermost)',
     'Select line by intersection points (Innermost)', 
     'Select line by intersection points (Energy)'))

    if algo == 'None':
       st.image(cvImg, caption = 'Without Cropping')
       real = cvImg
       path1 = cvToImage(real)
    else:
      if algo == 'Select line by its gradient':
          hough_image, drawed_img, drawed_img_lrtb, imgres, real, best_kiri_atas_crop, best_kiri_bawah_crop, best_kanan_atas_crop, best_kanan_bawah_crop = cr.crop_main(cvImg, 0)
          colcrop1, colcrop2, colcrop3, colcrop4 = st.columns(4)
          colcrop1.image(hough_image, caption = 'Hough Line')
          colcrop2.image(drawed_img, caption = 'All Line')
          colcrop3.image(drawed_img_lrtb, caption = 'Divided Line')
          colcrop4.image(imgres, caption = 'Before Cropping')
      elif algo == 'Select line by intersection points (Outermost)':
          hough_image1,drawed_img1, drawed_img_lrtb1, imgres1, real1, best_kiri_atas_crop, best_kiri_bawah_crop, best_kanan_atas_crop, best_kanan_bawah_crop  = cr.crop_main(cvImg, 1)
          hough_image2,drawed_img2, drawed_img_lrtb2, imgres2, real2, best_kiri_atas_crop, best_kiri_bawah_crop, best_kanan_atas_crop, best_kanan_bawah_crop  = cr.crop_main(real1, 1)
      elif algo == 'Select line by intersection points (Innermost)':
          hough_image1,drawed_img1, drawed_img_lrtb1, imgres1, real1, best_kiri_atas_crop, best_kiri_bawah_crop, best_kanan_atas_crop, best_kanan_bawah_crop  = cr.crop_main(cvImg, 1)
          hough_image2,drawed_img2, drawed_img_lrtb2, imgres2, real2, best_kiri_atas_crop, best_kiri_bawah_crop, best_kanan_atas_crop, best_kanan_bawah_crop  = cr.crop_main(real1, 2)
      elif algo == 'Select line by intersection points (Energy)':
          hough_image1,drawed_img1, drawed_img_lrtb1, imgres1, real1, best_kiri_atas_crop, best_kiri_bawah_crop, best_kanan_atas_crop, best_kanan_bawah_crop  = cr.crop_main(cvImg, 1)
          hough_image2,drawed_img2, drawed_img_lrtb2, imgres2, real2, best_kiri_atas_crop, best_kiri_bawah_crop, best_kanan_atas_crop, best_kanan_bawah_crop  = cr.crop_main(real1, 3)
      
      if algo != 'Select line by its gradient':
        st.text("Phase 1")
        colcrop1, colcrop2, colcrop3, colcrop4 = st.columns(4)
        st.text("Phase 2")
        colcrop5, colcrop6, colcrop7, colcrop8 = st.columns(4)
        colcrop1.image(hough_image1, caption = 'Hough Line')
        colcrop2.image(drawed_img1, caption = 'All Line')
        colcrop3.image(drawed_img_lrtb1, caption = 'Divided Line')
        colcrop4.image(imgres1, caption = 'Before Cropping')
        colcrop5.image(hough_image2, caption = 'Hough Line')
        colcrop6.image(drawed_img2, caption = 'All Line')
        colcrop7.image(drawed_img_lrtb2, caption = 'Divided Line')
        colcrop8.image(imgres2, caption = 'Before Cropping')
        real = real2
      st.image(real, caption = 'After Cropping')
      path1 = cvToImage(real)
      st.text(Image.open(path1).size)
      text_name = os.path.splitext(input_image.name)[0]
      gt_crop = {'001': [(0, 0), (real.shape[1], 0), (real.shape[1], real.shape[0]), (0, real.shape[0])],
                 '002': [(0, 0), (real.shape[1], 0), (real.shape[1], real.shape[0]), (0, real.shape[0])],
                 '003': [(0, 320), (real.shape[1], 413), (real.shape[1], 3293), (0, 3233)], 
                 '004': [(0, 723), (1610, 696), (real.shape[1], 3213), (0, 3306)],
                 '005': [(0, 486), (1656, 422), (real.shape[1], 2936), (0, 2983)],
                 '006': [(0, 0), (real.shape[1], 0), (real.shape[1], real.shape[0]), (0, real.shape[0])], 
                 '007': [(116, 52), (1430, 24), (1532, 1992), (28, 2028)],
                 '008': [(82, 42), (1996, 32), (2018, 1440), (50, 1424)],
                 '009': [(0, 0), (real.shape[1], 0), (real.shape[1], real.shape[0]), (0, real.shape[0])],
                 '010': [(0, 0), (real.shape[1], 0), (real.shape[1], real.shape[0]), (0, real.shape[0])]}
      float_coords = [best_kiri_atas_crop, best_kanan_atas_crop, best_kanan_bawah_crop, best_kiri_bawah_crop]
      integer_coords = [(int(x), int(y)) for x, y in float_coords]     
      gt_mask, pred_mask, tp, tn, fp, fn, precision, recall = pr.precrec(cvImg, gt_crop[text_name], integer_coords)
      st.subheader('Precision and Recall')
      colcrop1, colcrop2 = st.columns(2)
      colcrop1.image(gt_mask, caption="Ground Truth")
      colcrop2.image(pred_mask, caption="Prediction")
      st.write('Hasil Precision:', precision*100, '%')
      st.write('Hasil Recall:', recall*100, '%')
  else:
    st.write(" Please upload the image first!")

  # Deskewing ------------------------------
  st.header('Step 3: Deskewing image')
  if path1 != None:
    skew = st.radio(
        "Select Algorithm",
        ('Without Deskewing',
        'With Deskewing'))
    
    if skew == 'Without Deskewing':
      deskew_image = real
      path2 = cvToImage(deskew_image)
      st.image(path2, caption = 'Without Deskewing')
    elif skew == 'With Deskewing':
      delta = st.number_input('Insert a delta', min_value=0.01, max_value=1.0, value=0.1)
      resize = st.number_input('Insert a resize', min_value=0.01, max_value=1.0, value=0.1)
      if delta != 0 and resize != 0:
        angle, deskew_image = dsk.correct_skew(real, delta, 10, resize)
        path2 = cvToImage(deskew_image)
        st.write("Best Angle: ", angle)
        st.image(path2, caption = 'After Deskewing')
  else:
    st.write(" Please upload the image first!")

  # Denoising ------------------------------
  st.header('Step 4: Denoising image')

  list_model = {'Learning Rate, Epoch, Batch Size': None,
                '1e-3, 15, 4': 'model_denoising\model_potrait_17541240_15_4.h5',
                '1e-3, 15, 8': 'model_denoising\model_potrait_17541240_15_8.h5',
                '1e-3, 20, 4': 'model_denoising\model_potrait_17541240_20_4.h5',
                '1e-3, 20, 8': 'model_denoising\model_potrait_17541240_20_8.h5',
                '1e-3, 25, 4': 'model_denoising\model_potrait_17541240_25_4.h5',
                '1e-3, 25, 8': 'model_denoising\model_potrait_17541240_25_8.h5',
                '1e-4, 15, 4': 'model_denoising\model_potrait2_17541240_15_4.h5',
                '1e-4, 15, 8': 'model_denoising\model_potrait2_17541240_15_8.h5',
                '1e-4, 20, 4': 'model_denoising\model_potrait2_17541240_20_4.h5',
                '1e-4, 20, 8': 'model_denoising\model_potrait2_17541240_20_8.h5',
                '1e-4, 25, 4': 'model_denoising\model_potrait2_17541240_25_4.h5',
                '1e-4, 25, 8': 'model_denoising\model_potrait2_17541240_25_8.h5'}

  if path2 != None:
    temp_image = Image.open(path2)
    check_size = temp_image.size
    option = st.selectbox('Choose Model', list_model)
    if option == 'Learning Rate, Epoch, Batch Size':
       st.image(temp_image)
       after_denoise = True
       without_denoise = True
    else:
      st.write('You selected:', option)
      if check_size[0] > check_size[1]:
        rotated_image = temp_image.rotate(90, expand=True)
        resized_image = rotated_image.resize((1240,1754))
        path_resize = 'temp/resize_image.jpg'
        resized_image.save(path_resize)

        Autoencoder = load_model(list_model[option])
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
        # st.image(result_rotate)

        read_thres = cv.imread('temp/predict.jpg',2)
        ret, thres_img = cv.threshold(read_thres,235,255,cv.THRESH_BINARY)
        st.image(thres_img, caption="Threshold")
        cvToImage(thres_img)

        after_denoise = True
      else:
        resized_image = temp_image.resize((1240,1754))
        path_resize = 'temp/resize_image.jpg'
        resized_image.save(path_resize)

        Autoencoder = load_model(list_model[option])
        img_AE = keras.utils.load_img(path_resize, color_mode="grayscale")
        images = keras.utils.img_to_array(img_AE, dtype='float32')/255
        images = np.expand_dims(images, axis=0) # expand dimension of image

        prediction = Autoencoder.predict(images)
        # st.image(prediction)

        # Mengambil gambar asli dari hasil prediksi
        image_array = keras.preprocessing.image.array_to_img(prediction[0])
        image_array.save('temp/predict.jpg')

        read_thres = cv.imread('temp/predict.jpg',2)
        ret, thres_img = cv.threshold(read_thres,235,255,cv.THRESH_BINARY)
        
        st.image(thres_img, caption="Threshold")
        cvToImage(thres_img)

        after_denoise = True
  else:
    st.write(" Please upload the image first!")
  
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
      
      tp2 = 1487480
      fn2 = 800000
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      recall2 = tp2 / (tp2 + fn2)
      
      # Hitung MSE antara gambar asli dan hasil prediksi
      mse = np.mean((img2_asli - img2_prediksi) ** 2)

      col1, col2 = st.columns(2)
      col1.image(img2_asli)
      col2.image(img2_prediksi)

      st.subheader('1. Precision and Recall')
      st.write('Cropping')
      # colcrop1, colcrop2 = st.columns(2)
      # colcrop1.image(gt_mask)
      # colcrop2.image(pred_mask)
      # st.write('Hasil Precision:', precision*100, '%')
      # st.write('Hasil Recall:', recall*100, '%')
      st.write('Deskewing')
      # st.write('Hasil Recall:', recall2*100, '%')

      st.subheader('2. MSE')
      st.write('Hasil MSE:', mse)
      
      st.subheader('3. PSNR')
      # menghitung nilai PSNR
      if mse == 0:
          psnr = 100
      else:
          pixel_max = 255.0
          psnr = 10 * np.log10(pixel_max / mse)

      # mencetak nilai PSNR
      st.write('PSNR:', psnr)

      st.subheader('4. Jaccard Similarity Index')
      #open text file in read mode
      text_name = os.path.splitext(input_image.name)[0]
      text_file = open("ground_truth_jaccard/"+text_name+".txt", "r")
      #read whole file to a string
      data_string = text_file.read()

      jacc1, jacc2 = st.columns(2)
      pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
      custom_config=r'''--oem 3 --psm 4'''

      ret, bw_asli = cv.threshold(img2_asli,230,255,cv.THRESH_BINARY)
      text_asli = pytesseract.image_to_string(bw_asli,config=custom_config)
      jacc1.text("Expected")
      jacc1.write(data_string)

      if without_denoise:
        ret, bw_pred = cv.threshold(pred_image_toCV,230,255,cv.THRESH_BINARY)
        text_pred = pytesseract.image_to_string(bw_pred,config=custom_config)
      else:
        ret, bw_pred = cv.threshold(img2_prediksi,230,255,cv.THRESH_BINARY)
        text_pred = pytesseract.image_to_string(bw_pred,config=custom_config)
      jacc2.text("Resulted")
      jacc2.write(text_pred)

      calculate = jc.Jaccard_Similarity(text_asli, text_pred)
      st.write("Accuracy Jaccard:", calculate)
  else:
    st.write(" Please upload the image first!")

page_names_to_funcs = {
    "Welcome": intro,
    "Demo": upload_image
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
