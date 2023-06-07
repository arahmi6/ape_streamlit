import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

class Deskew():
  def correct_skew(image, delta=1, limit=5, resize=1):
      def determine_score(arr, angle):
          data = inter.rotate(arr, angle, reshape=False, order=0)
          histogram = np.sum(data, axis=1, dtype=float)
          score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
          return histogram, score
      orig_img = image.copy()
      image = cv2.resize(image, None, fx = resize, fy = resize, interpolation = cv2.INTER_CUBIC)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

      scores = []
      angles = np.arange(-limit, limit + delta, delta)
      for angle in angles:
          histogram, score = determine_score(thresh, angle)
          scores.append(score)

      best_angle = angles[scores.index(max(scores))]

      (h, w) = orig_img.shape[:2]
      center = (w // 2, h // 2)
      M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
      corrected = cv2.warpAffine(orig_img, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)
      # Menghitung koordinat 4 titik sudut
      corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
      transformed_corners = cv2.transform(np.array([corners]), M)[0]
      print("Transformed Corners:",transformed_corners)
      return best_angle, corrected