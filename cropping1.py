import cv2
import numpy as np
import math

class CropLuar():
  # TOO MANY - DEFINE ENERGY (ENERGY = MEAN_DISTANCE / SQRT(LENGTH +1)
  def myCandidates(imIn, lines):
    imTmp = imIn.copy() 
    imTmp = 255 - imTmp # inverse the imTmp
    dist = cv2.distanceTransform(imTmp, cv2.DIST_L2, 3)
    
    imLine = imIn.copy()
    candidates = []
    for x in range(len(lines)):
      for rho,theta in lines[x]:
        imLine[:][:] = 0
        distTmp = dist.copy()
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(imLine,(x1,y1),(x2,y2),255,1)
            
        imLine[dist>3]=0
        distTmp[imLine==0]=0
            
        length = (imLine!=0).sum()
        meanDist = distTmp.sum()/(imLine!=0).sum()
        energy = meanDist / math.sqrt(length + 1)
        candidates.append([rho, theta, energy])
    return candidates
  
  # Sort the candidate lines by their energies in ascending order
  def takeThird(elem):
    return elem[2]
  
  def takeX(elem):
    return (elem[0][0] + elem[1][0]) / 2
  
  def takeY(elem):
    return (elem[0][1] + elem[1][1]) /2
  
  def input_img(data):
    img = data
    orig_img = img.copy()

    # resize for speed
    img = cv2.resize(img,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)
    return orig_img, img

  def preprocess_img(path_img):
    orig_img, img = CropLuar.input_img(path_img)
    # CONVERT TO LAB AND SPLIT
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel,a_channel,b_channel = cv2.split(lab_image)

    # CLOSING CHANNEL - BEFORE AND AFTER
    kernel = np.ones((2,2), np.uint8)
    img_l = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((3,3), np.uint8)
    img_a = cv2.erode(a_channel, kernel, iterations = 1)

    img_b = b_channel

    # GRADIENT ON 3 CHANNEL, SUM THEM UP
    kernel = np.ones((3,3), np.uint8)
    imgra_l = cv2.morphologyEx(img_l, cv2.MORPH_GRADIENT, kernel)
    imgra_a = cv2.morphologyEx(img_a, cv2.MORPH_GRADIENT, kernel)
    imgra_b = cv2.morphologyEx(img_b, cv2.MORPH_GRADIENT, kernel)
    imgra = imgra_l + imgra_a + imgra_b

    # TRIANGLE THRESHOLD
    ret, thresh = cv2.threshold(imgra,0,255,cv2.THRESH_TRIANGLE)

    # BLACK AREAS CLASSIFIED INTO SEVERAL CONNECTED AREA, DEFINE MARKER FROM THAT
    marker = cv2.bitwise_not(thresh)
    kernel = np.ones((3,3), np.uint8)
    marker = cv2.erode(marker, kernel)
    ret, marker = cv2.connectedComponents(marker)

    # WATERSHED
    imgra = cv2.merge((imgra_l,imgra_a,imgra_b))
    imws = cv2.watershed(imgra,marker)

    # CONVERT WATERSHED
    imws8 = np.uint8(imws)

    # Set the boundaries to 255 and the other regions to 0. 
    lenx, leny = imws.shape
    imws8[imws>=0] = 0
    imws8[imws==-1] = 255

    # Since the global edge of the watershed image is set as boundaries, we have to delete its four boundaries.
    for i in range(lenx):
        imws8[i][0] = 0
        imws8[i][leny-1] = 0
    for j in range(leny):
        imws8[0][j] = 0
        imws8[lenx-1][j] = 0

    # HOUGHLINE
    imgcopy = img.copy()
    lines = cv2.HoughLines(imws8,1,np.pi/180,10)

    for x in range(0,len(lines)):
      for rho,theta in lines[x]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(imgcopy,(x1,y1),(x2,y2),(0,0,255),1)

    # Calculate the energy of each candidate line and obtain a list [rho, theta, energy]
    candidateLines = CropLuar.myCandidates(imws8, lines)

    # Divide the lines into vertical lines and horizontal lines
    verticalLines = []
    horizontalLines = []
    for x in candidateLines:
      if (x[1]>=np.pi/4 and x[1]<3*np.pi/4):
        horizontalLines.append(x)
      else:
        verticalLines.append(x)
    
    verticalLines.sort(key=CropLuar.takeThird)
    horizontalLines.sort(key=CropLuar.takeThird)
    
    return img, verticalLines, horizontalLines
  
  def crop_and_skew(image, points):
    # Convert points to numpy array
    points = np.array(points, dtype=np.float32)
    
    # Define the dimensions of the output image
    max_width = max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
    max_height = max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
    
    # Create the output points
    output_points = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype=np.float32)
    
    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(points, output_points)
    
    # Apply the perspective transform
    output_image = cv2.warpPerspective(image, matrix, (int(max_width), int(max_height)))
    
    return output_image

  def polar_dot_to_cartesian_line(candidateLines):
    lines=[]
    for rho, theta, _ in candidateLines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        lines.append([[x1, y1], [x2, y2]])
    return lines

  def interpolate_y(carte_line, target_x):
      x1 = carte_line[0][0]
      y1 = carte_line[0][1]
      x2 = carte_line[1][0]
      y2 = carte_line[1][1]
      if x1 != x2:
          y = y1 + ((target_x - x1) * (y2 - y1)) / (x2 - x1)
          return y
      else:
          return None

  def interpolate_x(carte_line, target_y):
      x1 = carte_line[0][0]
      y1 = carte_line[0][1]
      x2 = carte_line[1][0]
      y2 = carte_line[1][1]
      if y1 != y2:
          x = x1 + ((target_y - y1) * (x2 - x1)) / (y2 - y1)
          return x
      else:
          return None

  def verify_temp_res(temp_res, img):
    if(temp_res[0] != None and temp_res[1] != None):
        if(temp_res[0] >= 0 and temp_res[0] <= img.shape[1] and temp_res[1] >=0 and temp_res[1] <= img.shape[0]):
          return True
    return False

  def convert_lines_in_image(carte_lines, img):
    res = []

    for carte_line in carte_lines:
      dotone = []
      dottwo = []

      x = img.shape[1]
      temp_res1 = [x, CropLuar.interpolate_y(carte_line, x)]
      if(CropLuar.verify_temp_res(temp_res1, img)):
        if(dotone==[]):
          dotone = temp_res1
        elif(dottwo==[]):
          dottwo = temp_res1
      
      x = 0
      temp_res2 = [x, CropLuar.interpolate_y(carte_line, x)]
      if(CropLuar.verify_temp_res(temp_res2, img)):
        if(dotone==[]):
          dotone = temp_res2
        elif(dottwo==[]):
          dottwo = temp_res2
      
      y = img.shape[0]
      temp_res3 = [CropLuar.interpolate_x(carte_line, y), y]
      if(CropLuar.verify_temp_res(temp_res3, img)):
        if(dotone==[]):
          dotone = temp_res3
        elif(dottwo==[]):
          dottwo = temp_res3
      
      y = 0
      temp_res4 = [CropLuar.interpolate_x(carte_line, y), y]
      if(CropLuar.verify_temp_res(temp_res4, img)):
        if(dotone==[]):
          dotone = temp_res4
        elif(dottwo==[]):
          dottwo = temp_res4

      if(dotone!=[] and dottwo!=[]):
        res.append([dotone, dottwo])
        
    return res

  def vertical_divider(carte_vert_filtered, img):
    carte_left = []
    carte_right = []
    for carte_vert in carte_vert_filtered:
      center = [img.shape[1]/2, img.shape[0]/2]
      x1 = carte_vert[0][0]
      y1 = carte_vert[0][1]
      x2 = carte_vert[1][0]
      y2 = carte_vert[1][1]
      if(((x1 + x2)/2)<center[0]):
        carte_left.append(carte_vert)
      else:
        carte_right.append(carte_vert)
    return carte_left, carte_right
  
  def horizontal_divider(carte_vert_filtered, img):
    carte_top = []
    carte_bot = []
    for carte_vert in carte_vert_filtered:
      center = [img.shape[1]/2, img.shape[0]/2]
      x1 = carte_vert[0][0]
      y1 = carte_vert[0][1]
      x2 = carte_vert[1][0]
      y2 = carte_vert[1][1]
      if(((y1 + y2)/2)<center[1]):
        carte_top.append(carte_vert)
      else:
        carte_bot.append(carte_vert)
    return carte_top, carte_bot

  def find_intersection_points_exper(lines1, lines2, h, w, toZero = True, vertical = True):
      intersection_points = []
      max_intersections = 0
      line_with_max_intersections = None
      same_max_intersections = []
      retval = None
      count_max_intersections = 0
      for line1 in lines1:
          intersections = 0

          for line2 in lines2:
              x1, y1 = line1[0]
              x2, y2 = line1[1]
              x3, y3 = line2[0]
              x4, y4 = line2[1]

              denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

              if denominator != 0:
                  x = (((x1 * y2 - y1 * x2) * (x3 - x4)) - ((x1 - x2) * (x3 * y4 - y3 * x4))) / denominator
                  y = (((x1 * y2 - y1 * x2) * (y3 - y4)) - ((y1 - y2) * (x3 * y4 - y3 * x4))) / denominator
                  if(((x <= w) and (x >= 0)) and ((y <= h) and (y >= 0))):
                    intersection_points.append([x, y])
                    intersections += 1
          if intersections > max_intersections:
            max_intersections = intersections
            line_with_max_intersections = line1
            same_max_intersections = []
            count_max_intersections = 0
          if intersections == max_intersections:
            count_max_intersections += 1
            same_max_intersections.append(line1)
          else:
            line_with_max_intersections = line1
      
      if(vertical):
        same_max_intersections.sort(key=CropLuar.takeX)
      else:
        same_max_intersections.sort(key=CropLuar.takeY)

      if(toZero):
        retval = same_max_intersections[0]
      else:
        retval = same_max_intersections[-1]
      return same_max_intersections[0], intersection_points

  def find_intersection_points_from_line(line1, line2):
      x1, y1 = line1[0]
      x2, y2 = line1[1]
      x3, y3 = line2[0]
      x4, y4 = line2[1]
      denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
      if denominator != 0:
        x = (((x1 * y2 - y1 * x2) * (x3 - x4)) - ((x1 - x2) * (x3 * y4 - y3 * x4))) / denominator
        y = (((x1 * y2 - y1 * x2) * (y3 - y4)) - ((y1 - y2) * (x3 * y4 - y3 * x4))) / denominator

        return [x, y]

      return None

  def timesten(best_point):
    return best_point[0]*10, best_point[1]*10

  def getLinesAndCrop(path_img):
    # vert hori
    img, verticalLines, horizontalLines = CropLuar.preprocess_img(path_img)
    carte_vert = CropLuar.polar_dot_to_cartesian_line(verticalLines[0:10])
    carte_horiz = CropLuar.polar_dot_to_cartesian_line(horizontalLines[0:10])

    carte_vert_filtered = CropLuar.convert_lines_in_image(carte_vert, img)
    carte_horiz_filtered = CropLuar.convert_lines_in_image(carte_horiz, img)

    # kiri kanan
    carte_left, carte_right = CropLuar.vertical_divider(carte_vert_filtered, img)
    carte_left.append([[0.0 , 0.0], [0.0, img.shape[0]]])
    carte_right.append([[img.shape[1] , 0.0], [img.shape[1], img.shape[0]]])

    orig_img, img_skip = CropLuar.input_img(path_img)
    img1 = cv2.resize(orig_img,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)

    for carte in carte_left:
      x1 = int(carte[0][0])
      y1 = int(carte[0][1])
      x2 = int(carte[1][0])
      y2 = int(carte[1][1])
      cv2.line(img1,(x1,y1),(x2,y2),(255, 0, 0), 1)

    for carte in carte_right:
      x1 = int(carte[0][0])
      y1 = int(carte[0][1])
      x2 = int(carte[1][0])
      y2 = int(carte[1][1])
      cv2.line(img1,(x1,y1),(x2,y2),(0, 0, 255), 1)

    # atas bawah
    carte_top, carte_bot = CropLuar.horizontal_divider(carte_horiz_filtered, img)
    carte_top.append([[0.0 , 0.0], [img.shape[1], 0.0]])
    carte_bot.append([[0.0 , img.shape[0]], [img.shape[1], img.shape[0]]])

    img2 = cv2.resize(orig_img,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)
    for carte in carte_top:
      x1 = int(carte[0][0])
      y1 = int(carte[0][1])
      x2 = int(carte[1][0])
      y2 = int(carte[1][1])
      cv2.line(img2,(x1,y1),(x2,y2),(255, 255, 0), 1)

    for carte in carte_bot:
      x1 = int(carte[0][0])
      y1 = int(carte[0][1])
      x2 = int(carte[1][0])
      y2 = int(carte[1][1])
      cv2.line(img2,(x1,y1),(x2,y2),(0, 255, 255), 1)
    
    img3 = cv2.resize(orig_img,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)
    for carte in carte_horiz_filtered:
      x1 = int(carte[0][0])
      y1 = int(carte[0][1])
      x2 = int(carte[1][0])
      y2 = int(carte[1][1])
      cv2.line(img3,(x1,y1),(x2,y2),(255, 0, 0), 1)

    for carte in carte_vert_filtered:
      x1 = int(carte[0][0])
      y1 = int(carte[0][1])
      x2 = int(carte[1][0])
      y2 = int(carte[1][1])
      cv2.line(img3,(x1,y1),(x2,y2),(0, 0, 255), 1)
    
    carte_top_filtered, titik_potong = CropLuar.find_intersection_points_exper(carte_top, carte_vert_filtered, img.shape[0], img.shape[1], True, False)
    carte_bot_filtered, titik_potong = CropLuar.find_intersection_points_exper(carte_bot, carte_vert_filtered, img.shape[0], img.shape[1], False, False)
    carte_left_filtered, titik_potong = CropLuar.find_intersection_points_exper(carte_left, carte_horiz_filtered, img.shape[0], img.shape[1], True, True)
    carte_right_filtered, titik_potong = CropLuar.find_intersection_points_exper(carte_right, carte_horiz_filtered, img.shape[0], img.shape[1], False, True)

    best_kiri_atas = CropLuar.find_intersection_points_from_line(carte_top_filtered, carte_left_filtered)
    best_kiri_bawah = CropLuar.find_intersection_points_from_line(carte_bot_filtered, carte_left_filtered)
    best_kanan_atas = CropLuar.find_intersection_points_from_line(carte_top_filtered, carte_right_filtered)
    best_kanan_bawah = CropLuar.find_intersection_points_from_line(carte_bot_filtered, carte_right_filtered)

    imgres = cv2.resize(orig_img,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)

    best_kiri_atas = CropLuar.timesten(best_kiri_atas)
    best_kiri_bawah = CropLuar.timesten(best_kiri_bawah)
    best_kanan_atas = CropLuar.timesten(best_kanan_atas)
    best_kanan_bawah = CropLuar.timesten(best_kanan_bawah)

    points = [tuple(best_kiri_atas), tuple(best_kanan_atas), tuple(best_kanan_bawah), tuple(best_kiri_bawah)]
    output_image = CropLuar.crop_and_skew(orig_img.copy(), points)
    return output_image