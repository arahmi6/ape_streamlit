# terluar-terluar

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def resize(img):
    resized_img = cv2.resize(img,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)
    return resized_img

def convert_to_lab(img):
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel,a_channel,b_channel = cv2.split(lab_image)
    return l_channel, a_channel, b_channel, lab_image

def closing_l(l_channel):
    kernel = np.ones((2,2), np.uint8)
    img_l = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, kernel)
    return img_l

def erosion_a(a_channel):
    kernel = np.ones((3,3), np.uint8)
    img_a = cv2.erode(a_channel,kernel,iterations = 1)
    return img_a

def sumlab(img_l, img_a, img_b):
    kernel = np.ones((3,3), np.uint8)
    imgra_l = cv2.morphologyEx(img_l, cv2.MORPH_GRADIENT, kernel)
    imgra_a = cv2.morphologyEx(img_a, cv2.MORPH_GRADIENT, kernel)
    imgra_b = cv2.morphologyEx(img_b, cv2.MORPH_GRADIENT, kernel)
    imgra = imgra_l + imgra_a + imgra_b
    return imgra, imgra_l, imgra_a, imgra_b

def thresh_image(imgra):
    ret, thresh = cv2.threshold(imgra,0,255,cv2.THRESH_TRIANGLE)
    return thresh

def setmarker(thresh):
    marker = cv2.bitwise_not(thresh)
    kernel = np.ones((3,3), np.uint8)
    marker = cv2.erode(marker, kernel)
    ret, marker = cv2.connectedComponents(marker)
    return marker

def watershed_image(imgra_l,imgra_a,imgra_b,marker):
    imgra = cv2.merge((imgra_l,imgra_a,imgra_b))
    imws = cv2.watershed(imgra,marker)
    return imws, imgra

def convert_watershed_uint8(imws):
    imws8 = np.uint8(imws)
    lenx, leny = imws.shape
    imws8[imws>=0] = 0
    imws8[imws==-1] = 255
    for i in range(lenx):
        imws8[i][0] = 0
        imws8[i][leny-1] = 0
    for j in range(leny):
        imws8[0][j] = 0
        imws8[lenx-1][j] = 0
    return imws8

def hough(img, imws8):
    imgcopy = img.copy()
    lines = cv2.HoughLines(imws8,1,np.pi/180,10)
    if lines is None:
       return imgcopy, [[(0.0, 0.0)]]
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
            
    return imgcopy, lines

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

def divide_line_to_horizontal_and_vertical(imws8, lines):
    candidateLines = myCandidates(imws8, lines)
    verticalLines = []
    horizontalLines = []
    for x in candidateLines:
        if (x[1]>=np.pi/4 and x[1]<3*np.pi/4):
            horizontalLines.append(x)
        else:
            verticalLines.append(x)
    return verticalLines[:10], horizontalLines[:10]

def calc_energy(imws8, lines):
  candidateLines = myCandidates(imws8, lines)
  verticalLines = []
  horizontalLines = []
  for x in candidateLines:
      if (x[1]>=np.pi/4 and x[1]<3*np.pi/4):
          horizontalLines.append(x)
      else:
          verticalLines.append(x)

  retvalvert = []
  retvalhor = []  
  verticalLines.sort(key=takeThird)
  horizontalLines.sort(key=takeThird)
  for y in verticalLines[:10]:
    angle_deg = np.degrees(y[1])  # Konversi kemiringan garis ke derajat
    if (angle_deg <= 10 and angle_deg >= -10) or (angle_deg <= 190 and angle_deg >= 170):
      retvalvert.append(y)
  
  for z in horizontalLines[:10]:
    angle_deg = np.degrees(z[1])  # Konversi kemiringan garis ke derajat
    if (angle_deg >= 80 and angle_deg <= 100) or (angle_deg >= 260 and angle_deg <= 280):
      retvalhor.append(z)
  return retvalvert, retvalhor

def draw(imIn, candidateLines, color=(255,0,0), scale_factor=1, thick=1, length=1000):
    for rho, theta, _ in candidateLines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho*scale_factor
        y0 = b*rho*scale_factor
        x1 = int(x0 + length*(-b*scale_factor))
        y1 = int(y0 + length*(a*scale_factor))
        x2 = int(x0 - length*(-b*scale_factor))
        y2 = int(y0 - length*(a*scale_factor))
        cv2.line(imIn,(x1,y1),(x2,y2),color,thick)
    return imIn

def draw_all(img, verticalLines, horizontalLines):
    # Draw the lines
    imgcopy = img.copy()
    for rho, theta, _ in verticalLines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(imgcopy,(x1,y1),(x2,y2),(255,0,0),1)
        
    for rho, theta, _ in horizontalLines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(imgcopy,(x1,y1),(x2,y2),(0,255,0),1)
    return imgcopy

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
    temp_res1 = [x, interpolate_y(carte_line, x)]
    if(verify_temp_res(temp_res1, img)):
      if(dotone==[]):
        dotone = temp_res1
      elif(dottwo==[]):
        dottwo = temp_res1
    
    x = 0
    temp_res2 = [x, interpolate_y(carte_line, x)]
    if(verify_temp_res(temp_res2, img)):
      if(dotone==[]):
        dotone = temp_res2
      elif(dottwo==[]):
        dottwo = temp_res2
    
    y = img.shape[0]
    temp_res3 = [interpolate_x(carte_line, y), y]
    if(verify_temp_res(temp_res3, img)):
      if(dotone==[]):
        dotone = temp_res3
      elif(dottwo==[]):
        dottwo = temp_res3
    
    y = 0
    temp_res4 = [interpolate_x(carte_line, y), y]
    if(verify_temp_res(temp_res4, img)):
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

def handle_left_right(carte_left, carte_right, img):
    if(len(carte_left)==0):
        carte_right=[]
    if(len(carte_right)==0):
        carte_left=[]
    if(len(carte_left)<1):
        carte_left.append([[0.0 , 0.0], [0.0, img.shape[0]]])
    if(len(carte_right)<1):
        carte_right.append([[img.shape[1] , 0.0], [img.shape[1], img.shape[0]]])
    return carte_left, carte_right

def handle_left_right_2(carte_left, carte_right, img):
    carte_left.append([[0.0 , 0.0], [0.0, img.shape[0]]])
    carte_right.append([[img.shape[1] , 0.0], [img.shape[1], img.shape[0]]])
    return carte_left, carte_right

def draw_lr(carte_left, carte_right, img1):
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
    return img1

def handle_top_bot(carte_top, carte_bot, img):
    if(len(carte_top)==0):
        carte_bot=[]
    if(len(carte_bot)==0):
        carte_top=[]
    if(len(carte_top)<1):
        carte_top.append([[0.0 , 0.0], [img.shape[1], 0.0]])
    if(len(carte_bot)<1):
        carte_bot.append([[0.0 , img.shape[0]], [img.shape[1], img.shape[0]]])
    return carte_top, carte_bot

def handle_top_bot_2(carte_top, carte_bot, img):
    carte_top.append([[0.0 , 0.0], [img.shape[1], 0.0]])
    carte_bot.append([[0.0 , img.shape[0]], [img.shape[1], img.shape[0]]])
    return carte_top, carte_bot

def draw_tb(carte_top, carte_bot, img2):
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
    return img2

def draw_rl_tb( carte_left, carte_right, carte_top, carte_bot, img):
    imagedrawed = draw_lr(carte_left, carte_right, img)
    imagedrawed2 = draw_tb(carte_top, carte_bot, imagedrawed)
    return imagedrawed2

def takeX(elem):
  return (elem[0][0] + elem[1][0]) / 2

def takeY(elem):
  return (elem[0][1] + elem[1][1]) / 2 

def find_line_by_intersection_points(lines1, lines2, h, w, toZero = True, vertical = True):
    intersection_points = []
    max_intersections = 0
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
          same_max_intersections = []
          count_max_intersections = 0
        if intersections == max_intersections:
          count_max_intersections += 1
          same_max_intersections.append(line1)
    
    if(vertical):
      same_max_intersections.sort(key=takeX)
    else:
      same_max_intersections.sort(key=takeY)

    if(toZero):
      retval = same_max_intersections[0]
    else:
      retval = same_max_intersections[-1]
    return retval, intersection_points

def find_line_by_intersection_points_terdalam(lines1, lines2, h, w, toZero = True, vertical = True):
    intersection_points = []
    max_intersections = 0
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
          same_max_intersections = []
          count_max_intersections = 0
        if intersections == max_intersections:
          count_max_intersections += 1
          same_max_intersections.append(line1)
    
    if(vertical):
      same_max_intersections.sort(key=takeX)
    else:
      same_max_intersections.sort(key=takeY)

    if(toZero):
      retval = same_max_intersections[-1]
    else:
      retval = same_max_intersections[0]
    return retval, intersection_points

def find_line_by_intersection_points_energy(lines1, lines2, h, w, toZero = True, vertical = True):
    intersection_points = []
    max_intersections = 0
    same_max_intersections = []
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
          same_max_intersections = []
          count_max_intersections = 0
        if intersections == max_intersections:
          count_max_intersections += 1
          same_max_intersections.append(line1)
    
    if(vertical):
      same_max_intersections.sort(key=takeX)
    else:
      same_max_intersections.sort(key=takeY)
    return same_max_intersections[0], intersection_points

def draws(imIn, candidateLines, color=(255,0,0), scale_factor=1, thick=1, length=1000):
    for rho, theta in candidateLines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho*scale_factor
        y0 = b*rho*scale_factor
        x1 = int(x0 + length*(-b*scale_factor))
        y1 = int(y0 + length*(a*scale_factor))
        x2 = int(x0 - length*(-b*scale_factor))
        y2 = int(y0 - length*(a*scale_factor))
        cv2.line(imIn,(x1,y1),(x2,y2),color,thick)
    return imIn

def find_gradient_difference(lines):
    gradients = []
    line_gradients = []

    for line in lines:
        x1, y1 = line[0]
        x2, y2 = line[1]

        if x2 - x1 != 0:
            gradient = (y2 - y1) / (x2 - x1)
        else:
            return float('-inf'), float('inf'), float('inf'), lines[0], float('inf'), gradients

        gradients.append(gradient)
        line_gradients.append((gradient, line))

    if not gradients:
        return None, None, None, None, None, None  # Handling case when no valid gradients found

    min_gradient = min(gradients)
    max_gradient = max(gradients)
    avg_gradient = sum(gradients) / len(gradients)

    closest_line = line_gradients[0][1]  # Inisialisasi closest_line dengan garis pertama
    closest_difference = abs(line_gradients[0][0] - avg_gradient)  # Inisialisasi closest_difference dengan selisih pertama

    for gradient, line in line_gradients:
        difference = abs(gradient - avg_gradient)
        if difference < closest_difference:
            closest_difference = difference
            closest_line = line

    gradient_difference = max_gradient - min_gradient

    return min_gradient, max_gradient, avg_gradient, closest_line, gradient_difference, gradients

def cari_garis_terdekat(array, gradien):
    selisih_terkecil = float('inf')
    garis_terdekat = array[0]

    for garis in array:
        x1, y1 = garis[0]
        x2, y2 = garis[1]

        # Menghindari pembagian oleh nol
        if x2 - x1 != 0:
            gradien_garis = (y2 - y1) / (x2 - x1)
            selisih = abs(gradien_garis - gradien)
            if selisih <= selisih_terkecil:
                selisih_terkecil = selisih
                garis_terdekat = garis

    return garis_terdekat

def handle_tb_gradient_diff( top_gradient_difference, bot_gradient_difference, top_avg_gradient, bot_avg_gradient, top_closest_line, bot_closest_line, carte_top, carte_bot):
    if(top_gradient_difference <= bot_gradient_difference):
        xbestbot = cari_garis_terdekat(carte_bot, top_avg_gradient)
        xbesttop = top_closest_line
    else:
        xbesttop= cari_garis_terdekat(carte_top, bot_avg_gradient)
        xbestbot = bot_closest_line
    return xbesttop, xbestbot

def handle_lr_gradient_diff( left_gradient_difference, right_gradient_difference, left_avg_gradient, right_avg_gradient, left_closest_line, right_closest_line, carte_left, carte_right):
    if(left_gradient_difference <= right_gradient_difference):
        xbestright = cari_garis_terdekat(carte_right, left_avg_gradient)
        xbestleft = left_closest_line
    else:
        xbestleft= cari_garis_terdekat(carte_left, right_avg_gradient)
        xbestright = right_closest_line
    return xbestleft, xbestright

def draw_bestgaris_on_image(image, bestgaris, scale=1, thick=1):
    x1 = bestgaris[0][0]
    y1 = bestgaris[0][1]
    x2 = bestgaris[1][0]
    y2 = bestgaris[1][1]
    cv2.line(image,(int(x1)*scale,int(y1)*scale),(int(x2)*scale,int(y2)*scale),(255,50,255),thick)
    return image

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

def find_best(carte_left_filtered, carte_right_filtered, carte_top_filtered, carte_bot_filtered):
    best_kiri_atas = find_intersection_points_from_line(carte_top_filtered, carte_left_filtered)
    best_kiri_bawah = find_intersection_points_from_line(carte_bot_filtered, carte_left_filtered)
    best_kanan_atas = find_intersection_points_from_line(carte_top_filtered, carte_right_filtered)
    best_kanan_bawah = find_intersection_points_from_line(carte_bot_filtered, carte_right_filtered)
    return best_kiri_atas, best_kiri_bawah, best_kanan_atas, best_kanan_bawah

def showhasil( orig_img, best_kiri_atas, best_kiri_bawah, best_kanan_atas, best_kanan_bawah):
    imgres = cv2.resize(orig_img,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)

    cv2.line(imgres,(int(best_kiri_atas[0]),int(best_kiri_atas[1])),(int(best_kiri_bawah[0]),int(best_kiri_bawah[1])),(255, 0, 0), 1)
    cv2.line(imgres,(int(best_kiri_bawah[0]),int(best_kiri_bawah[1])),(int(best_kanan_bawah[0]),int(best_kanan_bawah[1])),(255, 0, 0), 1)
    cv2.line(imgres,(int(best_kanan_bawah[0]),int(best_kanan_bawah[1])),(int(best_kanan_atas[0]),int(best_kanan_atas[1])),(255, 0, 0), 1)
    cv2.line(imgres,(int(best_kanan_atas[0]),int(best_kanan_atas[1])),(int(best_kiri_atas[0]),int(best_kiri_atas[1])),(255, 0, 0), 1)
    return imgres

def same_orientation(best_kiri_atas_real, best_kiri_bawah_real, best_kanan_atas_real, best_kanan_bawah_real, img):
    curr_res_rot = -1
    img_rot = -1
    if(abs(best_kiri_atas_real[1] - best_kiri_bawah_real[1]) < abs(best_kiri_atas_real[0] - best_kanan_atas_real[0])):
        curr_res_rot = 0 #portrait 
    else:
        curr_res_rot = 1 #landscape
    
    if(img.shape[0] < img.shape[1]):
        img_rot = 0 #portrait
    else:
        img_rot = 1 #landscape
    if(curr_res_rot == img_rot):
        return 1 #same orientation
    else:
        return 0 #different orientation

def check_line_under_onefourth(img, best_kiri_atas_real, best_kiri_bawah_real, best_kanan_atas_real, best_kanan_bawah_real):
    if(((best_kanan_atas_real[0] - best_kiri_atas_real[0]) <= (img.shape[1]/2)) or 
       ((best_kanan_bawah_real[0] - best_kiri_bawah_real[0]) <= (img.shape[1]/2)) or
       ((best_kiri_bawah_real[1] - best_kiri_atas_real[1]) <= (img.shape[0]/2)) or
       ((best_kanan_bawah_real[1] - best_kanan_atas_real[1]) <= (img.shape[0]/2))):
        return 1
    else:
        return 0

def timesten(best_point):
    return best_point[0]*10, best_point[1]*10

def show_cropped_and_skewed(orig_img, best_kiri_atas, best_kiri_bawah, best_kanan_atas, best_kanan_bawah):
    points = [tuple(best_kiri_atas), tuple(best_kanan_atas), tuple(best_kanan_bawah), tuple(best_kiri_bawah)]
    output_image = crop_and_skew(orig_img.copy(), points)
    return output_image

def crop_main(path_image, cara = 0): 
    # 0 = crop berdasar gradien
    # 1 = crop berdasar banyaknya garis yg berpotongan (terluar), 
    # 2 = crop berdasar banyaknya garis yg berpotongan (terdalam),
    # 3 = crop berdasar energy terkecil 
    img = path_image
    orig_img = img.copy()
    resized_img = resize(img)
    l_channel, a_channel, b_channel, _ = convert_to_lab(resized_img)
    img_l = closing_l(l_channel)
    img_a = erosion_a(a_channel)
    img_b = b_channel
    imgra, imgra_l, imgra_a, imgra_b = sumlab(img_l, img_a, img_b)
    thresh = thresh_image(imgra)
    marker = setmarker(thresh)
    imws, imgra = watershed_image(imgra_l,imgra_a,imgra_b,marker)
    imws8 = convert_watershed_uint8(imws)
    hough_image, lines = hough(resized_img, imws8)
    if cara == 0:
        verticalLines, horizontalLines = calc_energy(imws8, lines)
    elif cara == 1 or cara == 2 or cara == 3:
        verticalLines, horizontalLines = divide_line_to_horizontal_and_vertical(imws8, lines)
    drawed_img = draw_all(resized_img, verticalLines, horizontalLines)
    carte_vert = polar_dot_to_cartesian_line(verticalLines)
    carte_horiz = polar_dot_to_cartesian_line(horizontalLines)
    carte_vert_filtered = convert_lines_in_image(carte_vert, resized_img)
    carte_horiz_filtered = convert_lines_in_image(carte_horiz, resized_img)
    carte_left, carte_right = vertical_divider(carte_vert_filtered, resized_img)
    if cara == 0:
        carte_left, carte_right = handle_left_right(carte_left, carte_right, resized_img)
    elif cara == 1 or cara == 2 or cara == 3:
        carte_left, carte_right = handle_left_right_2(carte_left, carte_right, resized_img)
    carte_top, carte_bot = horizontal_divider(carte_horiz_filtered, resized_img)
    if cara == 0:
        carte_top, carte_bot = handle_top_bot(carte_top, carte_bot, resized_img)
    elif cara == 1 or cara == 2 or cara == 3:
        carte_top, carte_bot = handle_top_bot_2(carte_top, carte_bot, resized_img)
    drawed_img_lrtb = draw_rl_tb(carte_left, carte_right, carte_top, carte_bot, resized_img)
    if cara == 0:
        top_min_gradient, top_max_gradient, top_avg_gradient, top_closest_line, top_gradient_difference, top_gradients = find_gradient_difference(carte_top)
        bot_min_gradient, bot_max_gradient, bot_avg_gradient, bot_closest_line, bot_gradient_difference, bot_gradients = find_gradient_difference(carte_bot)
        xbesttop, xbestbot = handle_tb_gradient_diff(top_gradient_difference, bot_gradient_difference, top_avg_gradient, bot_avg_gradient, top_closest_line, bot_closest_line, carte_top, carte_bot)
        left_min_gradient, left_max_gradient, left_avg_gradient, left_closest_line, left_gradient_difference, left_gradients = find_gradient_difference(carte_left)
        right_min_gradient, right_max_gradient, right_avg_gradient, right_closest_line, right_gradient_difference, right_gradients = find_gradient_difference(carte_right)
        xbestleft, xbestright = handle_lr_gradient_diff(left_gradient_difference, right_gradient_difference, left_avg_gradient, right_avg_gradient, left_closest_line, right_closest_line, carte_left, carte_right)
        best_kiri_atas, best_kiri_bawah, best_kanan_atas, best_kanan_bawah = find_best(xbestleft, xbestright, xbesttop, xbestbot)
    elif cara == 1 or cara == 2 or cara == 3:
        if cara == 1:
            carte_top_filtered, titik_potong = find_line_by_intersection_points(carte_top, carte_vert_filtered, img.shape[0], img.shape[1], True, False)
            carte_bot_filtered, titik_potong = find_line_by_intersection_points(carte_bot, carte_vert_filtered, img.shape[0], img.shape[1], False, False)
            carte_left_filtered, titik_potong = find_line_by_intersection_points(carte_left, carte_horiz_filtered, img.shape[0], img.shape[1], True, True)
            carte_right_filtered, titik_potong = find_line_by_intersection_points(carte_right, carte_horiz_filtered, img.shape[0], img.shape[1], False, True)
        elif cara == 2:
            carte_top_filtered, titik_potong = find_line_by_intersection_points_terdalam(carte_top, carte_vert_filtered, img.shape[0], img.shape[1], True, False)
            carte_bot_filtered, titik_potong = find_line_by_intersection_points_terdalam(carte_bot, carte_vert_filtered, img.shape[0], img.shape[1], False, False)
            carte_left_filtered, titik_potong = find_line_by_intersection_points_terdalam(carte_left, carte_horiz_filtered, img.shape[0], img.shape[1], True, True)
            carte_right_filtered, titik_potong = find_line_by_intersection_points_terdalam(carte_right, carte_horiz_filtered, img.shape[0], img.shape[1], False, True)
        elif cara == 3:
            carte_top_filtered, titik_potong = find_line_by_intersection_points_energy(carte_top, carte_vert_filtered, img.shape[0], img.shape[1], True, False)
            carte_bot_filtered, titik_potong = find_line_by_intersection_points_energy(carte_bot, carte_vert_filtered, img.shape[0], img.shape[1], False, False)
            carte_left_filtered, titik_potong = find_line_by_intersection_points_energy(carte_left, carte_horiz_filtered, img.shape[0], img.shape[1], True, True)
            carte_right_filtered, titik_potong = find_line_by_intersection_points_energy(carte_right, carte_horiz_filtered, img.shape[0], img.shape[1], False, True)
        best_kiri_atas = find_intersection_points_from_line(carte_top_filtered, carte_left_filtered)
        best_kiri_bawah = find_intersection_points_from_line(carte_bot_filtered, carte_left_filtered)
        best_kanan_atas = find_intersection_points_from_line(carte_top_filtered, carte_right_filtered)
        best_kanan_bawah = find_intersection_points_from_line(carte_bot_filtered, carte_right_filtered)
    imgres = showhasil(orig_img, best_kiri_atas, best_kiri_bawah, best_kanan_atas, best_kanan_bawah)
    best_kiri_atas_real = timesten(best_kiri_atas)
    best_kiri_bawah_real = timesten(best_kiri_bawah)
    best_kanan_atas_real = timesten(best_kanan_atas)
    best_kanan_bawah_real = timesten(best_kanan_bawah)
    if(check_line_under_onefourth(img, best_kiri_atas_real, best_kiri_bawah_real, best_kanan_atas_real, best_kanan_bawah_real)):
        real = orig_img.copy()
    else:
        real = show_cropped_and_skewed(orig_img, best_kiri_atas_real, best_kiri_bawah_real, best_kanan_atas_real, best_kanan_bawah_real)
    return  hough_image, drawed_img, drawed_img_lrtb, imgres, real
