from flask import Flask
from Fetch_images import storage, database
from datetime import date, datetime
import sys
import re

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import pytesseract
from skimage.filters import threshold_local
from PIL import Image
from pytesseract import Output

from joblib import load
with open('SVC_model.pkl', 'rb') as f:
    model = load(f)


app = Flask(__name__)
data = {
       'Beauty & Hygiene' : 0,
       'Kitchen, Garden & Pets': 0, 
       'Accessories' : 0,
       'Shoes' : 0,
       'Shirts' : 0,
       'Activewear' : 0,
       'Pants' : 0,
       'Cleaning & Household' : 0,
       'Foodgrains, Oil & Masala' : 0,
       'Gourmet & World Food' : 0,
       'Snacks & Branded Foods' : 0,
       'Coats' : 0,
       'Eggs, Meat & Fish' : 0,
       'Underwear and Nightwear' : 0,
       'Suits' : 0,
       'Sweaters' : 0,
       'Bakery, Cakes & Dairy' : 0,
       'Jewelry' : 0,
       'Beverages' : 0,
       'Baby Care' : 0,
       'Fruits & Vegetables' : 0 
}

def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def plot_rgb(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def plot_gray(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(image, cmap='Greys_r')

# approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)

def get_receipt_contour(contours):    
    # loop over the contours
    for c in contours:
        approx = approximate_contour(c)
        # if our approximated contour has four points, we can assume it is receipt's rectangle
        if len(approx) == 4:
            return approx

def contour_to_rect(contour, resize_ratio):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio

def wrap_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


#b&w scanner effect
def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255



@app.route('/')
def hello_world():
    # database.child()
    storage.child()
    storage.download("receipt.jpg" ,"temp.jpg" )
    file_name = "temp.jpg"
    img = Image.open(file_name)
    img.thumbnail((800,800), Image.ANTIALIAS)

    image = cv2.imread(file_name)
    # Downscale image as finding receipt contour is more efficient on a small image
    resize_ratio = 500 / image.shape[0]
    original = image.copy()
    image = opencv_resize(image, resize_ratio)

    # Convert to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #plot_gray(gray)

    # Get rid of noise with Gaussian Blur and Adaptive Threshold filter
    blurred=cv2.GaussianBlur(gray,(5,5),0)
    ret3,blurred = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blurred = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    #plot_gray(blurred)

    # Detect white regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    # plot_gray(dilated)

    edged = cv2.Canny(dilated, 50,100, apertureSize=3)
    # plot_gray(edged)

    # Detect all contours in Canny-edged image
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)
    # plot_rgb(image_with_contours)

    #for images that have other objects detected
    # Get 10 largest contours
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    image_with_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0,255,0), 3)
    # plot_rgb(image_with_largest_contours)

    #
    receipt_contour = get_receipt_contour(largest_contours)
    image_with_receipt_contour = cv2.drawContours(image.copy(), [receipt_contour], -1, (0, 255, 0), 2)
    # plot_rgb(image_with_receipt_contour)

    scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour, resize_ratio))
    # plt.figure(figsize=(16,10))
    # plt.imshow(scanned)

    result = bw_scanner(scanned)
    # plot_gray(result)

    ### uncomment this line and provide local path to tesseract, in case an error occurs
    # pytesseract.pytesseract.tesseract_cmd = r"C:/Users/USER/AppData/Local/Programs/Tesseract-OCR/tesseract"

    d = pytesseract.image_to_data(result, output_type=Output.DICT)
    n_boxes = len(d['level'])
    boxes = cv2.cvtColor(result.copy(), cv2.COLOR_BGR2RGB)
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])    
        boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # plot_rgb(boxes)
    extracted_text = pytesseract.image_to_string(result)
    # print(extracted_text)
    extracted_text=extracted_text.lower()
    lst= extracted_text.split('\n')
    lst1=['item','mrp','qty','product','rate']

    count=0
    ctr =0
    flag=0
    for i in lst:
        count+=1
        #print(i)
        # print(i.split())
        for j in lst1:
            # print(j)
            if j in i.split():
                flag=1
                break
        if flag==1:
            break

    extracted_text = pytesseract.image_to_string(result)

    lst2=['total','tolal','lolal','tatal','totol','lotol','yotal']
    flag=0
    lst3=lst[count:]
    words_pattern = '[a-z]+'
    total_amount = 0.0 ## for analytics total 
    for i in lst3:
        for j in lst2:
            if j in i.split():
                flag=1
                break
        if flag==1:
            break
        text=lst[count]
        num=re.findall(r'[-+]?(?:\d*\.*\d+)', text, flags=re.IGNORECASE)
        amt= num[-1]
        total_amount += float(amt)
        temp=re.findall(words_pattern, text, flags=re.IGNORECASE)
        text= " ".join(temp)
        txt = [text]
        prediction = model.predict(txt)
        print(txt)
        print(prediction)
        print(amt)
        print(type(amt))
        print()
        count+=1
        
        ## syntax correcting before pushing to firebase database
        prediction = np.array_str(prediction).replace("[","").replace("]","").replace("'","")
        if prediction is not None and prediction in data:
            data[prediction] += float(amt)
        else:
            data[prediction] = float(amt)
    data1 = {"data" : data}
    database.set(data1)
    
    current_time = datetime.now()
    ## first date is set to June 1st 2023, (yyyy,mm,dd)
    d0 = date(current_time.year, 6, 1)
    d1 = date(current_time.year, current_time.month, current_time.day)
    delta = d1 - d0
    data2 ={  
                "months":(delta.days)//30,
                "days":delta.days,
                "total":total_amount,
            }
    database.child("expenditure")
    database.set(data2)

    return '<h2>Hello World 2<h2>'

 
if __name__ == '__main__':
    app.run()
