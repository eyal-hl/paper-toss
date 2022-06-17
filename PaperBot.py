import cv2
from PIL import ImageGrab
import numpy as np
import pytesseract
import pyautogui
from math import sqrt
import time
from keyboard import on_press_key
# this code was made by Eyal Hermoni
# to run this code you need to install pyautogui and pytesseract and install the this https://github.com/UB-Mannheim/tesseract/wiki
# http://www.bestgames.com/Paper-Toss
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
running = True


def grab_image():
    img = ImageGrab.grab()
    img = np.array(img)
    img = img[:, :, ::-1].copy()
    return img


def find_important_screen(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([10, 205, 170])
    upper_red = np.array([20, 220, 200])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    points = []

    for contour in contours:
        moments = cv2.moments(contour)
        points.append((int(moments['m01']/moments['m00']), int(moments['m10']/moments['m00'])))

    return points


def find_ball(img,offY,offX,show=False):
    img = img[offY:offY + 350, offX - 70:offX + 50]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([100, 0, 100])
    upper_red = np.array([150, 50, 255])

    img = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((6, 6), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)


    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(contours[len(contours)-1])
    Y, X = moments['m01'] / moments['m00'], moments['m10'] / moments['m00']

    if (show):
        cv2.imshow('as', cv2.circle(img, (int(X), int(Y)), 10, (0, 0, 0), 3))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return int(X)+offX-70, int(Y)+offY


def find_wind(img, Y, X):
    img = img[Y - 100:Y, X - 120:X + 120]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, im_gray = cv2.threshold(img, 200, 240, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(im_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    number = get_number_from_image(crop_important_img(im_gray, contours))
    moments = cv2.moments(contours[0])
    X = moments['m10'] / moments['m00']
    if X < 100:
        number = number*-1
    return number


def throw_ball(X, Y, wind):
    wind = int(wind*10)
    newX = X+wind
    newY = 1000 - (wind ^ 2)
    newY = int(sqrt(newY))
    newY = Y - newY
    pyautogui.moveTo(X,Y,0.2)
    pyautogui.dragTo(newX,newY,0.5)


def get_score(img, point):
    Y, X = point
    #img = img[Y-460:Y-300, X-50:X+50]
    img = img[Y - 460:Y - 400, X - 50:X + 50]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.inRange(hsv, np.array([20, 200, 150]), np.array([30, 255, 255]))

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    number = get_number_from_image(crop_important_img(img, contours))
    return number
    #print(number)
    #number = get_number_from_image(img)
    #return number


def crop_important_img(img, cntrs):
    arrX, arrY, arrCX, arrCY = [], [], [], []
    for contour in cntrs:
        x, y, w, h = cv2.boundingRect(contour)
        arrX.append(x)
        arrY.append(y)
        arrCX.append(x+w)
        arrCY.append(y+h)

    return img[min(arrY)-10:max(arrCY)+10, min(arrX)-10:max(arrCX)+10]


def get_number_from_image(img):
    concateImg = np.concatenate((img, img), axis=1)
    concateImg = np.concatenate((concateImg, img), axis=1)
    concateImg = np.concatenate((concateImg, img), axis=1)
    concateImg = np.concatenate((concateImg, img), axis=1)
    textA = pytesseract.image_to_string(concateImg)
    textA = textA.strip()
    textA = textA.replace(" ", "")
    number = textA[0:int(len(textA) / 5)]
    if number == 'L11':
        number = '1.11'
    elif number == 'S':
        number = '5'
    elif number == 'Z':
        number = '2'

    try:
        number = float(number)
    except:
        number = 1.0
    #print(number)
    #cv2.imshow('wind.png', concateImg)
    #cv2.waitKey(0)
    return number


def control_one(img, point):
    Y, X = point
    wind = find_wind(img, Y, X)
    score = get_score(img, point)
    ball = find_ball(img, Y, X)
    print("your score is %s and the wind is %s" % (str(int(score)), str(wind)))
    throw_ball(ball[0], ball[1], wind)
    time.sleep(2)


def detect_stop(event):
    global running
    #time.sleep(10)
    running = False


if __name__ == '__main__':
    #threading.Thread(target=detect_stop).start()
    on_press_key('Escape', detect_stop)
    while running:
        try:
            img = grab_image()
            points = find_important_screen(img)
            if len(points) == 1:
                control_one(img, points[0])
            elif len(points) == 2:
                if get_score(img, points[1]) > get_score(img, points[0]):
                    print('Focusing right screen', end=' ')
                    control_one(img, points[0])
                else:
                    print('Focusing left screen', end=' ')
                    control_one(img, points[1])

            else:
                print("Why you opened %s windows?" % str(len(points)))

        except Exception as e:
            print(e)

