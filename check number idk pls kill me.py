import cv2
import pytesseract

img = cv2.imread('yo.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(pytesseract.image_to_string(img))
