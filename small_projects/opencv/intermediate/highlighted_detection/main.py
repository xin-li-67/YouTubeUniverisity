from utils import *
import pytesseract

path = 'test.png'
hsv = [0, 65, 59, 255, 0, 255]
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread(path)
imgResult = detect_color(img, hsv)
imgContours, contours = get_contours(imgResult, img, showCanny=True, minArea=1000, filter=4, cThr=[100,150], draw=True)
cv2.imshow("Contours", imgContours)


roi_list = get_roi(img, contours)
roi_display(roi_list)
highlighted = []

for i, roi in enumerate(roi_list):
    highlighted.append(pytesseract.image_to_string(roi))

save_text(highlighted)

imgStack = stack_images(0.7, ([img, imgResult, imgContours]))
cv2.imshow("Stacked", imgStack)
cv2.waitKey(0)