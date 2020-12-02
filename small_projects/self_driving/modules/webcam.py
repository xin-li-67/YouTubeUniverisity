import cv2

frame_width = 640
frame_height = 480

cap = cv2.VideoCapture(0)
cpa.set(3, frame_width)
cap.set(4, frame_height)

while True:
    _, img = cap.read()
    cv2.imshow("RES", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break