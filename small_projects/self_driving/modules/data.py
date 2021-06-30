import cv2

from time import sleep
# for triggering the modules
import data as dm
import motor as mm
import webcam as wm
import joystick as jm

maxThrottle = 0.25
motor = mm.Motor(2, 3, 4, 17, 22, 27)
record = 0

while True:
    joyval = jm.getJS()
    steering = joyval['axis1']
    throttle = joyval['o'] * maxThrottle

    if joyval['share'] == 1:
        if record ==0: 
            print('Recording Started ...')
        record +=1
        sleep(0.300)
    if record == 1:
        img = wm.getImg(True, size=[240,120])
        dm.saveData(img, steering)
    elif record == 2:
        dm.saveLog()
        record = 0

    motor.move(throttle, -steering)
    cv2.waitKey(1)