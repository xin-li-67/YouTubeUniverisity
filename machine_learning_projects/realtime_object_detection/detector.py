import cv2
import numpy as np

# params and model
net = cv2.dnn.readNet('yolov3-416.weights', 'yolov3-416.cfg')
classes = []
with open('coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Initialize
cap = cv2.VideoCapture(0)
whT = 416
confThres = 0.5
nmsThres = 0.2

def detection(outputs, img):
    hT, wT, _ = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThres:
                w, h = int(detection[2]*wT), int(detection[3]*hT)
                # x and y are center indices of the object
                x, y = int((detection[0]*wT) - w/2), int((detection[1]*hT) - h/2)

                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThres, nmsThres)

    for i in indices.flatten():
        x, y, w, h = bbox[i]
        label = str(classes[classIds[i]])
        confidence = str(round(confs[i], 2))
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,255), 2)
        cv2.putText(img, label + " " + confidence, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

while True:
    _, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT,whT), [0,0,0], swapRB=True, crop=False)
    net.setInput(blob)
    
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0]-1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    detection(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(0)