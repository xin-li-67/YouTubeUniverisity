import os
import cv2
import numpy as np
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
from twilio.rest import Client

# for Mask-RCNN lib
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80 # 80 classes in coco plus one background
    DETECTION_MIN_CONFIDENCE = 0.6

# filter Mask RCNN detection res o get only the detected cars and trucks
def get_veh_boxes(boxes, class_ids):
    veh_boxes = []
    for i, box in enumerate(boxes):
        # only keep vehs
        if class_ids[i] in [3, 8, 6]:
            veh_boxes.append(box)
    
    return np.array(veh_boxes)

# directories
ROOT_DIR = Path(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# download coco if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# working dir
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
VIDEO_DIR = "test_images/parking.mp4"

# twilio config
twilio_account_sid = 'TWILIO_SID'
twilio_auth_token = 'TWILIO_AUTH_TOKEN'
twilio_phone_number = 'TWILIO_SOURCE_PHONE_NUMBER'
destination_phone_number = 'PHONE_NUMBER_TO_TEXT'
client = Client(twilio_account_sid, twilio_auth_token)

# create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", config=MaskRCNNConfig(), model_dir=MODEL_DIR)

# load pre-trained
model.load_weights(COCO_MODEL_PATH, by_name=True)

parked_veh_boxes = None
free_space_frames = 0        # count how many frames a potential parking lot is open in a row in the video
sms_sent = False

# load video
video = cv2.VideoCapture(VIDEO_DIR)
while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    rgb_image = frame[:, :, ::-1]
    results = model.detect([rgb_image], verbose=0)
    r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)
    if parked_veh_boxes is None:
        parked_veh_boxes = get_veh_boxes(r['rois'], r['class_ids'])
    else:
        veh_boxes = get_veh_boxes(r['rois'], r['class_ids'])
        overlaps = mrcnn.utils.compute_overlaps(parked_veh_boxes, veh_boxes)
        free_space = False
        
        for parking_area, overlap_areas in zip(parked_veh_boxes, overlaps):
            max_IoU_overlap = np.max(overlap_areas)
            y1, x1, y2, x2 = parking_area

            # take 0.15 as the IoU threshold
            if max_IoU_overlap < 0.15:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                free_space = True
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255,255,255))
        
        # If at least one space was free, start counting frames
        if free_space:
            free_space_frames += 1
        else:
            free_space_frames = 0
        
        # take 10 frams as the threshold here
        if free_space_frames > 10:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"SPACE AVAILABLE", (10, 150), font, 3.0, (0,255,0), 2, cv2.FILLED)

            if not sms_sent:
                print("SENDING SMS")
                message = client.messages.create(body="Parking availabe, RUSH!",
                                                 from_=twilio_phone_number,
                                                 to=destination_phone_number
                                                )
                sms_sent = True
    
        # Show the frame of video on the screen
        cv2.imshow('Video', frame)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up everything when finished
video.release()
cv2.destroyAllWindows()