import cv2
import numpy as np

# Initialize
cap = cv2.VideoCapture(0)
img_target = cv2.imread('test.jpg')
my_video = cv2.VideoCapture("sample.mp4")

detection = False
frame_counter = 0

# Grab the first frame: to resize it to the target image
success, img_video = my_video.read()
hT, wT, cT = img_target.shape
img_video = cv2.resize(img_video, (wT, hT))

# ORB detector with 1000 features
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img_target, None)

# Put everything in one window
def stack_images(img_array, scale, labels=[]):
    w_size = img_array[0][0].shape[1]
    h_size = img_array[0][0].shape[0]
    rows = len(img_array)
    cols = len(img_array[0])
    row_available = isinstance(img_array[0], list)

    if row_available:
        for x in range(0, rows):
            for y in range(0, cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (w_size, h_size), None, scale, scale)
                
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BAR)
                
        img_blank = np.zeros((h_size, w_size, 3), np.unit(8))
        hor = [img_blank] * rows
        hor_con = [img_blank] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
            hor_con[x] = np.concatenate(img_array[x])
        
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            img_array[x] = cv2.resize(img_array[x], (w_size, h_size), None, scale, scale)

            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BAR)
        
        hor = np.hstack(img_array)
        hor_con = np.concatenate(img_array)
        ver = hor
    
    if len(labels) != 0:
        each_img_width = int(ver.shape[1] / cols)
        each_img_height = int(ver.shape[0] / rows)
        
        for r in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c*each_img_width, r*each_img_height), (c*each_img_width + len(labels[r])*13+27, 30+each_img_height*r), (255,255,255), cv2.FILLED)
                cv2.putText(ver, labels[r], (c*each_img_width+10, r*each_img_height+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,255), 2)
    
    return ver

while True:
    success, img_webcam = cap.read()
    img_aug = img_webcam.copy()
    kp2, des2 = orb.detectAndCompute(img_webcam, None)
    # img_webcam = cv2.DrawKeypoints(img_webcam, kp2, None)

    if detection == False:
        my_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_counter = 0
    else:
        # Reach the maximum frame or not
        if frame_counter == my_video.get(cv2.CAP_PROP_POS_FRAME_COUNT):
            my_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
        
        success, img_video = my_video.read()
        img_video = cv2.resize(img_video, (wT, hT))
    
    # Use KNN brute force match to find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    valid = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            valid.append(m)
        
    img_features = cv2.drawMatches(img_target, kp1, img_webcam, kp2, valid, None, flags=2)

    # Homography: find the relationship(matrix) between points in training image and the location of these points in the query(target) image
    if len(valid) > 20:
        detection = True
        src_pts = np.float32([kp1[m.queryIdx].pt for m in valid]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in valid]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
        print(matrix)
        
        # Find the bounding box
        pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        img2 = cv2.polylines(img_webcam, [np.int32(dst)], True, (255,0,255), 3)
        # Augmentation
        img_warp = cv2.warpPerspective(img_video, matrix, (img_webcam.shape[1], img_webcam.shape[0]))

        # Create the appropraite mask to overlay the target on sample images
        mask_new = np.zeros((img_webcam.shape[0],img_webcam.shape[1]), np.uint8)
        cv2.fillPoly(mask_new, [np.int32(dst)], (255,255,255))
        mask_inv = cv2.bitwise_not(mask_new)

        img_aug = cv2.bitwise_and(img_aug, img_aug, mask=mask_inv)
        # Add final img with warped image
        img_aug = cv2.bitwise_or(img_warp, img_aug)
 
        img_stacked = stack_images(([img_webcam, img_video, img_target], [img_features, img_warp, img_aug]), 0.5)
    
    # Add FPS info on the augmentation view
    # timer = cv2.getTickCount()
    # fps = cv2.getTickCount() / (cv2.getTickCount() - timer)
    # cv2.putText(stack_images, 'FPS: {} '.format(int(fps)), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,20,20), 3)
    # cv2.putText(stack_images, 'Target Found: {} '.format(detection), (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,20,20), 3)

    cv2.imshow('Stacked Img', img_stacked)
    cv2.waitKey(1)

    frame_counter += 1