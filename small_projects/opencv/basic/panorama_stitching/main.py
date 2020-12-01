import cv2
import os

# Initialize
mainFolder = 'Images'
myFolders = os.listdir(mainFolder)

for folder in myFolders:
    path = mainFolder + '/' + folder
    images = []
    myList = os.listdir(path)
    
    print(f'Total umber of images detected here is {len(myList)}')

    for img in myList:
        curImg = cv2.imread(f'{path}/{img}')
        curImg = cv2.resize(curImg, (0, 0), None, 0.2, 0.2)
        images.append(curImg)
    
    stitcher = cv2.Stitcher.create()
    (status, result) = stitcher.stitch(images)

    if status == cv2.STITCHER_OK:
        print('Panorama Generated')

        cv2.imshow(folder, result)
        cv2.waitKey(1)
    else:
        print('Failed')

cv2.waitKey(0)