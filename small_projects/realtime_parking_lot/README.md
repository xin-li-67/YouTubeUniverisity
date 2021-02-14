# Realtime Parking Space Detector

Combining OpenCV and Mask RCNN together, the detector script is able to tell if there is any available parking lots from the loaded video. If there is a match, the script will trigger twilio service to send a notification message to the target phone number.  

## Workflow:  
Read input video stream -> Detect parking space -> Detect vehicles -> Detect available spaces -> Send messages  

## Detect Parking Spaces:  
This could be finished by either hardcoding the locations of all parking spaces or letting the script to detect. In order to run this step automatically, we can try detecting things like the lines of parking lots, the meters next to the parking lots, or any other specific signs. However, all these approaches are quite complicated. To better solve this problem, we can use the vechicles already parked along the street to help us. So if we can detect vehicles and figure out which ones are not moving between frasmes of video, we can infer the locations of parking spaces.

## Detect Vehicles:  
There are a lot of approaches to use for this step. Despite the old school image detection methods, there are still many machine learning based algorithms and models, such as HOG (Histogram of oriented gradients), CNN, and morden deep learning methods.  

For HOD based object detector, it runs pretty fast, but not be able to handle vehicles rotated in different orientations very well. For CNN based object detector, it runs accuratedly but not that efficient. For newr deeplearning learning based approaches such as Mask RCNN or YOLO, they combines the accuracy of CNNs with better, and more efficient designs to speed up the whole process.  

The Mask RCNN architecture is designed in such a way where it detects objects across the entire image in a computationally efficient manner without using a sliding window approach. With a modern GPU, we should be able to detect objects in high-res videos at several frames a second. Moreover, most object detection algorithms only return the bounding box of each object, but Mask RCNN will also give us an object outline (or mask).  

What we get from the Mask RCNN model includes:  
1. The type of the object;  
2. A confidence score of this detection;
3. The bounding box with x/y pixel locations;
4. A bitmap mask covers the object.

## Detect Empty Parking Spaces:
Using the previous assumption on detecting all parking spaces, we can determine if there is an available space by measuring how much two object overlap and then find the mostly empty boxes.  

This is also called Intersection Over Union (IoU) method. This will give us a measure of how much a car’s bounding box is overlapping a parking spot’s bounding box. With this, we can easily work out if a car is in a parking space or not. If the IoU measure is low, that means the car isn’t really occupying much of the parking space. Otherwise, we can be sure that the space is occupied.  

Before flagging a parking space as free, we should make sure it remains free for a little while — maybe 5 or 10 sequential frames of video. That will prevent the system from incorrectly detecting open parking spaces just because object detection had a temporary hiccup on one frame of video.  

## Send Notification Message:
There are many SMS services we can use for this step. Here I use Twilio which is easy to setup and program. Just need to install it to the virtual environment by: 
```pip install twilio```  
