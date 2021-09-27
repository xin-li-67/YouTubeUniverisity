# YoLoV3 Implementation in TF2.0 and TFX Serving

### Convert pre-trained Darknet weights

yolov3: 
```
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
```

yolov3-tiny:
```
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
python convert.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny
```

### Detection

yolov3:
```
python detect.py --image ./demo/meme.jpg
```

yolov3-tiny:
```
python detect.py --weights ./checkpoints/yolov3-tiny.tf --tiny --image ./demo/street.jpg
```

webcam:
```
python detect_video.py --video 0
```

video file:
```
python detect_video.py --video path_to_file.mp4 --weights ./checkpoints/yolov3-tiny.tf --tiny
```

video file with output:
```
python detect_video.py --video path_to_file.mp4 --output ./output.avi
```