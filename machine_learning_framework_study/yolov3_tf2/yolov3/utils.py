import cv2
import numpy as np
from numpy.lib.arraysetops import isin
from six import b
import tensorflow as tf

from absl import logging

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]

def load_darknet_weights(model, weights_file, tiny=False):
    weight_file = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(weight_file, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_module = model.get_layer(layer_name)
        
        for i, layer in enumerate(sub_module.layers):
            if not layer.name.startswith('conv2d'):
                continue

            batch_norm = None

            if i + 1 < len(sub_module.layers) and sub_module.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_module.layers[i + 1]
            
            logging.info("{}/{} {}".format(sub_module.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            input_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(weight_file, dtype=np.float32, count=filters)
            else:
                # from darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(weight_file, dtype=np.float32, count=filters)
                # to tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[1, 0, 2, 3]


            # from darknet shape (out_dim, input_dim, height, width)
            conv_shape = (filters, input_dim, size, size)
            conv_weights = np.fromfile(weight_file, dtype=np.float32, count=np.product(conv_shape))
            # to tf shape (height, width, input_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(weight_file.read()) == 0, 'failed to read all data'
    weight_file.close()

def broadcast_iou(bbox_1, bbox_2):
    # bbox_1: (..., (x1, y1, x2, y2))
    # bbox_2: (N, (x1, y1, x2, y2))

    # broadcast bboxes
    bbox_1 = tf.expand_dims(bbox_1, -2)
    bbox_2 = tf.expand_dims(bbox_2, 0)

    # newshape (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(bbox_1), tf.shape(bbox_2))
    bbox_1 = tf.broadcast_to(bbox_1, new_shape)
    bbox_2 = tf.broadcast_to(bbox_2, new_shape)

    overlapped_w = tf.maximum(tf.minimum(bbox_1[..., 2], bbox_2[..., 2]) -
                       tf.maximum(bbox_1[..., 0], bbox_2[..., 0]), 0)
    overlapped_h = tf.maximum(tf.minimum(bbox_1[..., 3], bbox_2[..., 3]) -
                       tf.maximum(bbox_1[..., 1], bbox_2[..., 1]), 0)
    overlapped_area = overlapped_w * overlapped_h

    bbox_1_area = (bbox_1[..., 2] - bbox_1[..., 0]) * (bbox_1[..., 3] - bbox_2[..., 1])
    bbox_2_area = (bbox_2[..., 2] - bbox_2[..., 0]) * (bbox_2[..., 3] - bbox_2[..., 1])
    return overlapped_area / (bbox_1_area + bbox_2_area - overlapped_area)

def draw_outputs(img, outputs, class_names):
    bboxes, objectness, classes, nums = outputs
    bboxes, objectness, classes, nums = bboxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])

    for i in range(nums):
        x1y1 = tuple((np.array(bboxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(bboxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangele(img, x1y1, x2y2, (255,0,0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2) 
    return img

def draw_labels(x, y, class_names):
    img = x.numpy()
    bboxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])

    for i in range(bboxes):
        x1y1 = tuple((np.array(bboxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(bboxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255,0,0), 2)
        img = cv2.putText(img, class_names[classes[i]], x1y1, 
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
    return img

def freeze_all(model, frozen=True):
    model.trainable = not frozen

    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)