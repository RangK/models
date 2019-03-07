import numpy as np
import os
import sys
import tensorflow as tf

from distutils.version import StrictVersion
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
 raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
# MODEL_NAME = "faster_rcnn_nas_coco_2018_01_28"

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
 od_graph_def = tf.GraphDef()
 with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
  serialized_graph = fid.read()
  od_graph_def.ParseFromString(serialized_graph)
  tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
 (im_width, im_height) = image.size
 return np.array(image.getdata()).reshape(
  (im_height, im_width, 3)).astype(np.uint8)


if __name__ == "__main__":
 video_path = "/home/storage_disk2/datasets/pangyo_pro/2th/starbucks/Starbucks_aveneufrance_front_door_1.MP4"
 cap = cv2.VideoCapture(video_path)

 if cap.isOpened():
  nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 total = nFrames

 with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   while nFrames > 0:
    print("{} / {}".format(total, nFrames), end="\r")
    result, frame = cap.read()
    if result is False:
     continue

    index = total - nFrames
    nFrames -= 1
    base_name = "frame{:04d}.jpg".format(index)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_np = np.array(rgb).astype(np.uint8)

    image_np_expanded = np.expand_dims(Image.fromarray(rgb), axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
     [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded}
    )

    vis_util.visualize_boxes_and_labels_on_image_array(
     image_np,
     boxes[0],
     classes[0].astype(np.uint8),
     scores[0],
     category_index,
     instance_masks=None,
     use_normalized_coordinates=True,
     line_thickness=8
    )

    Image.fromarray(image_np).save("/home/storage_disk2/datasets/pangyo_pro/2th/starbucks/Starbucks_aveneufrance_front_door_1/{}".format(base_name))

 cap.release()
 print("\n")
 print("end")

