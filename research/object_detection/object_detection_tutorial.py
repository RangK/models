import numpy as np
import os
import sys
import tensorflow as tf

from distutils.version import StrictVersion
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
 raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

MODEL_NAME = 'faster_rcnn_nas_0303_wfs'
# MODEL_NAME = "faster_rcnn_nas_coco_2018_01_28"

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'wfs_label_map.pbtxt')
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

def read_image_path(root_dir):
 return [os.path.join(root_dir, name) for name in os.listdir(root_dir)]

def read_tests_from_xml(xml_dri):
 return [name.replace(".xml", ".jpg") for name in os.listdir(xml_dri)]

if __name__ == "__main__":
 # video_path = "/home/storage_disk2/datasets/pangyo_pro/2th/starbucks/Starbucks_aveneufrance_front_door_1.MP4"
 # cap = cv2.VideoCapture(video_path)
 #
 # if cap.isOpened():
 #  nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

 PATH_TO_TEST_IMAGES_DIR = '/home/storage_disk2/datasets/lcd/2th_0104/bbox_datasets/crop_pattern_images'
 PATH_TO_XML_DIR = '/home/storage_disk2/datasets/lcd/2th_0104/bbox_datasets/xmls'
 image_paths = read_image_path(PATH_TO_TEST_IMAGES_DIR)
 tests = read_tests_from_xml(PATH_TO_XML_DIR)
 total = nFrames = len(image_paths)

 with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   for i, file_name in enumerate(image_paths):
    print("{:03} / {:03}".format(total, nFrames), end="\r")
    base_name = os.path.basename(file_name)
    if base_name not in tests:
     continue

    frame = cv2.imread(file_name)

    index = total - nFrames
    nFrames -= 1

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

    best_score = max(scores[0])
    best_index = scores[0].index(best_score)

    best_box = boxes[0][best_index]
    best_class = classes[0][best_index].astype(np.uint8)

    vis_util.visualize_boxes_and_labels_on_image_array(
     image_np,
     [best_box],
     [best_class],
     [best_score],
     category_index,
     instance_masks=None,
     use_normalized_coordinates=True,
     line_thickness=8
    )
    Image.fromarray(image_np).save("/home/storage_disk2/datasets/lcd/2th_0104/bbox_datasets/predict_images/{}".format(base_name))

 # cap.release()
 print("\n")
 print("end")

