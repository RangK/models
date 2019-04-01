import numpy as np
import os
from imutils.video import FPS
import sys
import tensorflow as tf
import imutils
from distutils.version import StrictVersion
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2
import dlib

box_colors = [(0,0,0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255), (192, 192, 192), (128, 128, 128),
              (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops


def np_vec_no_jit_iou(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

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

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
op_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
op_scores = detection_graph.get_tensor_by_name('detection_scores:0')
op_classes = detection_graph.get_tensor_by_name('detection_classes:0')
op_num_detections = detection_graph.get_tensor_by_name('num_detections:0')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
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
    video_path = "/Users/rangkim/projects/yodanjedi/ai/dev/object_tracker/data/videos/go_to_work_sample.mp4"
    # video_path = "/home/storage_disk2/datasets/pangyo_pro/sample/go_to_work_sample.mp4"
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = nFrames
    else:
        raise AssertionError("can't open a video {}".format(video_path))

        # PATH_TO_TEST_IMAGES_DIR = '/home/storage_disk2/datasets/lcd/2th_0104/bbox_datasets/crop_pattern_images'
        # PATH_TO_XML_DIR = '/home/storage_disk2/datasets/lcd/2th_0104/bbox_datasets/xmls'
        # image_paths = read_image_path(PATH_TO_TEST_IMAGES_DIR)
        # tests = read_tests_from_xml(PATH_TO_XML_DIR)
        # total = nFrames = len(image_paths) \

    writer = None
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # writer = cv2.VideoWriter("./tracking_output/output.avi", fourcc, 60,
    #                          (1920, 1080), True)
    thres_same_iou = 0.5

    trackers = []
    tracker_labels = []
    tracker_scores = []

    fps = FPS().start()

    frame_width = cap.get(3)  # float
    frame_height = cap.get(4)
    print("frame size : {} / {}".format(frame_width, frame_height))

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for _ in range(total):
                print("{:03} / {:03}".format(total, nFrames), end="\r")
                # base_name = os.path.basename(file_name)
                # if base_name not in tests:
                #  continue
                # frame = cv2.imread(file_name)
                index = total - nFrames
                nFrames -= 1
                result, frame = cap.read()
                if frame is None:
                    break

                # frame = imutils.resize(frame, width=600)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_np = np.array(rgb).astype(np.uint8)

                draw_boxs = []
                image_np_expanded = np.expand_dims(Image.fromarray(rgb), axis=0)
                (boxes, scores, classes, num_detections) = sess.run(
                    [op_boxes, op_scores, op_classes, op_num_detections], feed_dict={image_tensor: image_np_expanded}
                )

                for j, box in enumerate(boxes[0]):
                    category = category_index[classes[0][j]]
                    score = scores[0][j]
                    if score < 0.8 or category['id'] is not 1:
                        continue

                    minY, minX, maxY, maxX = box
                    minX = int(minX * frame_width)
                    maxX = int(maxX * frame_width)
                    minY = int(minY * frame_height)
                    maxY = int(maxY * frame_height)

                    bbox = [minX, minY, maxX, maxY]

                    is_tracking = False
                    for tracking_box in draw_boxs:
                        iou_score = np_vec_no_jit_iou(np.array([box]), np.array([tracking_box]))
                        if iou_score >= thres_same_iou:
                            is_tracking = True

                    if is_tracking:
                        continue

                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(minX, minY, maxX, maxY)
                    t.start_track(rgb, rect)

                    trackers.append(t)

                    draw_boxs.append(bbox)
                    tracker_labels.append(classes[0][j])
                    tracker_scores.append(scores[0][j])

                # for (t, l) in zip(trackers, tracker_labels):
                #     # update the tracker and grab the position of the tracked
                #     # object
                #     t.update(rgb)
                #     pos = t.get_position()
                #
                #     startX = int(pos.left())
                #     startY = int(pos.top())
                #     endX = int(pos.right())
                #     endY = int(pos.bottom())
                #
                #     draw_boxs.append([startX, startY, endX, endY])

                for box_i, draw_box in enumerate(draw_boxs):
                    startX = draw_box[0]
                    startY = draw_box[1]
                    endX = draw_box[2]
                    endY = draw_box[3]

                    cv2.rectangle(frame, (startX, startY), (endX, endY), box_colors[box_i], 2)

                if writer is not None:
                    writer.write(frame)
                else:
                    cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                fps.update()

    cap.release()
    fps.stop()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

