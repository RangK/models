from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import xml.etree.ElementTree as ET

from absl import flags
import numpy as np
import cv2
from PIL import Image

import utils.file_utils as futils
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from enum import Enum
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

flags = tf.app.flags
flags.DEFINE_string('freeze_model', None, 'Path to directory holding a checkpoint')
flags.DEFINE_string('xml_dir', None, 'Path to read xmls')
flags.DEFINE_string('label_path', None, 'Path to read label')
flags.DEFINE_string('image_root_dir', None, 'Path to read image')
flags.DEFINE_string('output_dir', None, 'Path to output results of predict')

FLAGS = flags.FLAGS

# TODO : read from file
category_map = [
  {
    'id': 1,
    'name': 'nut'
  }, {
    'id': 2,
    'name': 'defect'
  }, {
    'id': 3,
    'name': 'box'
  }, {
    'id': 4,
    'name': 'empty'
  }
]

"""    
category_map = [
    {
        'id': 1,
        'name': 'p1'
    }, {
        'id': 2,
        'name': 'p6'
    }, {
        'id': 3,
        'name': 'r1'
    }, {
        'id': 4,
        'name': 'd1'
    }, {
        'id': 5,
        'name': 'r2'
    }, {
        'id': 6,
        'name': 'r7'
    }, {
        'id': 7,
        'name': 'v1'
    }, {
        'id': 8,
        'name': 't1'
    }, {
        'id': 9,
        'name': 't2'
    }, {
        'id': 10,
        'name': 'r5'
    }
]
"""


class InputDataFields(Enum):
  FILE_NAME = 'file_name'
  ANNOTATIONS = 'annotations'


class EvalSummary:
  def __init__(self):
    self.total_count = 0
    self.result_per_category = {}

  def success_of_category(self, category_id):
    self.total_count += 1
    summary = self.result_per_category.setdefault(category_id, {
      "success": 0,
      "not_found": 0,
      "failed": 0,
      "fn": 0,
      "tn": 0,
      "fp": 0,
      'failed_paths': []
    })

    summary['success'] += 1

  def failed_of_category(self, category_id, predict_category_id, target_path=None):
    self.total_count += 1
    summary = self.result_per_category.setdefault(category_id, {
      "success": 0,
      "not_found": 0,
      "failed": 0,
      "fn": 0,
      "tn": 0,
      "fp": 0,
      'failed_paths': []
    })

    summary['fp'] += 1
    if target_path is not None:
      summary['failed_paths'].append(target_path)

    if predict_category_id < 1:
      summary['not_found'] += 1
    else:
      summary['failed'] += 1
      fp_summary = self.result_per_category.setdefault(predict_category_id, {
        "success": 0,
        "not_found": 0,
        "failed": 0,
        "fn": 0,
        "tn": 0,
        "fp": 0,
        'failed_paths': []
      })

      fp_summary['fp'] += 1

  def precision(self, category_id):
    # tp / (tp + fp)
    summary = self.result_per_category.setdefault(category_id, {
      "success": 0,
      "not_found": 0,
      "failed": 0,
      "fn": 0,
      "tn": 0,
      "fp": 0,
      'failed_paths': []
    })
    tp_fp = summary['success'] + summary['fp']
    if tp_fp == 0:
      return 0

    return summary['success'] / tp_fp

  def recall(self, category_id):
    # tp / (tp + fn)
    summary = self.result_per_category.setdefault(category_id, {
      "success": 0,
      "not_found": 0,
      "failed": 0,
      "fn": 0,
      "tn": 0,
      "fp": 0,
      'failed_paths': []
    })
    tp_fn = summary['success'] + summary['fn']
    if tp_fn == 0:
      return 0

    return summary['success'] / tp_fn

  def __repr__(self):
    result = "[Evalutation result]\n"

    for category_id in self.result_per_category:
      category_name = ""
      for category in category_map:
        if category['id'] == category_id:
          category_name = category['name']

      success = self.result_per_category[category_id]['success']
      failed = self.result_per_category[category_id]['failed']
      not_found = self.result_per_category[category_id]['not_found']
      precision = self.precision(category_id)
      recall = self.recall(category_id)

      failed_path_str = ""
      for path in self.result_per_category[category_id]['failed_paths']:
        failed_path_str = "{}/{}".format(failed_path_str, path)

      result += "[{}] : success {}, failed {}, not_found {}, precision {}, recall {}, failed paths:{}\n".format(
        category_name,
        success,
        failed,
        not_found,
        precision,
        recall,
        failed_path_str
      )

    return result

  def __str__(self):
    return self.__repr__()


class InputData:
  class Annotation:
    def __init__(self, bbox, segmentation, category_id):
      # {x:, y:, r:, b:}
      self.bbox = bbox
      # [[x1, y1, x2, y2, ...]..]
      self.segmentation = segmentation
      # int
      self.category_id = category_id

  def __init__(self, file_name):
    self.file_name = file_name
    self.annotations = []

  def add_annotation(self, annotation):
    assert isinstance(annotation, InputData.Annotation)
    self.annotations.append(annotation)


def _read_images_from_xml(xml_file, begin_image_id=0):
  next_id = begin_image_id
  tree = ET.parse(xml_file)
  root = tree.getroot()
  images = []
  annotations = []

  for member in root.findall('image'):
    box_elements = member.findall('box')
    # polygon_element = member.find('polygon')
    # code = box_element[0].text
    # if box_element[0].attrib['name'] == "quality":
    #     code = box_element[1].text

    # polygon = polygon_element.attrib['points']
    # poly_points = [[int(float(p)) for p in point.split(",")] for point in polygon.split(";")]
    # poly_points = [int(float(p)) for point in polygon.split(";") for p in point.split(",")]

    image_attr = member.attrib
    images.append({
      "id": next_id,
      'height': int(image_attr['height']),
      'width': int(image_attr['width']),
      'file_name': image_attr['name'],
      'license': "SAMJINLND,.LTD."
    })

    for box_element in box_elements:
      code = box_element.attrib['label']
      category_id = -1
      for category in category_map:
        if category['name'] == code:
          category_id = category['id']
      if category_id == -1:
        print("category -1 : {}".format(code))

      annotations.append({
        "image_id": next_id,
        "bbox": [
          int(float(box_element.attrib['xtl'])),
          int(float(box_element.attrib['ytl'])),
          int(float(box_element.attrib['xbr'])),
          int(float(box_element.attrib['ybr']))
        ],
        "segmentation": [],
        "category_id": category_id,
        "id": next_id
      })

    next_id += 1

  return images, annotations, next_id


def read_xmls(root_dir):
  input_datas = []

  begin_image_id = 0
  for xml_file in glob.glob(root_dir + '/*.xml'):
    images, annotations, next_id = _read_images_from_xml(xml_file, begin_image_id)
    for image in images:
      input_data = InputData(file_name=image[InputDataFields.FILE_NAME.value])
      for annotation in annotations:
        if annotation['image_id'] == image['id']:
          input_data.add_annotation(
            InputData.Annotation(
              bbox=annotation['bbox'],
              segmentation=annotation['segmentation'],
              category_id=annotation['category_id']
            )
          )

      input_datas.append(input_data)

    begin_image_id = next_id

  return input_datas


def _read_single_image_for_predict(path):
  frame = cv2.imread(path)
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  return rgb


def get_target_image_paths(input_datas, image_root_dir):
  sample_image_paths = futils.get_files(image_root_dir, extension='.jpg', max_depth=2)
  target_image_paths = {}
  for input_data in input_datas:
    code = 3
    for ann in input_data.annotations:
      if ann.category_id == 1:
        code = 1
        break
      elif ann.category_id == 2:
        code = 2
        break

    target_image_paths.setdefault(input_data.file_name, {"code": code})

  for sample_image_path in sample_image_paths:
    file_name = sample_image_path.split("/")[-1]
    if file_name in target_image_paths:
      target_image_paths[file_name]["path"] = sample_image_path

  return target_image_paths


def predict_bbox_and_segmentation(input_datas, image_root_dir, categories, show_processing=False):
  result = EvalSummary()
  target_image_paths = get_target_image_paths(input_datas, image_root_dir)
  min_score_thresh = 0.5

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.freeze_model, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

    with tf.Session(graph=detection_graph) as sess:
      for i, target_name in enumerate(target_image_paths):
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # segmentation = detection_graph.get_tensor_by_name('detection_masks:0')

        print("{} / {}".format(target_name, target_image_paths[target_name]))
        full_path = target_image_paths[target_name]['path']
        category_id = target_image_paths[target_name]['code']

        rgb = _read_single_image_for_predict(full_path)
        image_np = np.array(rgb).astype(np.uint8)

        image_np_expanded = np.expand_dims(Image.fromarray(rgb), axis=0)

        (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded}
        )

        valid_scores = []
        valid_classes = []
        valid_boxs = []

        for score_i, score in enumerate(scores[0]):
          if score > 0.5:
            valid_scores.append(score)
            valid_classes.append(classes[0][score_i])
            valid_boxs.append(boxes[0][score_i])

        if len(valid_scores) == 0:
          result.failed_of_category(category_id, -1)
          Image.fromarray(image_np).save(os.path.join(FLAGS.output_dir, "{}".format(target_name)))
          continue

        valid_scores = np.array(valid_scores)
        valid_classes = np.array(valid_classes)
        valid_boxs = np.array(valid_boxs)

        is_defect = 2 in valid_classes
        is_nut = 1 in valid_classes
        selected_box_index = -1

        if is_defect:
          selected_box_index = np.where(valid_classes == 2)[0]
          if category_id == 2:
            result.success_of_category(category_id)
          else:
            result.failed_of_category(category_id, 2)
        elif is_nut:
          selected_box_index = np.where(valid_classes == 1)[0]
          if category_id == 1:
            result.success_of_category(category_id)
          else:
            result.failed_of_category(category_id, 1)
        else:
          if category_id == 3:
            result.success_of_category(category_id)
          else:
            result.failed_of_category(category_id, 3)

        selected_box_index = int(selected_box_index)
        best_score = valid_scores[selected_box_index]
        best_box = np.array(valid_boxs[selected_box_index])
        best_class = valid_classes[selected_box_index].astype(np.uint8)
        # best_mask = np.array(masks[0][best_index]).astype(np.uint8)

        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.array([best_box]),
          [best_class],
          [best_score],
          categories,
          instance_masks=None,
          use_normalized_coordinates=True,
          line_thickness=8,
          min_score_thresh=min_score_thresh
        )

        Image.fromarray(image_np).save(os.path.join(FLAGS.output_dir, "{}".format(target_name)))
        print("progress : {} / {}".format(i, len(target_image_paths)), end='\r')

  print("\n")
  print(result)
  with open(os.path.join(FLAGS.output_dir, "result.txt"), "w") as f:
    f.write(str(result))

  return result


def inference(category_id):
  result = EvalSummary()
  NUM_CLASSES = 3
  min_score_thresh = 0.5
  label_map = label_map_util.load_labelmap(FLAGS.label_path)
  categories = label_map_util.convert_label_map_to_categories(label_map,
                                                              max_num_classes=NUM_CLASSES,
                                                              use_display_name=True)
  categories = label_map_util.create_category_index(categories)

  image_paths = futils.get_files(FLAGS.image_root_dir, extension='.jpg', max_depth=2)
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.freeze_model, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

    with tf.Session(graph=detection_graph) as sess:
      for i, image_path in enumerate(image_paths):
        target_name = os.path.basename(image_path)
        rgb = _read_single_image_for_predict(image_path)
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

        valid_scores = []
        valid_classes = []
        valid_boxs = []

        for score_i, score in enumerate(scores[0]):
          if score > min_score_thresh:
            valid_scores.append(score)
            valid_classes.append(classes[0][score_i])
            valid_boxs.append(boxes[0][score_i])

        if len(valid_scores) == 0:
          result.failed_of_category(category_id, -1)
          Image.fromarray(image_np).save(os.path.join(FLAGS.output_dir, "{}".format(target_name)))
          continue

        valid_scores = np.array(valid_scores)
        valid_classes = np.array(valid_classes)
        valid_boxs = np.array(valid_boxs)

        defect_class_index = np.where(valid_classes == 2)[0]
        normal_class_index = np.where(valid_classes == 1)[0]

        if len(defect_class_index) == 0 and len(normal_class_index) == 0:
          result.failed_of_category(category_id, -1)
          Image.fromarray(image_np).save(os.path.join(FLAGS.output_dir, "{}".format(target_name)))
          continue

        elif len(defect_class_index) == 0:
          best_index = normal_class_index[0]
        elif len(normal_class_index) == 0:
          best_index = defect_class_index[0]
        else:
          normal_index = normal_class_index[0]
          defect_index = defect_class_index[0]

          print("{} / {}".format(normal_index, defect_index))
          if valid_scores[normal_index] < 0.8:
            best_index = defect_index
          else:
            best_index = defect_index if valid_scores[defect_index] > valid_scores[normal_index] else normal_index

        best_class = valid_classes[best_index].astype(np.uint8)
        is_defect = (best_class == 2)
        is_nut = (best_class == 1)

        if is_defect:
          if category_id == 2:
            result.success_of_category(category_id)
          else:
            result.failed_of_category(category_id, 2, image_path)
        elif is_nut:
          if category_id == 1:
            result.success_of_category(category_id)
          else:
            result.failed_of_category(category_id, 1, image_path)
        else:
          if category_id == 3:
            result.success_of_category(category_id)
          else:
            result.failed_of_category(category_id, 3, image_path)

        best_score = valid_scores[best_index]
        best_box = np.array(valid_boxs[best_index])

        # best_mask = np.array(masks[0][best_index]).astype(np.uint8)
        print(categories)
        print(best_box)
        print(best_class)
        print(best_score)
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.array([best_box]),
          [best_class],
          [best_score],
          categories,
          instance_masks=None,
          use_normalized_coordinates=True,
          line_thickness=8,
          min_score_thresh=min_score_thresh
        )

        Image.fromarray(image_np).save(os.path.join(FLAGS.output_dir, "{}".format(target_name)))
        print("progress : {} / {}".format(i, len(image_paths)), end='\r')

  print("\n")
  print(result)
  with open(os.path.join(FLAGS.output_dir, "result.txt"), "w") as f:
    f.write(str(result))


def validation():
  input_datas = read_xmls(FLAGS.xml_dir)
  print("STEP 02 : load category\n")
  NUM_CLASSES = 3
  label_map = label_map_util.load_labelmap(FLAGS.label_path)
  categories = label_map_util.convert_label_map_to_categories(label_map,
                                                              max_num_classes=NUM_CLASSES,
                                                              use_display_name=True)
  categories = label_map_util.create_category_index(categories)

  print("STEP 03 : start predict bounding boxes\n")
  predict_bbox_and_segmentation(input_datas, FLAGS.image_root_dir, categories, show_processing=True)


if __name__ == "__main__":
  if FLAGS.xml_dir is None:
    inference(category_id=1)
  else:
    print("STEP 01 : load input data from xmls\n")
    validation()

