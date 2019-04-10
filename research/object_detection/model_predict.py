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

flags = tf.app.flags
flags.DEFINE_string('freeze_model', None, 'Path to directory holding a checkpoint')
flags.DEFINE_string('xml_dir', None, 'Path to read xmls')
flags.DEFINE_string('label_path', None, 'Path to read label')
flags.DEFINE_string('image_root_dir', None, 'Path to read image')
flags.DEFINE_string('output_dir', None, 'Path to output results of predict')

FLAGS = flags.FLAGS


class InputDataFields(Enum):
    FILE_NAME = 'file_name'
    ANNOTATIONS = 'annotations'


class EvalSummary:
    def __init__(self):
        self.total_count = 0
        self.result_per_category = {}

    def success_of_category(self, category_id):
        summary = self.result_per_category.setdefault(category_id, {"success": 0, "failed": 0})
        summary['success'] += 1

    def failed_of_category(self, category_id):
        summary = self.result_per_category.setdefault(category_id, {"success": 0, "failed": 0})
        summary['failed'] += 1


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


# TODO : read from file
categories = [
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


def _read_images_from_xml(xml_file, begin_image_id=0):
    next_id = begin_image_id
    tree = ET.parse(xml_file)
    root = tree.getroot()
    images = []
    annotations = []

    for member in root.findall('image'):
        box_element = member.find('box')
        polygon_element = member.find('polygon')

        code = box_element[0].text
        if box_element[0].attrib['name'] == "quality":
            code = box_element[1].text

        polygon = polygon_element.attrib['points']
        # poly_points = [[int(float(p)) for p in point.split(",")] for point in polygon.split(";")]
        poly_points = [int(float(p)) for point in polygon.split(";") for p in point.split(",")]

        image_attr = member.attrib
        images.append({
            "id": next_id,
            'height': int(image_attr['height']),
            'width': int(image_attr['width']),
            'file_name': image_attr['name'],
            'license': "SAMJINLND,.LTD."
        })

        category_id = -1
        for category in categories:
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
            "segmentation": [poly_points],
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
    image_np = np.array(rgb).astype(np.uint8)

    image_np_expanded = np.expand_dims(Image.fromarray(rgb), axis=0)

    return image_np_expanded


def get_target_image_paths(input_datas, image_root_dir):
    sample_image_paths = futils.get_files(image_root_dir, extension='.jpg', max_depth=2)
    target_image_paths = {}
    for input_data in input_datas:
        target_image_paths.setdefault(input_data.file_name, "")

    for sample_image_path in sample_image_paths:
        file_name = sample_image_path.split("/")[-1]
        if file_name in target_image_paths:
            target_image_paths[file_name] = sample_image_path

    return target_image_paths


def predict_bbox_and_segmentation(input_datas, image_root_dir, show_processing=False):
    result = EvalSummary()
    target_image_paths = get_target_image_paths(input_datas, image_root_dir)

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
                segmentation = detection_graph.get_tensor_by_name('detection_masks:0')

                full_path = target_image_paths[target_name]

                frame = cv2.imread(full_path)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_np = np.array(rgb).astype(np.uint8)

                image_np_expanded = np.expand_dims(Image.fromarray(rgb), axis=0)

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded}
                )

                best_index = np.argmax(scores[0])
                best_score = scores[0][best_index]

                best_box = np.array(boxes[0][best_index])
                best_class = classes[0][best_index].astype(np.uint8)
                # best_mask = np.array(masks[0][best_index]).astype(np.uint8)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.array([best_box]),
                    [best_class],
                    [best_score],
                    category_index,
                    instance_masks=None,
                    use_normalized_coordinates=True,
                    line_thickness=8
                )

                Image.fromarray(image_np).save("./tmp_{}.jpg".format(i))

    return result


if __name__ == "__main__":
    input_datas = read_xmls(FLAGS.xml_dir)

    NUM_CLASSES = 10
    label_map = label_map_util.load_labelmap(FLAGS.label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    results = predict_bbox_and_segmentation(input_datas, FLAGS.image_root_dir, show_processing=True)