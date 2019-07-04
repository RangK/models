import os
import glob
import json
import xml.etree.ElementTree as ET
from utils.label_map_util import get_label_map_dict


def get_categories_from_map_file(label_map_path):
  label_map = get_label_map_dict(label_map_path)
  categories = []
  for label_key in label_map:
    categories.append({
      'name': label_key,
      'id': label_map[label_key]
    })

  return categories


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


def _get_box_code(box_element):
  if box_element.attrib['name'] == "code":
    code = box_element.text
  else:
    code = box_element.text

  return code


def _parse_box_element(root_element):
  box_elements = root_element.findall('box')
  codes = []
  bboxes = []

  for box_element in box_elements:
    if len(box_element) > 0:
      code = _get_box_code(box_element[0])
    else:
      code = box_element.attrib['label']

    bbox = [
      int(float(box_element.attrib['xtl'])),
      int(float(box_element.attrib['ytl'])),
      int(float(box_element.attrib['xbr'])),
      int(float(box_element.attrib['ybr']))
    ]

    codes.append(code)
    bboxes.append(bbox)

  return codes, bboxes


def _parse_polygon_elements(root_element):
  polygon_elements = root_element.findall('polygon')
  polygons = []
  for polygon_element in polygon_elements:
    polygon = polygon_element.attrib['points']
    polygon = [int(float(p)) for point in polygon.split(";") for p in point.split(",")]
    polygons.append(polygon)

  return polygons


def _create_annotations(image_id, codes, bboxes, polygons, categories):
  annotations = []
  annotation_id_of_image = 0

  is_segmentation = polygons is not None
  iterator = zip(codes, bboxes, polygons) if is_segmentation else zip(codes, bboxes)

  for values in iterator:
    code = values[0]
    bbox = values[1]

    category_id = -1
    for category in categories:
      if category['name'] == code:
        category_id = category['id']
    if category_id == -1:
      print("category -1 : {}".format(code))

    if is_segmentation:
      annotations.append({
        "image_id": image_id,
        "bbox": bbox,
        "segmentation": [values[2]],
        "category_id": category_id,
        "id": annotation_id_of_image
      })
    else:
      annotations.append({
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id_of_image
      })

    annotation_id_of_image += 1

  return annotations


def _parse_xml(xml_tree, categories, begin_image_id, is_segmentation=False):
  images = []
  annotations = []
  next_id = begin_image_id

  for member in xml_tree.findall('image'):
    codes, bboxes = _parse_box_element(member)
    polygons = None
    if is_segmentation:
      polygons = _parse_polygon_elements(member)

    image_attr = member.attrib
    images.append({
      "id": next_id,
      'height': int(image_attr['height']),
      'width': int(image_attr['width']),
      'file_name': image_attr['name'],
      'license': "SAMJINLND,.LTD."
    })

    annotations_in_member = _create_annotations(
      next_id,
      codes,
      bboxes,
      polygons,
      categories
    )

    annotations += annotations_in_member
    next_id += 1

  return images, annotations, next_id


def xml_to_json(path, begin_next_id):
  label_map_file_path = './data/ess_label_map.pbtxt'
  categories = get_categories_from_map_file(label_map_file_path)
  print("==== Load categories ====")
  print(categories)
  print("=========================")

  json_data = {"images": [], "annotations": [], "categories": categories}
  count_xml_file = 0

  for xml_file in glob.glob(path + '/*.xml'):
    print("processing {} ..".format(count_xml_file), end="\r")
    count_xml_file += 1

    tree = ET.parse(xml_file)
    root = tree.getroot()

    images, annotations, next_id = _parse_xml(root, categories, begin_next_id)
    json_data['images'] += images
    json_data['annotations'] += annotations

    begin_next_id = next_id

  print("\n")
  print("end.")

  return json_data


def main():
  output_file_name = "ess_object_detection_annotations.json"
  json_output_dir = "/Users/rangkim/projects/datasets/ess/annotations/jsons_box"
  xml_root_dir = "/Users/rangkim/projects/datasets/ess/annotations/xmls_box"

  train_json = xml_to_json(os.path.join(xml_root_dir, 'train'), begin_next_id=0)
  eval_json = xml_to_json(os.path.join(xml_root_dir, 'eval'), begin_next_id=0)

  train_output_path = os.path.join(os.path.join(json_output_dir, 'train'), output_file_name)
  with open(train_output_path, 'w', encoding='utf-8') as output:
    json.dump(train_json, output)

  eval_output_path = os.path.join(os.path.join(json_output_dir, 'eval'), output_file_name)
  with open(eval_output_path, 'w', encoding='utf-8') as output:
    json.dump(eval_json, output)

  print('Successfully converted xml to json.\n * train : {} \n* eval : {}'.format(train_output_path, eval_output_path))


if __name__ == "__main__":
  main()