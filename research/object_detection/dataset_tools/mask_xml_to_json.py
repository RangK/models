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


def xml_to_json(path):
    categories = get_categories_from_map_file('./data/wfs_label_map.pbtxt')
    json_data = {"images": [], "annotations": [], "categories": categories}
    begin_image_id = 0

    count_xml_file = 0
    for xml_file in glob.glob(path + '/*.xml'):
        print("processing {} ..".format(count_xml_file), end="\r")
        count_xml_file += 1

        tree = ET.parse(xml_file)
        root = tree.getroot()
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
            json_data['images'].append({
                "id": begin_image_id,
                'height': int(image_attr['height']),
                'width': int(image_attr['width']),
                'file_name': image_attr['name'],
                'license': "SAMJINLND,.LTD."
            })

            category_id = -1
            for category in json_data['categories']:
                if category['name'] == code:
                    category_id = category['id']
            if category_id == -1:
                print("category -1 : {}".format(code))

            json_data['annotations'].append({
                "image_id": begin_image_id,
                "bbox": [
                    int(float(box_element.attrib['xtl'])),
                    int(float(box_element.attrib['ytl'])),
                    int(float(box_element.attrib['xbr'])),
                    int(float(box_element.attrib['ybr']))
                ],
                "segmentation": [poly_points],
                "category_id": category_id,
                "id": begin_image_id
            })

            begin_image_id += 1

    print("\n")
    print("end.")
    # xml_df = pd.DataFrame(xml_list, columns=column_name)
    return json_data


def main():
    annotations_file = "wfs_mask_annotations.json"

    image_path = os.path.join(os.getcwd(), 'annotations')
    json_data = xml_to_json(image_path)

    with open(os.path.join('./', annotations_file), 'w', encoding='utf-8') as output:
        json.dump(json_data, output)

    # xml_df.to_csv('wfs_mask_labels.csv', index=None)
    print('Successfully converted xml to json.')


if __name__ == "__main__":
    main()