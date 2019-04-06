import os
import glob
import json
import xml.etree.ElementTree as ET


def xml_to_json(path):
    # TODO : generated categories from file (wfs_label_map.pbtxt)
    json_data = {
        'images': [],
        'categories': [
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
            }
        ],
        'annotations': []
    }
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


main()