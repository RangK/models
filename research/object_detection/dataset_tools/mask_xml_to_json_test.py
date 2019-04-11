import shutil
import os
import tempfile
import unittest
import glob
import xml.etree.ElementTree as ET
from dataset_tools.mask_xml_to_json import get_categories_from_map_file, _parse_box_element, _parse_xml


class XMLToJSONTest(unittest.TestCase):

    def setUp(self):
        label_map_path = "../data/wfs_label_map.pbtxt"
        self.categories = get_categories_from_map_file(label_map_path)

    def test_get_categories(self):
        self.assertEqual(len(self.categories), 10)

        self.assertTrue('id' in self.categories[0])
        self.assertTrue('name' in self.categories[0])

    def test_parse_multi_bboxs(self):
        with open('../test_data/mask_annotation_xmls/test_mask_annotations_multi_bboxs.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()

            for member in root.findall('image'):
                codes, bboxes = _parse_box_element(member)

                self.assertEqual(len(codes), len(bboxes), "lengths of two list(codes, bboxes) must be equal")
                self.assertEqual(len(bboxes), 2)

    def test__parse_xml(self):
        with open('../test_data/mask_annotation_xmls/test_mask_annotations.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()

            images, annotations = _parse_xml(root, self.categories)

            self.assertEqual(len(images), 2)
            self.assertEqual(len(annotations), 3)
            self.assertEqual(annotations[0]['segmentation'][0],
                             [538, 491, 536, 488, 532, 485, 529, 484, 528, 484, 527, 484, 527, 496, 531, 497, 534, 497,
                              536, 495, 537,
                              494, 537, 493])
            self.assertEqual(annotations[1]['segmentation'][0],
                             [534, 460, 528, 455, 524, 452, 521, 454, 521, 460, 522, 464, 523, 468, 525, 473, 526, 474,
                              528, 473, 530,
                              469, 531, 470, 532, 472, 534, 471, 535, 470, 535, 467, 535, 466, 535, 464, 535, 462])
            self.assertEqual(annotations[1]['segmentation'][0], annotations[2]['segmentation'][0])

    # def test_multiple_mask_xml(self):
    #     xml_file_one =
        """
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>19</id>
      <name>WinForSys_LCD_T2</name>
      <size>120</size>
      <mode>annotation</mode>
      <overlap>0</overlap>
      # <bugtracker></bugtracker>
      <flipped>False</flipped>
      <created>2019-03-30 08:27:01.511919+03:00</created>
      <updated>2019-03-31 10:23:01.171991+03:00</updated>
      <source>120 images: 763a8z0125a5bab03_20181218_141223_490157_632315.jpg, 763a8z0125a5bak05_20181218_141223_819331_-32193.jpg, ...</source>
      <labels>
        <label>
          <name>defect</name>
          <attributes>
            <attribute>~radio=quality:good,bad</attribute>
            <attribute>@select=code:__undefined__,p1,p6,r1,r2,r5,r7,t1,t2,v1</attribute>
          </attributes>
        </label>
      </labels>
      <segments>
        <segment>
          <id>19</id>
          <start>0</start>
          <stop>119</stop>
          <url>http://label.yodanjedi.com/?id=19</url>
        </segment>
      </segments>
      <owner>
        <username>ideadripper</username>
        <email>ideadripper@gmail.com</email>
      </owner>
    </task>
    <dumped>2019-04-01 12:47:43.082206+03:00</dumped>
  </meta>
  <image id="0" name="defect_test.jpg" width="1024" height="1024">
    <box label="defect" xtl="385.15" ytl="401.17" xbr="556" ybr="568.06" occluded="0">
      <attribute name="quality">good</attribute>
      <attribute name="code">t2</attribute>
    </box>
    <polygon label="defect" points="411.16,442.70;459.19,414.79;506.57,419.33;541.62,450.49;551.35,521.24;503.97,559.53;444.26,560.18;408.56,534.22;405.96,519.94;394.93,485.54;403.37,455.68" occluded="0">
      <attribute name="quality">good</attribute>
      <attribute name="code">t2</attribute>
    </polygon>
  </image>
</annotations>
        """