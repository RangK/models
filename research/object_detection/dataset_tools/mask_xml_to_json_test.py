import shutil
import os
import tempfile
import unittest
import object_detection.dataset_tools.mask_xml_to_json as xml_parser
from xml.etree import ElementTree as ET


class XMLToCSVTest(unittest.TestCase):
    def test_multiple_mask_xml(self):
        xml_file_one = """
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

        xml = ET.fromstring(xml_file_one)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tree = ET.ElementTree(xml)
            tree.write(tmpdirname + '/test_multiple_mask.xml')
            raccoon_df = xml_parser.xml_to_json(tmpdirname)
            print(raccoon_df)
            # self.assertEqual(raccoon_df.columns.values.tolist(),
            #                  ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'segmentation'])
            # self.assertEqual(raccoon_df.values.tolist()[0], ['defect_test.jpg', 1024, 1024, 't2', 385, 401, 556, 568,
            #                                                  [411, 442, 459, 414, 506, 419, 541, 450, 551, 521, 503,
            #                                                   559, 444, 560, 408, 534, 405, 519, 394, 485, 403, 455]])