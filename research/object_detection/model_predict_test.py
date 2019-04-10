from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model_predict import read_xmls, get_target_image_paths, InputData, EvalSummary

correct_index_map = {
    "763a8z0136b6aae04_20181218_042832_-329078_380494.jpg": 0,
    "763a8z0136b6aal04_20181218_042832_-348927_-78318.jpg": 1,
    "763a8z0136b6aaq02_20181218_042832_-619803_-353217.jpg": 2
}

correct_input_datas = [{
    'annotations': [{
        "bbox": [526, 482, 539, 498],
        "segmentation": [
            [538, 491, 536, 488, 532, 485, 529, 484, 528, 484, 527, 484, 527, 496, 531, 497, 534, 497, 536, 495, 537,
             494, 537, 493]
        ],
        "category_id": 5
    }]
}, {
    'annotations': [{
        "bbox": [516, 449, 539, 477],
        "segmentation": [
            [534, 460, 528, 455, 524, 452, 521, 454, 521, 460, 522, 464, 523, 468, 525, 473, 526, 474, 528, 473, 530,
             469, 531, 470, 532, 472, 534, 471, 535, 470, 535, 467, 535, 466, 535, 464, 535, 462]
        ],
        "category_id": 5
    }]
}, {
    'annotations': [{
        "bbox": [450, 477, 492, 516],
        "segmentation": [
            [469, 481, 468, 483, 467, 487, 467, 491, 467, 494, 467, 495, 466, 497, 464, 498, 460, 500, 456, 502, 455,
             502, 453, 505, 454, 507, 455, 509, 456, 511, 460, 511, 462, 510, 466, 509, 467, 508, 469, 507, 471, 507,
             481, 506, 485, 506, 487, 505, 488, 501, 488, 499, 487, 496, 486, 493, 485, 492, 484, 492, 482, 492, 481,
             492, 479, 490, 478, 488, 477, 486, 475, 483, 473, 482, 471, 481]
        ],
        "category_id": 5
    }]
}]


class ModelPredictTest(tf.test.TestCase):

    def setUp(self):
        xmls_root_path = "./test_data/mask_annotation_xmls"
        self.input_datas = read_xmls(xmls_root_path)

    def test_read_xmls(self):
        # Image count = 30
        self.assertEqual(len(self.input_datas), 3, "length of input datas read from xmls must be 3")
        for input_data in self.input_datas:
            self.assertTrue(isinstance(input_data, InputData))
            self.assertTrue(input_data.file_name in correct_index_map)
            correct_input_data = correct_input_datas[correct_index_map[input_data.file_name]]

            self.assertEqual(len(input_data.annotations), len(correct_input_data['annotations']), "Length of Annotations")
            for i, annotation in enumerate(input_data.annotations):
                correct_annotation = correct_input_data['annotations'][i]
                self.assertEqual(annotation.bbox, correct_annotation['bbox'])

                self.assertEqual(annotation.segmentation[0], correct_annotation['segmentation'][0])
                self.assertEqual(annotation.category_id, correct_annotation['category_id'])

    def test_predict_bbox_and_segmentation(self):
        image_root_dir = "/Users/rangkim/projects/datasets/winforsys/origin/v2/sample/1th_images"
        target_image_paths = get_target_image_paths(self.input_datas, image_root_dir)

        correct_full_paths = {
            '763a8z0136b6aaq02_20181218_042832_-619803_-353217.jpg': '/Users/rangkim/projects/datasets/winforsys/origin/v2/sample/1th_images/ACTR2D/763a8z0136b6aaq02_20181218_042832_-619803_-353217.jpg',
            '763a8z0136b6aae04_20181218_042832_-329078_380494.jpg': '/Users/rangkim/projects/datasets/winforsys/origin/v2/sample/1th_images/ACTR2D/763a8z0136b6aae04_20181218_042832_-329078_380494.jpg',
            '763a8z0136b6aal04_20181218_042832_-348927_-78318.jpg': '/Users/rangkim/projects/datasets/winforsys/origin/v2/sample/1th_images/ACTR2D/763a8z0136b6aal04_20181218_042832_-348927_-78318.jpg'
        }
        self.assertEqual(target_image_paths, correct_full_paths)




if __name__ == "__main__":
    tf.test.main()