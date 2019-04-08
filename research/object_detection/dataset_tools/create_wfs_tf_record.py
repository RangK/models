# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw COCO dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_wfs_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import contextlib2
import numpy as np
import PIL.Image

from pycocotools import mask
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
tf.flags.DEFINE_boolean('include_masks', False,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('train_image_dir', '',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '',
                       'Validation image directory.')
tf.flags.DEFINE_string('test_image_dir', '',
                       'Test image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('testdev_annotations_file', '',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def get_directory_name(category_name):
    if category_name == 'p1':
        return "ACTP1C"
    elif category_name == 'p6':
        return "ACTP6C"
    elif category_name == 'r1':
        return "ACTR1C"
    elif category_name == 'r2':
        return "ACTR2D"
    elif category_name == 'r5':
        return "ACTR5T"
    elif category_name == 'r7':
        return "ACTR7W"
    elif category_name == 't1':
        return "ACTT1F"
    elif category_name == 't2':
        return "ACTT2C"
    elif category_name == 'v1':
        return "ACTV1R"

    raise IndexError("category name({}) is not exist in the category list".format(category_name))


def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()

    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    category_names = []
    category_ids = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    for object_annotations in annotations_list:
        (x, y, r, b) = tuple(object_annotations['bbox'])
        #TODO : checking x,y,r,b data is valid

        xmin.append(float(x) / image_width)
        xmax.append(float(r) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(b) / image_height)
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode('utf8'))

        if include_masks:
            # print(object_annotations['segmentation'][0])
            run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                image_height, image_width)
            # print(run_len_encoding)
            binary_mask = mask.decode(run_len_encoding)
            # print(binary_mask)
            np_bi_mask = np.array(binary_mask)
            binary_mask = np.reshape(np_bi_mask, (np_bi_mask.shape[0], np_bi_mask.shape[1]))
            print(np.array(binary_mask).shape)
            pil_image = PIL.Image.fromarray(binary_mask, mode="L")
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names)
    }

    if include_masks:
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


def _create_tf_record_from_wfs_annotations(annotations_file, image_root_dir, output_path, include_masks, num_shards):
    with contextlib2.ExitStack() as tf_record_close_stack, tf.gfile.GFile(annotations_file, 'r') as fid:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        category_index = label_map_util.create_category_index(
            groundtruth_data['categories'])

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            tf.logging.info(
                'Found groundtruth annotations. Building annotations index.')
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)
        missing_annotation_count = 0
        for image in images:
            image_id = image['id']
            if image_id not in annotations_index:
                missing_annotation_count += 1
                annotations_index[image_id] = []
        tf.logging.info('%d images are missing annotations.',
                        missing_annotation_count)

        total_num_annotations_skipped = 0
        for idx, image in enumerate(images):
            if idx % 100 == 0:
                tf.logging.info('On image %d of %d', idx, len(images))
            annotations_list = annotations_index[image['id']]
            category_name = category_index[annotations_list[0]['category_id']]['name']
            category_folder = get_directory_name(category_name)
            image_dir = os.path.join(image_root_dir, category_folder)

            _, tf_example, num_annotations_skipped = create_tf_example(
                image, annotations_list, image_dir, category_index, include_masks)
            total_num_annotations_skipped += num_annotations_skipped
            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        tf.logging.info('Finished writing, skipped %d annotations.',
                        total_num_annotations_skipped)


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    # assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    # assert FLAGS.test_image_dir, '`test_image_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    # assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
    # assert FLAGS.testdev_annotations_file, '`testdev_annotations_file` missing.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
        
    train_output_path = os.path.join(FLAGS.output_dir, 'wfs_mask_train.record')
    # val_output_path = os.path.join(FLAGS.output_dir, 'wfs_mask_val.record')
    # testdev_output_path = os.path.join(FLAGS.output_dir, 'wfs_mask_testdev.record')

    _create_tf_record_from_wfs_annotations(
        FLAGS.train_annotations_file,
        FLAGS.train_image_dir,
        train_output_path,
        FLAGS.include_masks,
        num_shards=100)

    # _create_tf_record_from_wfs_annotations(
    #     FLAGS.val_annotations_file,
    #     FLAGS.val_image_dir,
    #     val_output_path,
    #     FLAGS.include_masks,
    #     num_shards=10)
    #
    # _create_tf_record_from_wfs_annotations(
    #     FLAGS.testdev_annotations_file,
    #     FLAGS.test_image_dir,
    #     testdev_output_path,
    #     FLAGS.include_masks,
    #     num_shards=100)


if __name__ == '__main__':
    tf.app.run()
