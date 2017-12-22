import numpy as np
import os
import sys
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image
import time
import functools
import cv2
from datetime import datetime

from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from utils import visualization_utils as vis_util
from utils import label_map_util

slim = tf.contrib.slim

sys.path.append("..")

#pipeline_config_path = 'samples/configs/ssd_mobilenet_v1_voc.config'
pipeline_config_path = 'samples/configs/ssd_inception_v2_coco.config'
#checkpoint_dir = '../data/output/ssd_mobilenet_v1_voc_1.0'
checkpoint_dir = '../data/pre_model/ssd_inception_v2_coco_11_06_2017'

test_image_path = 'test_images'


class SSD_MobileNet_Detection:
    def __init__(self):
        self.PATH_TO_LABELS = os.path.join('data', 'pascal_label_map.pbtxt')
        self.NUM_CLASSES = 20
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.NUM_CLASSES,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.IMAGE_SIZE = (12, 8)

        #get model config from config file
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.gfile.GFile(pipeline_config_path, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)
        model_config = pipeline_config.model

        #get model from config file model_name
        create_model_fn = functools.partial(
            model_builder.build,
            model_config=model_config,
            is_training=False)
        self.model = create_model_fn()

        #placeholder image data and do predict.
        self.img_data = tf.placeholder(tf.uint8, shape=(None, None, 3))
        expand_image = tf.expand_dims(self.img_data, 0)
        preprocessed_image = self.model.preprocess(tf.to_float(expand_image))
        prediction_dict = self.model.predict(preprocessed_image)
        self.detections = self.model.postprocess(prediction_dict)

        #restore graph model from ckpt file
        variables_to_restore = tf.global_variables()
        global_step = slim.get_or_create_global_step()
        variables_to_restore.append(global_step)
        saver = tf.train.Saver(variables_to_restore)
        self.sess = tf.Session()
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(self.sess, latest_checkpoint)

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def detect_img(self):
        for img_name in os.listdir(test_image_path):
            if '.jpg' in img_name:
                image = Image.open(os.path.join(test_image_path,img_name))
                #image = image.resize((224, 224))
                start_time = time.time()
                print('@ailias')
                # result image with boxes and labels on it.
                image_data = self.load_image_into_numpy_array(image)

                detections = self.sess.run(self.detections, feed_dict={self.img_data: image_data})

                # # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_data,
                    np.squeeze(detections['detection_boxes']),
                    np.squeeze(detections['detection_classes']).astype(np.int32),
                    np.squeeze(detections['detection_scores']),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4)
                plt.figure(figsize=self.IMAGE_SIZE)
                plt.imshow(image_data)
                #plt.waitforbuttonpress()
                print('tot time:', time.time() - start_time)
                #print(detections)

    def detect_video(self, video_name=None):
        if video_name is None:
            capture = cv2.VideoCapture(0)
        else:
            capture = cv2.VideoCapture(video_name)
        while True:
            ret, image_data = capture.read()
            if not ret:  # no img data
                break
            start_time = time.time()
            detections = self.sess.run(self.detections, feed_dict={self.img_data: image_data})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_data,
                np.squeeze(detections['detection_boxes']),
                np.squeeze(detections['detection_classes']).astype(np.int32)+1,
                np.squeeze(detections['detection_scores']),
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4)
            cv2.imshow('Detected', image_data)
            cv2.waitKey(1)
            print("detect time(ms):%f" % (time.time() - start_time))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    detect = SSD_MobileNet_Detection()
    detect.detect_video()
