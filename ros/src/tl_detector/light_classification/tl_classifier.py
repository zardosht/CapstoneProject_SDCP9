from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import os
import tensorflow as tf
from tf_object_detection_utils import label_map_util
import cv2


class TLClassifier(object):
    def __init__(self):
        self.light_state = TrafficLight.UNKNOWN
        self.model_loaded = False

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        graph_path = curr_dir + '/model/frozen_inference_graph.pb'

        num_classes = 13
        label_map = label_map_util.load_labelmap(curr_dir + '/label_map.pbtext')
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
            self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        self.model_loaded = True

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        self.light_state = TrafficLight.UNKNOWN
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image, axis=0)
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        min_score_thresh = .5
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                rospy.loginfo("TLClassifier: classes:{}".format(classes))
                class_name = self.category_index[classes[i]]['name']
                if class_name == 'Red':
                    self.light_state = TrafficLight.RED
                elif class_name == 'Green':
                    self.light_state = TrafficLight.GREEN
                elif class_name == 'Yellow':
                    self.light_state = TrafficLight.YELLOW

        if (self.light_state == TrafficLight.UNKNOWN):
            class_name = 'UNKNOWN'

        rospy.loginfo("Traffic light detected: class_name:{}".format(class_name))

        return self.light_state