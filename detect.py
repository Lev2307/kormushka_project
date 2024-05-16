import sys
import time
import threading
import queue

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

CAMERA_RTSP_URL = "rtsp://admin_camera:camera_password@192.168.222.105:554/stream2"

bird_detection = 'tf_bird_trained_models/bird_detection/birds_model_lite1.tflite' 
bird_classification = 'tf_bird_trained_models/bird_classification/10_birds_species_model_lite1.tflite'
first_retrained_bird_detection_model = "tf_bird_trained_models/retrained_models_for_detection/retrain1_birds_detection_lite0.tflite"
second_retrained_bird_detection_model = "tf_bird_trained_models/retrained_models_for_detection/retrain2_birds_detection_lite0.tflite"


q = queue.Queue()

def initialize_livestream():    
    # init livestream
    cap = cv2.VideoCapture(CAMERA_RTSP_URL)
    while cap.isOpened():
        ret, frame = cap.read()
        q.put(frame)

def display_livestream():
      # init tf models
      base_options = core.BaseOptions(
          file_name=first_retrained_bird_detection_model, num_threads=4)
      detection_options = processor.DetectionOptions(
          max_results=2, score_threshold=0.5)
      options = vision.ObjectDetectorOptions(
          base_options=base_options, detection_options=detection_options)
      detector = vision.ObjectDetector.create_from_options(options)

      counter = 0
      while True:
        if q.empty() != True:
            frame = q.get()
            counter += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = vision.TensorImage.create_from_array(rgb_frame)
            detection_result = detector.detect(input_tensor)
            frame = utils.visualize(frame, detection_result, counter)
            cv2.imshow("bird_detector", frame)
        if cv2.waitKey(15) & 0xFF == ord('q'):
          break

if __name__ == '__main__':
    thread1 = threading.Thread(target=initialize_livestream)
    thread2 = threading.Thread(target=display_livestream)
    thread1.start()
    thread2.start()
