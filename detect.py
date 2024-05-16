import sys
import time
import threading
import queue

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

CAMERA_RTSP_URL = "" # Ссылка для подключения к ip камере, либо номер порта usb камеры

# 4 работающие модели для обнаружения птиц внутри кормушки
bird_detection = 'tf_bird_trained_models/bird_detection/birds_model_lite1.tflite' 
bird_classification = 'tf_bird_trained_models/bird_classification/10_birds_species_model_lite1.tflite'
first_retrained_bird_detection_model = "tf_bird_trained_models/retrained_models_for_detection/retrain1_birds_detection_lite0.tflite"
second_retrained_bird_detection_model = "tf_bird_trained_models/retrained_models_for_detection/retrain2_birds_detection_lite0.tflite"

# в проекте используется мультипоточность ( при помощи библиотек threading, queue ) для более качественной работы OpenCV
q = queue.Queue() 

def initialize_livestream():    
    # запуск видео с камеры
    cap = cv2.VideoCapture(CAMERA_RTSP_URL)
    while cap.isOpened():
        ret, frame = cap.read() # считывание одного фрэйма
        q.put(frame) # сохранение его в очередь

def display_livestream():
      # инициализация классов нейронной сети для обработки видео
      base_options = core.BaseOptions(
          file_name=first_retrained_bird_detection_model, num_threads=4)
      detection_options = processor.DetectionOptions(
          max_results=2, score_threshold=0.5)
      options = vision.ObjectDetectorOptions(
          base_options=base_options, detection_options=detection_options)
      detector = vision.ObjectDetector.create_from_options(options)

      counter = 0 # подсчёт фрэймов для корректной отправки фото на яндекс диск
      while True:
        if q.empty() != True: 
            frame = q.get() # получение фрэйма из очереди
            counter += 1

            # обработка полученного изображения
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = vision.TensorImage.create_from_array(rgb_frame)
            detection_result = detector.detect(input_tensor) 
            frame = utils.visualize(frame, detection_result, counter)

            cv2.imshow("bird_detector", frame) # вывод изображения на экран
        if cv2.waitKey(15) & 0xFF == ord('q'):
          break

if __name__ == '__main__':
    thread1 = threading.Thread(target=initialize_livestream)
    thread2 = threading.Thread(target=display_livestream)
    thread1.start()
    thread2.start()
