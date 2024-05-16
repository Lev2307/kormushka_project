import os
import cv2
import numpy as np
import random
from datetime import datetime, timedelta

from tflite_support.task import processor
import yadisk

_MARGIN = 10
_ROW_SIZE = 10
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 256) # Цвет текста в BGR
BORDER_COLOR = (114, 128, 250) # Цвет границ изобрадения в BGR 

BIRDS_FOLDER = '/birds/'
TIME_BREAK = 900
LAST_IMAGE_DATETIME_FILE="last_image_datetime_file.txt"
SYMBOLS_FOR_MAKING_RANDOM_FILE_NAME_SUFFIX = 'zwertyuiopasdfghjklzxcvbnmZQWERTYUIOPASDFGHJKLXCVBNM123456789'
    
def upload_image_to_yadisk(image_path):
    # загрузка изображения на яндекс диск
    disk = yadisk.YaDisk(token=SECRET_TOKEN) # вместо SECRET_TOKEN используете токен выданный апи яндекс диска
    if not disk.exists(BIRDS_FOLDER):
        disk.mkdir(BIRDS_FOLDER) # создание папки если такой нет на диске
    disk.upload(image_path, BIRDS_FOLDER+image_path) # загрузка изображения в папку

def get_difference_in_seconds(now, file_path):
    # просмотр файла last_image_datetime_file.txt и получение разницы в секундах с момента последней отправки изображения на яндекс диск
    with open(file_path, 'r+') as f:
        file_content = f.read()
        if file_content == "":
            one_hour_earlier_time = (now - timedelta(hours=1))
            time_dif = (now - one_hour_earlier_time).total_seconds()
            one_hour_earlier_time = datetime.strftime(one_hour_earlier_time, "%Y-%m-%d %H:%M:%S")
            f = f.write(one_hour_earlier_time)
        else:
            file_content_time = datetime.strptime(file_content, "%Y-%m-%d %H:%M:%S")
            time_dif = (now - file_content_time).total_seconds()
            f.close()
    return time_dif

def create_image(img_path, image):
    # сохранение изображение локально для его отправки на облачный сервис с последующим его удалением
    cv2.imwrite(img_path, image)
    upload_image_to_yadisk(img_path)
    os.remove(img_path)
    
def visualize(image: np.ndarray, detection_result: processor.DetectionResult, frame_number) -> np.ndarray:
    # отрисовка на изображении найденных объектов
    prev_frame = 0
    now = datetime.now().replace(microsecond=0)
    time_difference_in_seconds = get_difference_in_seconds(now, LAST_IMAGE_DATETIME_FILE)
    all_probabilities = []
    for detection in detection_result.detections:
        # получение найденных объектов с фрэйма
        category = detection.categories[0]
        probability = round(category.score, 2)
        result_text = str(probability)
        all_probabilities.append(probability)

        # отрисовка границ найденных объектов
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, BORDER_COLOR, 3)
        text_location = (bbox.origin_x + (bbox.width - 52), _MARGIN + + _ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    
    
    if all_probabilities:
        if sum(all_probabilities) / len(all_probabilities) >= 0.5 and prev_frame != frame_number: # сравнение средней вероятности всех найденных объектов со определённым значением ( 0.5 - 50% )
            if time_difference_in_seconds > TIME_BREAK: # Если с момента последней отправки изображения прошло 15 минут
                random_file_name_suffix = "".join(random.choices(SYMBOLS_FOR_MAKING_RANDOM_FILE_NAME_SUFFIX, k=10))
                img_path = 'bird_picture' + f'_{random_file_name_suffix}' + '.jpg'
                create_image(img_path, image) # создание изображения
                with open(LAST_IMAGE_DATETIME_FILE, 'w') as f:
                    f.write(datetime.strftime(now, "%Y-%m-%d %H:%M:%S")) # запись нового значения времени в файл
                    f.close()
                prev_frame = frame_number
    return image
