import os
import cv2
import numpy as np
import random
from datetime import datetime, timedelta

from tflite_support.task import processor
import yadisk

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 256) # BGR
BORDER_COLOR = (114, 128, 250)

BIRDS_FOLDER = '/birds/'
TIME_BREAK = 900
LAST_IMAGE_DATETIME_FILE="last_image_datetime_file.txt"
SYMBOLS_FOR_MAKING_RANDOM_FILE_NAME_SUFFIX = 'zwertyuiopasdfghjklzxcvbnmZQWERTYUIOPASDFGHJKLXCVBNM123456789'
    
def upload_image_to_yadisk(image_path):
    disk = yadisk.YaDisk(token=SECRET_TOKEN) # SECRET_TOKEN is token given by yadisk api
    if not disk.exists(BIRDS_FOLDER):
        disk.mkdir(BIRDS_FOLDER)
    disk.upload(image_path, BIRDS_FOLDER+image_path)
    

def get_difference_in_seconds(now, file_path):
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
    cv2.imwrite(img_path, image)
    upload_image_to_yadisk(img_path)
    os.remove(img_path)
    
def visualize(image: np.ndarray, detection_result: processor.DetectionResult, frame_number) -> np.ndarray:
    prev_frame = 0
    now = datetime.now().replace(microsecond=0)
    time_difference_in_seconds = get_difference_in_seconds(now, LAST_IMAGE_DATETIME_FILE)
    all_probabilities = []
    for detection in detection_result.detections:
        # labels
        category = detection.categories[0]
        category_name = category.category_name
        if category_name == "hama_burung":
            category_name = "bird"
        probability = round(category.score, 2)
        result_text = str(probability)
        #"bird"
        #f"{category_name}"
        all_probabilities.append(probability)

        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, BORDER_COLOR, 3)
        text_location = (bbox.origin_x + (bbox.width - 52), _MARGIN + + _ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    
    if all_probabilities:
        if sum(all_probabilities) / len(all_probabilities) >= 0.45 and prev_frame != frame_number:
            if time_difference_in_seconds > TIME_BREAK:
                random_file_name_suffix = "".join(random.choices(SYMBOLS_FOR_MAKING_RANDOM_FILE_NAME_SUFFIX, k=10))
                img_path = 'bird_picture' + f'_{random_file_name_suffix}' + '.jpg'
                create_image(img_path, image)
                with open(LAST_IMAGE_DATETIME_FILE, 'w') as f:
                    new = f.write(datetime.strftime(now, "%Y-%m-%d %H:%M:%S"))
                    f.close()
                prev_frame = frame_number
    return image
