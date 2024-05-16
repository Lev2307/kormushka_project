from flask import Flask, render_template, Response, request 
import cv2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

CAMERA_RTSP_URL = "rtsp://admin_camera:camera_password@192.168.152.105:554/stream2"

bird_detection = 'bird_trained_models/bird_detection/birds_model_lite1.tflite' 
bird_classification = 'bird_trained_models/bird_classification/10_birds_species_model_lite1.tflite'
first_retrained_bird_detection_model = "bird_trained_models/retrained_models_for_detection/retrain1_birds_detection_lite0.tflite"
second_retrained_bird_detection_model = "bird_trained_models/retrained_models_for_detection/retrain2_birds_detection_lite0.tflite"

app = Flask(__name__)

cap = cv2.VideoCapture(CAMERA_RTSP_URL)

def generate_frames():
	base_options = core.BaseOptions(file_name=first_retrained_bird_detection_model, num_threads=4)
	detection_options = processor.DetectionOptions(max_results=1, score_threshold=0.4)
	options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
	detector = vision.ObjectDetector.create_from_options(options)
	counter = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		counter += 1
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		input_tensor = vision.TensorImage.create_from_array(rgb_frame)
		detection_result = detector.detect(input_tensor)
		frame = utils.visualize(frame, detection_result, counter)
		buffer = cv2.imencode('.jpg',frame)[1]
		frame = buffer.tobytes()
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
