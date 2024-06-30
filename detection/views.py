# detection/views.py
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import threading
import numpy as np
import os
from datetime import datetime
from playsound import playsound

# Load YOLO model
net = cv2.dnn.readNet("static/yolov3.weights", "static/yolov3.cfg")

# Load class names
with open("static/coco.names", "r") as f:
    classes = f.read().strip().split("\n")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Configuration
config = {
    'known_width': {
        'car': 200,
        'person': 170,
        'truck': 300,
        'motorbike': 100,
        'bus': 300,
        'bicycle': 100,
        'traffic light': 100,
        'traffic sign': 100,
        'stop sign': 100
    },
    'focal_length': 800,
    'roi_x': 300,
    'roi_y': 240,
    'roi_width': 800,
    'roi_height': 800,
    'initial_frame': 0,
    'confidence_threshold': 0.5,
    'distance_threshold': 1000,
    'beep_path': r"static/beep.mp3",
    'log_file': 'detection_log.txt',
    'output_video': 'output.avi',
    'cam': "0"
}

paused = [False]
video_camera = None  # Global video camera object


def distance_to_camera(known_width, focal_length, per_width, label):
    if label in known_width.keys():
        return (known_width[label] * focal_length) / per_width
    else:
        return (50 * focal_length) / per_width

def play_beep():
    if os.path.exists(config['beep_path']):
        def play_sound():
            playsound(config['beep_path'])
        threading.Thread(target=play_sound).start()
    else:
        print("Error: beep.mp3 file not found at the specified path:", config['beep_path'])

def log_detection(label, distance):
    with open(config['log_file'], 'a') as f:
        f.write(f"{datetime.now()} - Object: {label}, Distance: {distance:.2f} cm\n")

class VideoCamera:
    def __init__(self):
        self.video = None
        self.setup_camera()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        if self.video is not None:
            self.video.release()

    def setup_camera(self):
        if self.video is not None:
            self.video.release()
        if config['cam'] == 0 or config['cam'] == '0':
            self.video = cv2.VideoCapture(0)
        else:
            self.video = cv2.VideoCapture(str(config['cam']))
        self.grabbed, self.frame = self.video.read()

    def get_frame(self):
        if not self.grabbed or self.frame is None:
            print('A')
            return None
        image = self.frame
        if not paused[0]:
            roi_frame = image[config['roi_y']:config['roi_y'] + config['roi_height'], config['roi_x']:config['roi_x'] + config['roi_width']]
            height, width, _ = roi_frame.shape

            blob = cv2.dnn.blobFromImage(roi_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            objects_distances = {}
            nearby_objects = {}

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > config['confidence_threshold']:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        label = str(classes[class_id])

                        distance = distance_to_camera(config['known_width'], config['focal_length'], w, label)
                        objects_distances[label] = distance

                        if distance < config['distance_threshold']:
                            nearby_objects[label] = distance
                            cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            play_beep()
                            log_detection(label, distance)
                        else:
                            cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        cv2.putText(roi_frame, f"{label}: {distance:.2f} cm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            _, jpeg = cv2.imencode('.jpg', roi_frame)
            return jpeg.tobytes()
        else:
            _, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()

    def update(self):
        while True:
            self.grabbed, self.frame = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    global video_camera
    if not video_camera:
        video_camera = VideoCamera()
    else:
        video_camera.setup_camera()
    return StreamingHttpResponse(gen(video_camera), content_type='multipart/x-mixed-replace; boundary=frame')

def first(request):
    return render(request, 'detection/first.html')

def index(request):
    context = {'config': config}
    return render(request, 'detection/index.html', context)

def toggle_pause(request):
    paused[0] = not paused[0]
    return JsonResponse({'paused': paused[0]})

def get_log(request):
    if not os.path.exists(config['log_file']):
        return JsonResponse({'log': []})
    with open(config['log_file'], 'r') as f:
        logs = f.readlines()
    return JsonResponse({'log': logs})

def update_config(request):
    try:
        config['focal_length'] = int(request.GET.get('focal_length'))
        config['confidence_threshold'] = float(request.GET.get('confidence_threshold'))
        config['distance_threshold'] = float(request.GET.get('distance_threshold'))
        config['roi_x'] = int(request.GET.get('roi_x'))
        config['roi_y'] = int(request.GET.get('roi_y'))
        config['roi_width'] = int(request.GET.get('roi_width'))
        config['roi_height'] = int(request.GET.get('roi_height'))
        config['cam'] = str(request.GET.get('cam'))
        if video_camera:
            video_camera.setup_camera()
        return JsonResponse({'status': 'success'})
    except ValueError:
        return JsonResponse({'status': 'failed'})
