import argparse
import cv2
import torch
from emotion import EmotionRecognizer
from detect_face import FaceDetector

parser = argparse.ArgumentParser(description='FaceAnalizer')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()

face_detector = FaceDetector("traced_models/face.pt", args)
emotion_detector = EmotionRecognizer("traced_models/emotion.pt", args)


device = torch.cuda.current_device()


class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self):
        # extracting frames
        ret, frame = self.video.read()
        list_of_detections = face_detector.detect_face(frame)
        frame = emotion_detector.recognize_faces(frame, list_of_detections)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


from flask import Flask, render_template, Response

app = Flask(__name__)


@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')

def gen(camera):
    while True:
        # get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') \

@app.route('/video_vid')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0', port='5000', debug=True)
