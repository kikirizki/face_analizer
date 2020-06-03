import argparse
import cv2
import torch
from emotion import EmotionRecognizer
from detect_face import FaceDetector

parser = argparse.ArgumentParser(description='FaceAnalizer')
parser.add_argument('-m', '--trained_model', default='weights/PlateDetection_epoch_90.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='PASCAL', type=str, choices=['AFW', 'PASCAL', 'FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()

face_detector = FaceDetector("traced_models/face.pt", args)
emotion_detector = EmotionRecognizer("traced_models/emotion.pt", args)

cap = cv2.VideoCapture(0)
device = torch.cuda.current_device()
face_size = (224, 224)
while 1:
    ret, img_raw = cap.read()
    list_of_detections = face_detector.detect_face(img_raw)
    for detection in list_of_detections:
        x1, y1, x2, y2, confidence = detection
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        if confidence < 0.5:
            continue
        detection = list(map(int, detection))
        cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cropped_face = img_raw[x1:x2, y1:y2]
        cropped_face = cv2.resize(cropped_face, face_size)
        cropped_face = torch.tensor(cropped_face).unsqueeze(0)
        list_of_emotions, probability = emotion_detector.detect_emotion(cropped_face)
        for emotion in list_of_emotions:
            img_raw = cv2.putText(img_raw, emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('res', img_raw)
    cv2.waitKey(10)
