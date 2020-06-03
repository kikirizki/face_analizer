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

net = FaceDetector("traced_models/face.pt", args)
emotion_detector = EmotionRecognizer("traced_models/emotion.pt", args)

cap = cv2.VideoCapture(0)
device = torch.cuda.current_device()

while 1:
    ret, img_raw = cap.read()
    detections = FaceDetector.detect_face(img_raw)

    # show image
    # if args.show_image:
    for b in detections:
        if b[4] < 0.5:
            continue

        b = list(map(int, b))

        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        y = b[0]
        x = b[1]
        h = b[2] - b[0]
        w = b[3] - b[1]
        face_only = img_raw[x:x + w, y:y + h]
        face_only = cv2.resize(face_only, (224, 224))
        face_only = torch.tensor(face_only).unsqueeze(0)
        list_of_emotions, probab = emotion_detector.detect_emotion(face_only)
        for emotion in list_of_emotions:
            img_raw = cv2.putText(img_raw, emotion, (y, x), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('res', img_raw)
    cv2.waitKey(10)
