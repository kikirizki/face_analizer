import torch
import numpy as np
import cv2
labels = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])


class EmotionRecognizer:
    def __init__(self, path, args, device='cuda'):
        self.args = args
        self.device = device
        self.model = torch.jit.load(path)
        self.face_size = (224, 224)
        self.model.to(device).eval()
    def detect_emotion(self, faces):
        if len(faces) > 0:
            faces = faces.permute(0, 3, 1, 2)
            faces = faces.float().div(255).to(self.device)
            emotions = self.model(faces)
            prob = torch.softmax(emotions, dim=1)
            emo_prob, emo_idx = torch.max(prob, dim=1)
            return labels[emo_idx.tolist()], emo_prob.tolist()
        else:
            return 0, 0
    def recognize_faces(self,img_raw,list_of_detections):
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
            try:
                cropped_face = img_raw[x1:x2, y1:y2]
                cropped_face = cv2.resize(cropped_face, self.face_size)
                cropped_face = torch.tensor(cropped_face).unsqueeze(0)
                list_of_emotions, probability = self.detect_emotion(cropped_face)
                for emotion in list_of_emotions:
                    img_raw = cv2.putText(img_raw, emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                          1, (255, 255, 255), 1, cv2.LINE_AA)
            except:
                print("No face")
            return img_raw

