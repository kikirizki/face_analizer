import torch
import numpy as np

labels = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
class EmotionRecognizer:

    def __init__(self, path, device='cuda'):
        self.device = device
        self.model = torch.jit.load(path)
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