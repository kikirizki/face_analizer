import torch
import numpy as np
from data import cfg
from postprocessing.prior_box import PriorBox
from utils.nms import nms
from utils.box_utils import decode

labels = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])


class FaceDetector:
    def __init__(self, path, args, device='cuda'):
        self.device = device
        self.model = torch.jit.load(path)
        self.model.to(device).eval()
        self.args = args

    def detect_face(self, img_raw):
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()
        scale = scale.cuda()

        loc, conf = self.model(img)  # forward pass

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores

        inds = np.where(scores > self.args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)

        keep = nms(torch.tensor(boxes), torch.tensor(scores), overlap=self.args.nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:self.args.keep_top_k, :]
        return dets
