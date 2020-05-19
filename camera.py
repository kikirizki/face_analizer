import torch
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
# from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode
# from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
import numpy as np
import torch
import argparse

from data import cfg
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from utils.nms_wrapper import nms

parser = argparse.ArgumentParser(description='FaceBoxes')

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

resize = 1
net = torch.jit.load("traced_models/face.pt")

cap = cv2.VideoCapture(0)
device = torch.cuda.current_device()
while 1:
    ret, img_raw = cap.read()
    #img_raw = cv2.imread("14_5_2020__13_47_30_raw.jpg")
    img = np.float32(img_raw)
    if resize != 1:
        img = cv2.resize(np.float32(img), None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.cuda()
    scale = scale.cuda()

    loc, conf = net(img)  # forward pass
    print("loc shape {}".format(loc.shape))
    print("conf shape {}".format(conf.shape))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    print(scale)
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    # keep = py_cpu_nms(dets, args.nms_threshold)
    keep = nms(dets, args.nms_threshold, force_cpu=args.cpu)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]

    # show image
    # if args.show_image:
    for b in dets:
        if b[4] < 0.5:
            continue

        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        cx = b[0]
        cy = b[1] + 12
    # break
    cv2.imshow('res', img_raw)
    cv2.waitKey(10)
