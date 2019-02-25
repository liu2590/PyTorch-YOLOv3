from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


os.chdir(r'D:\ML\2src\vscode\PyTorch-YOLOv3')
print('\n开始了------->\n\n')
CUDA = torch.cuda.is_available()
print(CUDA, '\n', os.getcwd())
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
classes = load_classes('data/coco.names')  # Extracts class labels from file

# 模型 权重 Set up model
model = Darknet('config/yolov3.cfg')
# model.load_weights(r'D:\ML\data\yolo_weight\ls.weights')
model.load_weights(r'D:\ML\data\yolo_weight\yolov3.weights')

if CUDA:
    model.cuda()

model.eval()  # Set in evaluation mode

# model.net_info["height"] = 160
# inp_dim = int(model.net_info["height"])
inp_dim = 480

assert inp_dim % 32 == 0
assert inp_dim > 32

cap = cv2.VideoCapture(0)
assert cap.isOpened(), '打不开'
frames = 0
colors = (100, 222, 6)
start = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        img, _, _ = prep_image(frame, inp_dim)
        print(img.shape,img.shape)
        img = Variable(img.type(Tensor))
        with torch.no_grad():
            output = model(img)
            output = non_max_suppression(output, 80)
            # output = write_results(output, 0.25, 80, nms=True, nms_conf=0.4)
        frames += 1
        print("({:3d}) 图片 FPS: {:5.2f}".format(frames,
                                               frames / (time.time() - start)))
        # Iterate through images and save plot of detections
        for img_i, detections in enumerate(output):

            # The amount of padding that was added
            pad_x = max(img.shape[0] - img.shape[1],
                        0) * (416 / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0],
                        0) * (416 / max(img.shape))
            # Image height and width after padding is removed
            unpad_h = 416 - pad_y
            unpad_w = 416 - pad_x
            # print(unpad_h,unpad_w)
            # Draw bounding boxes and labels of detections
            if detections is not None:
                # unique_labels = detections[:, -1].cpu().unique()
                # n_cls_preds = len(unique_labels)
                # bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    ls = '%s,%.2f' % (classes[int(cls_pred)],
                                        cls_conf.item())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, ls, (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 155),
                                1, cv2.LINE_AA)
                    print('\t+ 标签: %10s, 可靠性: %.2f' % (classes[int(cls_pred)],
                                                       cls_conf.item()))

        cv2.imshow("图像", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
