import numpy as np

from experiments.yolo.yolo import Model
import yaml
import torch
import cv2

from utils.yolo.general import non_max_suppression


with open("txts/coco_label.txt") as fd:
    labels = fd.readlines()

conf_thres = 0.001  # confidence threshold
iou_thres = 0.6  # NMS IoU threshold
device = torch.device("cuda")
model = Model("checkpoints/yolo/yolov5s/yolov5s.yaml", nc=80)

sd = torch.load("checkpoints/yolo/yolov5s/yolov5s_0.pth", map_location="cpu")
model.load_state_dict(sd)
model.to(device)
model.eval()


img = cv2.imread(r"D:\datasets\300w\300w\02_Outdoor\outdoor_015.png")
h, w, c = img.shape
net_in = np.zeros((640, 640, c))
scale = max(h, w) / 640
l = t = r = b = 0
if scale > 1.0:
    if h > w:
        h = 640
        w = int(w / scale)
        l = (640 - w) // 2
        net_in[:, l:l+w] = cv2.resize(img, (w,h))
    else:
        w = 640
        h = int(h / scale)
        t = (640 - h) // 2
        net_in[t:t+h, :] = cv2.resize(img, (w,h))

net_in = net_in.astype(float) / 255.0
net_in = np.transpose(net_in, (2,0,1))[None, ...]
out, train_out = model(torch.from_numpy(net_in).float().to(device))

with torch.no_grad():
    out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)

    for si, pred in enumerate(out):
        pred = pred.cpu().numpy()
        pred[:, 0:4] -= [l, t, l, t]
        pred[:, 0:4] *= scale

        for predn in pred:
            bbox = predn[:4].astype(int)
            conf = predn[4]
            c = int(predn[5])
            c = labels[c]
            x1, y1, x2, y2 = bbox
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0))
            img = cv2.putText(img, c, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255))
            if conf < 0.9:
                break
    cv2.imwrite("temp.jpg", img)