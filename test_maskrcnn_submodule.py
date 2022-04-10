import pickle
from collections import OrderedDict

import numpy as np
import cv2
import torch
import yaml

from experiments.detectron2 import build_model
from utils.config import load_config_file, dump_config_to_file
from utils.config import Config
import os

from utils.detectron2.pkl2pth import align_and_update_state_dicts

dir = os.path.abspath(__file__)
dir = os.path.dirname(dir)
EXP_DIR = f"{dir}/checkpoints/RCNN-FPN"

cfg_file = "checkpoints/RCNN-FPN/mask_rcnn_R_50_FPN_3x.yaml"
cfg_dict = load_config_file(cfg_file, EXP_DIR=EXP_DIR)
config=Config(cfg_dict, EXP_DIR=EXP_DIR)
dump_config_to_file(config.dump(), "temp.yaml")

model = build_model(config)

PKL = "checkpoints/model_final_f10217.pkl"
PTH = PKL.replace(".pkl", ".pth")
with open(PKL, "rb") as fd:
    data = pickle.load(fd)
with open("keys.csv", "w") as fd:
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(data['model'].keys())
    keys = [["None", "None"] for i in range(max(len(model_keys), len(ckpt_keys)))]
    for i, k in enumerate(model_keys):
        keys[i][0] = k
    for i, k in enumerate(ckpt_keys):
        keys[i][1] = k
    for pairs in keys:
        fd.write(f"{','.join(pairs)}\n")
state_dict = align_and_update_state_dicts(model.state_dict(), data['model'], False)

sd = OrderedDict()
for k, v in model.state_dict().items():
    sd[k] = torch.from_numpy(state_dict[k])
state_dict = sd

torch.save(state_dict, PTH)

model.load_state_dict(state_dict)
model.to(torch.device("cuda"))
model.eval()

img = cv2.imread(r"D:\datasets\300w\300w\02_Outdoor\outdoor_015.png")
h, w = img.shape[:2]
scale = min(h, w) / 800
nh = int(h / scale + 0.5)
nw = int(w / scale + 0.5)
img = cv2.resize(img, (nw, nh))
input_img = img.astype(np.float32)
input_img = np.transpose(input_img, (2,0,1))
input_img = torch.from_numpy(input_img)

with open(f"txts/coco_label.txt", "r") as fd:
    labels = [_.strip() for _ in fd.readlines()]

with torch.no_grad():
    ret = model([{"image":input_img}])
    instances = ret[0]["instances"]
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
    # labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
    # keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
    for bbox, cls, score in zip(boxes, classes, scores):
        # instance = instances[i]         # dict_keys(['pred_boxes', 'scores', 'pred_classes', 'pred_masks'])
        # instance = instances.get_fields()
        # print(instances[i].get_fields().keys())
        bbox= bbox.cpu().numpy().astype(int)
        l, t, r, b = bbox.tolist()
        img = cv2.rectangle(img, (l,t), (r,b), (255,0,0), 3)
        idx = int(cls)
        img = cv2.putText(img, labels[idx], (l,t), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255))
    print("XXXXXXXXXX", img.shape)
    cv2.imwrite("temp.jpg", img)
    # for bbox in ret:
    #     print(bbox)
    #     if not bbox[0]:
    #         break
    #     print(bbox)

