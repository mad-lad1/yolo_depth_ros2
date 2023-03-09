# running yolov7 with webcam input and pytorch
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.torch_utils import TracedModel
from utils.general  import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load('yolov7.pt', map_location=device)  
stride = int(model.stride.max())  # model stride
imgz = 640  # inference size (pixels)
model = TracedModel(model, device, 640)
#  get model names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


# load the webcam
cap = cv2.VideoCapture(0)
model(torch.zeros(1, 3, imgz, imgz).to(device).type_as(next(model.parameters())))
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 640))
    image = frame.copy()
    frame = torch.from_numpy(frame).to(device)
    frame = frame.float()  # uint8 to fp16/32
    frame /= 255.0  # 0 - 255 to 0.0 - 1.0
    # reformat to N, C, W, H
    frame = frame.permute(2, 0, 1)
    if frame.ndimension() == 3:
        frame = frame.unsqueeze(0)
    
    with torch.no_grad():
        pred = model(frame, augment=False)[0]
    
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    for i, det in enumerate(pred):  # detections per image

        if det is not None and len(det):
            det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], image.shape).round()
    
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, image, label=label, color=colors[int(cls)], line_thickness=3)    

    
        cv2.imshow('frame', image)
        cv2.waitKey(1)
    

