import pyautogui
import pydirectinput
import keyboard
import mss
import torch
import torchvision.transforms as trns
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import time

# initial setups
# control
pydirectinput.FAILSAFE = True
screen_size = pydirectinput.size()
screen_width, screen_height = screen_size
print(f'Screen size: w:{screen_width}, h:{screen_height}')
mouse_last_pos = pydirectinput.position()

# screenshot
monitor_number = 2
sct = mss.mss()
print(f'monitors: {sct.monitors}')
mon = sct.monitors[monitor_number]
monitor = {
    "top": mon["top"],
    "left": mon["left"],
    "width": mon["width"],
    "height": mon["height"],
    "mon": monitor_number,
}

# detect
# labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# load model
model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()
transform = trns.ToTensor()

# functions
def shoot(x, y):
    pydirectinput.move(-2585, -393, 1)
    pydirectinput.click()

def detect(img_raw):
    img_tensor = transform(img_raw)

    # get outputs
    outputs = model([img_tensor])[0]

    # post-process
    outputs = {k: v.detach().numpy() for k, v in outputs.items()}
    threshold = 0.1
    img_show = img_raw
    label_filter = ['person']
    # print('\n\n id |     label      |        score\n------------------------------------------')
    for i, (bbox, label, score) in enumerate(zip(outputs['boxes'], outputs['labels'], outputs['scores'])):
        if score > threshold:
            # print(f'{i:3} - {COCO_INSTANCE_CATEGORY_NAMES[label]:15}: {score}')
            if COCO_INSTANCE_CATEGORY_NAMES[label] in label_filter:
                # draw bbox
                img_show = img_show.astype(np.uint8)
                x0, y0, x1, y1 = bbox
                cv2.rectangle(img_show, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), thickness=1)

                # draw text
                cv2.putText(img_show, f'{i} {COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.3f}',
                    (int(x0), int(y0)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1, cv2.LINE_AA)

                # draw circle
                cv2.circle(img_show,(int(x0+(x1-x0)/2), int(y0+(y1-y0)*0.075)),5, (255, 0, 0), -1)

    # show result
    cv2.imshow('result', img_show)

# timer
cnt = 0
time_start = time.time()

# run
while True:
    cnt += 1
    mouse_current_pos = pydirectinput.position()
    if mouse_current_pos != mouse_last_pos:
        print(mouse_current_pos)
        mouse_last_pos = mouse_current_pos

    if True or keyboard.is_pressed('f'):
        img_cv2 = np.array(sct.grab(monitor))
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGRA2BGR)
        detect(img_cv2)
    
    if keyboard.is_pressed('c'):
        print(f'triggered! @{pydirectinput.position()}')
        shoot(0, 0)
        print(f'done! @{pydirectinput.position()}')

    if cv2.waitKey(1) and keyboard.is_pressed('q'):
        cv2.destroyAllWindows()
        break


# summery
time_total = time.time() - time_start
print(f'Total time: {time_total:.2f} sec')
print(f'Fps: {cnt / time_total:.2f}')
