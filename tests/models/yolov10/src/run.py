import cv2
import torch
import supervision as sv
from yolov10 import YOLOv10
model = YOLOv10('./weights/yolov10n.pt') # wget -P ./weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
image = cv2.imread(f'./data/dog.jpeg') # wget -P ./data -q https://media.roboflow.com/notebooks/examples/dog.jpeg
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
cv2.imwrite('./data/annotated_dog.jpeg', annotated_image)
