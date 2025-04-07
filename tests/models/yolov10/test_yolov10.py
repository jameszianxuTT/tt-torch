# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/THU-MIG/yolov10
import sys
import os
import cv2
import torch
import pytest
from pathlib import Path
import requests
# import urllib.request
import supervision as sv
from tests.utils import ModelTester, flatten_tensor_lists
from tt_torch.tools.utils import CompilerConfig, CompileDepth, OpByOpBackend
sys.path.append("tests/models/yolov10/src")
from src.ultralytics import YOLOv10
sys.path.append("tests/models/yolov10/src")

class ThisTester(ModelTester):
    def _load_model(self):
        # Download model weights
        url = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt"

        download_dir = Path.home() / ".cache/yolov10_weights"
        download_dir.mkdir(parents=True, exist_ok=True)

        load_path = download_dir / url.split("/")[-1]
        if not load_path.exists():
            response = requests.get(url, stream=True)
            with open(str(load_path), "wb") as f:
                f.write(response.content)

        # Load model
        model = YOLOv10(load_path)
        # model.load_state_dict(
        #     torch.load(
        #         str(load_path),
        #         map_location=torch.device("cpu"),
        #     )
        # )
        return model

    def _load_inputs(self):
        # Image preprocessing
        image_url = (
            "https://media.roboflow.com/notebooks/examples/dog.jpeg"
        )
        download_dir = Path.home() / ".cache/yolov10_data"
        download_dir.mkdir(parents=True, exist_ok=True)

        load_path = download_dir / image_url.split("/")[-1]
        if not load_path.exists():
            response = requests.get(image_url, stream=True)
            with open(str(load_path), "wb") as f:
                f.write(response.content)
        self.image = cv2.imread(load_path)
        return self.image
    
    def _extract_outputs(self, output_object):
        boxes = output_object.boxes
        box_data = (
            boxes.xyxy,    # Bounding box coordinates in (x1, y1, x2, y2) format
            boxes.cls,     # Class indices
            boxes.conf     # Confidence scores
        )
        return box_data

@pytest.mark.parametrize(
    "mode",
    ["eval"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)

def test_yolov10(record_property, mode, op_by_op):
    model_name = "YOLOv10"

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True
    if op_by_op:
        cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        if op_by_op == OpByOpBackend.STABLEHLO:
            cc.op_by_op_backend = OpByOpBackend.STABLEHLO
    cc.verify_op_by_op = True
    tester = ThisTester(
        model_name,
        mode,
        assert_pcc=False,
        assert_atol=False,
        compiler_config=cc,
        record_property_handle=record_property,
    )
    results = tester.test_model()[0]
    breakpoint()
    if mode == "eval":
        # res = results[0]
        detections = sv.Detections.from_ultralytics(results)
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = bounding_box_annotator.annotate(scene=tester.image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        sv.plot_image(annotated_image)
        cv2.imwrite("tests/models/yolov10/data/annotated_dog.jpeg", annotated_image)
    tester.finalize()


# # Create directories if they don't exist
# os.makedirs("tests/models/yolov10/weights", exist_ok=True)
# os.makedirs("tests/models/yolov10/data", exist_ok=True)

# # Define paths
# weights_path = "tests/models/yolov10/weights/yolov10n.pt"
# image_path = "tests/models/yolov10/data/dog.jpeg"

# # Download weights file if it doesn't exist
# if not os.path.exists(weights_path):
#     print("Downloading YOLOv10 weights...")
#     urllib.request.urlretrieve(
#         "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
#         weights_path,
#     )

# # Download image file if it doesn't exist
# if not os.path.exists(image_path):
#     print("Downloading sample image...")
#     urllib.request.urlretrieve(
#         "https://media.roboflow.com/notebooks/examples/dog.jpeg", image_path
#     )
# model = YOLOv10(weights_path)
# image = cv2.imread(image_path)
# results = model(image)[0]
# detections = sv.Detections.from_ultralytics(results)

# bounding_box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
# annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# sv.plot_image(annotated_image)
# cv2.imwrite("tests/models/yolov10/data/annotated_dog.jpeg", annotated_image)