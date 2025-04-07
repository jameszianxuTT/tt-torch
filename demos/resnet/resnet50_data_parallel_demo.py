# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import tabulate
from threading import Thread
import requests
from tt_torch.tools.device_manager import DeviceManager

# A custom thread class to simplify returning values from threads
class CustomThread(Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None
    ):
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return


def main():
    """
    This demo shows how to run the ResNet model on all available devices on the board in parallel.
    """
    weights = models.ResNet152_Weights.IMAGENET1K_V2
    model = models.resnet152(weights=weights).to(torch.bfloat16).eval()
    classes = weights.meta["categories"]
    preprocess = weights.transforms()

    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = {}
    options["compiler_config"] = cc
    devices = (
        DeviceManager.get_available_devices()
    )  # Get all available devices on board
    tt_models = []
    for device in devices:
        options["device"] = device
        # Compile the model for each device
        tt_models.append(
            torch.compile(model, backend=backend, dynamic=False, options=options)
        )

    headers = ["Top 5 Predictions"]
    topk = 5
    prompt = 'Enter the path of the image (type "stop" to exit or hit enter to use a default image): '
    img_path = input(prompt)
    while img_path != "stop":
        if img_path == "":
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            img = Image.open(requests.get(url, stream=True).raw)
        elif img_path.startswith("http"):
            img = Image.open(requests.get(img_path, stream=True).raw)
        else:
            img = Image.open(img_path)
        img = preprocess(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(torch.bfloat16)

        # Send image to all devices
        threads = []
        for i in range(len(tt_models)):
            tt_model = tt_models[i]
            device_used = devices[i]
            thread = CustomThread(
                target=torch.topk, args=(tt_model(img).squeeze().softmax(-1), topk)
            )
            threads.append((device_used, thread))

        for _, thread in threads:
            thread.start()

        for device_used, thread in threads:
            top5, top5_indices = thread.join()
            tt_classes = []
            for class_likelihood, class_idx in zip(
                top5.tolist(), top5_indices.tolist()
            ):
                tt_classes.append(f"{classes[class_idx]}: {class_likelihood}")

            rows = []
            for i in range(topk):
                rows.append([tt_classes[i]])

            print(f"Results from Device: {device_used}")
            print(tabulate.tabulate(rows, headers=headers))
            print()

        img_path = input(prompt)
    DeviceManager.release_devices()  # Release all devices after use


if __name__ == "__main__":
    main()
