import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ament_index_python import get_package_share_directory

from ros2_sam.sam_client import SAMClient
from ros2_sam.utils import show_box, show_mask, show_points


original = False


def main(args: List[str] = None) -> None:
    rclpy.init(args=args)

    sam_client = SAMClient(
        node_name="sam_client",
        service_name="sam_server/segment",
    )

    try:
        if original:
            img_path = os.path.join(
                get_package_share_directory("ros2_sam"), "data/car.jpg")
            points = np.array([[1035, 640], [1325, 610]])
            labels = np.array([0, 0])
            boxes = np.asarray([[54, 350, 1700, 1300]])
        else:
            root_dir = os.path.abspath(os.path.dirname(__file__))
            img_path = os.path.join(root_dir,
                                    os.path.pardir, 
                                    os.path.pardir, 
                                    os.path.pardir,
                                    os.path.pardir,
                                    os.path.pardir,
                                    "gigaPose_datasets",
                                    "datasets",
                                    "custom",
                                    "camera_data",
                                    "scene",
                                    "image",
                                    "scene.png")
            points = np.array([[538, 334]])
            labels = np.array([1])
            boxes = None#np.asarray([[516, 130, 576, 323]])#, [633, 125, 708, 330]])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks, scores = sam_client.sync_segment_request(
            image, points, labels, boxes=boxes
        )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(mask, plt.gca(), color=(255, 200, 40, 150))
            if points is not None:
                show_points(points, labels, plt.gca())
            if boxes is not None:
                for box in boxes:
                    show_box(box, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.show()
            
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
