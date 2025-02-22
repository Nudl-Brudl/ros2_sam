import cv2
import numpy as np
from cv_bridge import CvBridge
from colorama import Fore, Back, Style

from rclpy.node import Node

from ros2_sam.sam import SAM
from ros2_sam.utils import SAMDownloader
from ros2_sam_msgs.srv import Segmentation as SegmentationSrv


class SAMServer(Node):
    def __init__(self, node_name: str = "sam_server") -> None:
        super().__init__(node_name)
        self.my_logger("Starting SAM server...")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("checkpoint_dir", ""),
                ("model_type", "vit_h"),
                ("device", "cuda"),
            ],
        )
        self._bridge = CvBridge()
        self._checkpoint_dir = self.get_parameter("checkpoint_dir").value
        self._model_type = self.get_parameter("model_type").value
        self._device = self.get_parameter("device").value

        if not self._checkpoint_dir:
            sam_downloader = SAMDownloader()
            self.my_logger("Checking model availability...")
            if not sam_downloader.check_model_availability(self._model_type):
                self.my_logger("Model not available.")
                self.my_logger(
                    f"Downloading model {self._model_type} from " + 
                    f"{sam_downloader.model_url_dict[self._model_type]} " + 
                    f"to {sam_downloader.checkpoint_dir}..."
                )
                if not sam_downloader.download(self._model_type):
                    raise RuntimeError("Failed to download model.")
            self._checkpoint_dir = sam_downloader.checkpoint_dir
            self.my_logger("Model available.")

        self.my_logger(
            f"Loading SAM model '{self._model_type}' from " + 
            f"'{self._checkpoint_dir}'. This may take some time..."
        )

        self._sam = SAM(
            checkpoint_dir=self._checkpoint_dir,
            model_type=self._model_type,
            device=self._device,
        )
        self.my_logger("SAM model loaded.")
        self._sam_segment_service = self.create_service(
            SegmentationSrv, "~/segment", self._on_segment
        )
        self.my_logger("SAM server is ready.")

    def _on_segment(
        self, req: SegmentationSrv.Request, res: SegmentationSrv.Response
    ) -> SegmentationSrv.Response:
        self.my_logger("Received segmentation request.")
        try:
            img = cv2.cvtColor(self._bridge.imgmsg_to_cv2(req.image), 
                               cv2.COLOR_BGR2RGB)
            points = np.vstack([(p.x, p.y) for p in req.query_points])
            boxes = (
                np.asarray(req.boxes.data).reshape((len(req.boxes.data) // 4, 4))[0]
                if len(req.boxes.data) > 0
                else None
            )
            labels = np.asarray(req.query_labels)

            self.my_logger(
                f"Segmenting image of shape {img.shape} with pixel " +
                f"prior: {[[point[0], point[1]] for point in points]}"
            )
            start = self.get_clock().now().nanoseconds
            masks, scores, logits = self._sam.segment(
                img, points, labels, boxes, req.multimask
            )
            self.my_logger(
                f"Segmentation completed in " + 
                f"{round((self.get_clock().now().nanoseconds - start)/1.e9, 2)}s."
            )

            res.masks = [self._bridge.cv2_to_imgmsg(m.astype(np.uint8)) 
                         for m in masks]
            res.scores = scores.tolist()
            if req.logits:
                res.logits = [self._bridge.cv2_to_imgmsg(l) for l in logits]
            return res
        except Exception as e:
            err_msg = f"Failure during service call. Full message: {e}."
            self.get_logger().error(err_msg)
            raise RuntimeError(err_msg)
        

    def my_logger(self, message: str):
        '''
        My custom logger
        '''
        self.get_logger().info(Fore.CYAN + message + Style.RESET_ALL)
