# built-in dependencies
import os
from typing import List, Any, Union
from enum import Enum

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger
from deepface.commons import weight_utils

logger = Logger()


class YoloModel(Enum):
    V8N = 0
    V11N = 1
    V11S = 2
    V11M = 3


# Model's weights paths
WEIGHT_NAMES = ["yolov8n-face.pt",
                "yolov11n-face.pt",
                "yolov11s-face.pt",
                "yolov11m-face.pt"]

# Google Drive URL from repo (https://github.com/derronqi/yolov8-face) ~6MB
WEIGHT_URLS = ["https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11s-face.pt",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt"]


class YoloDetectorClient(Detector):
    def __init__(self, model: YoloModel):
        super().__init__()
        self.model = self.build_model(model)

    def build_model(self, model: YoloModel) -> Any:
        """
        Build a yolo detector model
        Returns:
            model (Any)
        """

        # Import the optional Ultralytics YOLO model
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as e:
            raise ImportError(
                "Yolo is an optional detector, ensure the library is installed. "
                "Please install using 'pip install ultralytics'"
            ) from e

        weight_file = weight_utils.download_weights_if_necessary(
            file_name=WEIGHT_NAMES[model.value], source_url=WEIGHT_URLS[model.value]
        )

        # Return face_detector
        return YOLO(weight_file)

    def detect_faces(self, img: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[List[FacialAreaRegion],List[List[FacialAreaRegion]]]:
        """
        Detect and align face with yolo

        Args:
            img: Union[np.ndarray, List[np.ndarray]]: pre-loaded image as numpy array 
            or list of images as List[np.ndarray] or batch of images as np.ndarray (N, H, W, C)

        Returns:
            results Union[List[FacialAreaRegion],List[List[FacialAreaRegion]]]: A list of FacialAreaRegion objects(for single image)
                or a list of lists of FacialAreaRegion objects(for batch of images) where each object contains:
                - facial_area (FacialAreaRegion): The facial area region represented as x, y, w, h, left_eye,right_eye and confidence.
        """
        
        if (not isinstance(img, list)) and (not isinstance(img, np.ndarray)):
            raise ValueError("img must be a numpy array or a list of numpy arrays")
        is_batch = (
            isinstance(img, list) or img.ndim == 4
        )
        if not is_batch:
            img = [img]

        # Detect faces
        results_list = self.model.predict(
            img,
            verbose=False,
            show=False,
            conf=float(os.getenv("YOLO_MIN_DETECTION_CONFIDENCE", "0.25")),
        )
        resp_list = []
        for results in results_list:
            resp = []
            # For each face, extract the bounding box, the landmarks and confidence
            for result in results:
                if result.boxes is None:
                    continue

                # Extract the bounding box and the confidence
                x, y, w, h = result.boxes.xywh.tolist()[0]
                confidence = result.boxes.conf.tolist()[0]

                right_eye = None
                left_eye = None

                # yolo-facev8 is detecting eyes through keypoints,
                # while for v11 keypoints are always None
                if result.keypoints is not None:
                    # right_eye_conf = result.keypoints.conf[0][0]
                    # left_eye_conf = result.keypoints.conf[0][1]
                    right_eye = result.keypoints.xy[0][0].tolist()
                    left_eye = result.keypoints.xy[0][1].tolist()

                    # eyes are list of float, need to cast them tuple of int
                    left_eye = tuple(int(i) for i in left_eye)
                    right_eye = tuple(int(i) for i in right_eye)

                x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                facial_area = FacialAreaRegion(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    confidence=confidence,
                )
                resp.append(facial_area)
            resp_list.append(resp)
        if not is_batch:
            return resp_list[0]
        return resp_list


class YoloDetectorClientV8n(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V8N)


class YoloDetectorClientV11n(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V11N)


class YoloDetectorClientV11s(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V11S)


class YoloDetectorClientV11m(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V11M)
