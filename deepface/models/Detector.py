from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

# Notice that all facial detector models must be inherited from this class


# pylint: disable=unnecessary-pass, too-few-public-methods, too-many-instance-attributes
class Detector(ABC):

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

                - facial_area (FacialAreaRegion): The facial area region represented
                    as x, y, w, h, left_eye and right_eye. left eye and right eye are
                    eyes on the left and right respectively with respect to the person
                    instead of observer.
        """
        if isinstance(img, np.ndarray) and img.ndim == 3:
            # single image
            return self._detect_in_single_img(img)
        elif isinstance(img,list) or (isinstance(img, np.ndarray) and img.ndim == 4):
            return [self._detect_in_single_img(image) for image in img]
        else:
            raise ValueError("Invalid input type. Expected numpy array or list of numpy arrays.")
    
    def _detect_in_single_img(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Helper function to detection in single image

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        raise NotImplementedError("This method should be overridden by subclasses if batching is not supported.")

@dataclass
class FacialAreaRegion:
    """
    Initialize a Face object.

    Args:
        x (int): The x-coordinate of the top-left corner of the bounding box.
        y (int): The y-coordinate of the top-left corner of the bounding box.
        w (int): The width of the bounding box.
        h (int): The height of the bounding box.
        left_eye (tuple): The coordinates (x, y) of the left eye with respect to
            the person instead of observer. Default is None.
        right_eye (tuple): The coordinates (x, y) of the right eye with respect to
            the person instead of observer. Default is None.
        confidence (float, optional): Confidence score associated with the face detection.
            Default is None.
    """

    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None
    nose: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None


@dataclass
class DetectedFace:
    """
    Initialize detected face object.

    Args:
        img (np.ndarray): detected face image as numpy array
        facial_area (FacialAreaRegion): detected face's metadata (e.g. bounding box)
        confidence (float): confidence score for face detection
    """

    img: np.ndarray
    facial_area: FacialAreaRegion
    confidence: float
