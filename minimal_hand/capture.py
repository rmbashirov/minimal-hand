import cv2
import numpy as np


class OpenCVCapture:
  """
  OpenCV wrapper to read from webcam.
  """
  def __init__(self, video_fp):
    """
    Init.
    """
    # self.cap = cv2.VideoCapture(0)

    self.cap = cv2.VideoCapture(video_fp)

  def read(self):
    """
    Read one frame. Note this function might be blocked by the sensor.

    Returns
    -------
    np.ndarray
      Read frame. Might be `None` is the webcam fails to get on frame.
    """
    flag, frame = self.cap.read()
    if not flag:
      return None
    # flag, frame = self.cap.read()
    # if not flag:
    #   return None
    h, w = frame.shape[:2]
    pad = 0.2
    frame = frame[:, int(h * pad):int(h * (1 + pad))]
    print(frame.shape)
    return np.flip(frame, -1).copy()  # BGR to RGB
