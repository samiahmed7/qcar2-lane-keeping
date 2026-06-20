"""Small sensor_msgs/Image conversion helpers that avoid cv_bridge."""
import cv2
import numpy as np
from sensor_msgs.msg import Image


def image_msg_to_bgr(msg: Image):
    """Convert a ROS Image message to an OpenCV BGR uint8 image."""
    encoding = msg.encoding.lower()
    channels_by_encoding = {
        'bgr8': 3,
        'rgb8': 3,
        'bgra8': 4,
        'rgba8': 4,
        'mono8': 1,
        '8uc1': 1,
        '8uc3': 3,
        '8uc4': 4,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f'unsupported image encoding: {msg.encoding}')

    channels = channels_by_encoding[encoding]
    expected_row_bytes = int(msg.width) * channels
    if int(msg.step) < expected_row_bytes:
        raise ValueError(
            f'image step {msg.step} is smaller than expected {expected_row_bytes}'
        )

    try:
        data = np.frombuffer(msg.data, dtype=np.uint8)
    except TypeError:
        data = np.asarray(msg.data, dtype=np.uint8)

    expected_size = int(msg.step) * int(msg.height)
    if data.size < expected_size:
        raise ValueError(
            f'image data has {data.size} bytes, expected at least {expected_size}'
        )

    rows = data[:expected_size].reshape((int(msg.height), int(msg.step)))
    packed = rows[:, :expected_row_bytes]
    image = packed.reshape((int(msg.height), int(msg.width), channels))

    if encoding in ('bgr8', '8uc3'):
        return image.copy()
    if encoding == 'rgb8':
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == 'bgra8':
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding in ('rgba8', '8uc4'):
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    mono = image.reshape((int(msg.height), int(msg.width)))
    return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)


def bgr_to_image_msg(image_bgr, header=None):
    """Convert an OpenCV BGR uint8 image to a ROS Image message."""
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('expected HxWx3 BGR image')
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = np.ascontiguousarray(image)

    msg = Image()
    if header is not None:
        msg.header = header
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = int(image.shape[1]) * 3
    msg.data = image.tobytes()
    return msg
