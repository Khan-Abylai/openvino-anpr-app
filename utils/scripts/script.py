import time
import os
import cv2
import numpy as np
import openvino as ov

def nms_np(predictions, conf_thres=0.2, nms_thres=0.2, include_conf=False):
    filter_mask = (predictions[:, -1] >= conf_thres)
    predictions = predictions[filter_mask]

    if len(predictions) == 0:
        return np.array([])

    output = []

    while len(predictions) > 0:
        max_index = np.argmax(predictions[:, -1])

        if include_conf:
            output.append(predictions[max_index])
        else:
            output.append(predictions[max_index, :-1])

        ious = bbox_iou_np(np.array([predictions[max_index, :-1]]), predictions[:, :-1], x1y1x2y2=False)

        predictions = predictions[ious < nms_thres]

    return np.stack(output)

def bbox_iou_np(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:

        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0,
                                                                               None)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def draw_plate_detection(image, plate):
    """
    Draw plate detection visualization
    plate format: [center_x, center_y, width, height, lt_x, lt_y, lb_x, lb_y, rt_x, rt_y, rb_x, rb_y]
    """
    # Colors
    RED = (0, 0, 255)      # For center point
    GREEN = (0, 255, 0)    # For corner points
    BLUE = (255, 0, 0)     # For width-height box

    # Extract points
    center_x, center_y = int(plate[0]), int(plate[1])
    width, height = int(plate[2]), int(plate[3])
    lt = (int(plate[4]), int(plate[5]))    # left top
    lb = (int(plate[6]), int(plate[7]))    # left bottom
    rt = (int(plate[8]), int(plate[9]))    # right top
    rb = (int(plate[10]), int(plate[11]))  # right bottom

    # Draw center point
    cv2.circle(image, (center_x, center_y), 5, RED, -1)

    # Draw corner points
    for point in [lt, lb, rt, rb]:
        cv2.circle(image, point, 3, GREEN, -1)

    # Draw lines between corners
    cv2.line(image, lt, rt, GREEN, 2)  # Top line
    cv2.line(image, rt, rb, GREEN, 2)  # Right line
    cv2.line(image, rb, lb, GREEN, 2)  # Bottom line
    cv2.line(image, lb, lt, GREEN, 2)  # Left line

    # Draw width-height box
    half_w, half_h = width//2, height//2
    top_left = (center_x - half_w, center_y - half_h)
    bottom_right = (center_x + half_w, center_y + half_h)
    cv2.rectangle(image, top_left, bottom_right, BLUE, 2)

    return image

class DetectonOV:
    def __init__(self, image_path, model_path, th):
        self.shape = 512
        self.images, self.im0 = self.preprocess(image_path)
        self.model = self.load_model(model_path)
        self.th = th
    
    def __call__(self):
        h, w = self.im0.shape[0:2]
        rx, ry = w/self.shape, h/self.shape
        inp = self.images.transpose((0, 3, 1, 2))
        sec = []
        for im in inp:
            st = time.time()
            out = self.model(np.expand_dims(im, axis=0))
            end = time.time()
            sec.append(end-st)
        out = nms_np(out[0][0])
        out[..., [4, 6, 8, 10]] += out[..., [0]]
        out[..., [5, 7, 9, 11]] += out[..., [1]]
        out[..., 0::2] *= rx
        out[..., 1::2] *= ry

        for plate in out:
            if plate[-1] > self.th:
                print(plate)
                image = draw_plate_detection(self.im0, plate)
                cv2.imwrite("output.jpg", image)
        return sec

    def preprocess(self, path):
        if os.path.isdir(path):
            l = [os.path.join(path, i) for i in os.listdir(path)]
        else:
            l = [path]
        arr = []
        for im in l:
            img = cv2.imread(im)
            im0 = img.copy()
            img = cv2.resize(img, (self.shape, self.shape))
            img = img.astype(np.float32)
            img = 2* (img/254.5)
            arr.append(img)
        return np.stack(arr, axis=0), im0

    def load_model(self, model_path):
        """
        Load the OpenVINO IR model.

        :param model_path: Path to the OpenVINO `.xml` model file.
        :return: Compiled model and input layer information.
        """
        core = ov.Core()
        model = core.read_model(model=model_path)
        compiled_model = core.compile_model(model=model, device_name="CPU")

        return compiled_model


if __name__ == "__main__":
    model_path = "../models/detection.xml"
    image_path = "../images/test.jpeg"
    th = 0.5
    detector = DetectonOV(image_path, model_path, th)
    detector()