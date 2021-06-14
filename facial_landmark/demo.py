'''
Reference from https://github.com/ainrichman/Peppa-Facial-Landmark-PyTorch

Three detector options: 
1. Original PyTorch inference detector 
2. MNN Python inference detector (experimental) 
3. ONNX inference detector based on onnxruntime

MNN detector is only tested on Windows 10 and Centos7.
'''

import cv2
from face_onnx.detector import Detector as FaceDetector
from onnx_detector import Detector
import numpy as np
import onnxruntime as ort

# close useless warning
ort.set_default_logger_severity(3)

face_detector = FaceDetector()
lmk_detector = Detector()


def demo(frame):
    bboxes, _ = face_detector.detect(frame)
    if len(bboxes) != 0:
        bbox = bboxes[0]
        bbox = bbox.astype(np.int)
        print(bbox.shape)
        print(bbox)
        lmks, _ = lmk_detector.detect(frame, bbox)

        lmks = lmks.astype(np.int)
        frame = cv2.rectangle(frame, tuple(bbox[0:2]), tuple(bbox[2:4]), (0, 0, 255), 1, 1)

        for point in lmks:
            frame = cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1, 1)

    return frame


def mask_test(src_frame, dst_frame):
    h, w, _ = dst_frame.shape
    lmks, _ = lmk_detector.detect(dst_frame, np.array([0, 0, w, h]))

    half_nose = abs(lmks[28][1] - lmks[29][1])
    # top_h = (lmks[19][1] + lmks[24][1]) / 2 - half_nose * 8

    # lmks_part2 = lmks[17:27]
    # # lmks_part2[:, 1] = top_h
    # lmks_part2 = lmks_part2.tolist()
    # lmks_part2.reverse()

    lmks_part1 = lmks[:17]
    left = lmks_part1[:, 0].min()
    left += 5
    right = lmks_part1[:, 0].max()
    right += 5

    lmks_part1 = lmks_part1 + np.array([0, half_nose * 2])
    lmks_part1[:7][0] -= 5
    lmks_part1[8:][0] += 5
    lmks_part1 = lmks_part1.tolist()
    lmks_part1.append([right, 0])
    lmks_part1.append([left, 0])

    new_lmks = lmks_part1

    new_lmks = np.array(new_lmks).astype(np.int)
    mask = np.zeros([h, w], dtype=np.uint8)

    cv2.fillPoly(mask, [new_lmks], 255)
    # for point in lmks:
    #     dst_frame = cv2.circle(dst_frame, tuple(point), 2, (0, 0, 255), -1, 1)

    mask = mask > 127
    src_frame[mask] = dst_frame[mask]

    return src_frame


if __name__ == "__main__":
    # a = "./test.jpg"
    # frame = cv2.imread(a)
    # r = demo(frame)
    # cv2.imwrite(a.replace('.jpg', '_lmk.jpg'), r)

    dd = cv2.imread("/workspace/face/data/my_golden_wheel/01150_gan.jpg")
    cc = cv2.imread('/workspace/face/data/my_golden_wheel/01150_crop.jpg')
    r = mask_test(cc, dd)

    cv2.imwrite("/workspace/face/data/my_golden_wheel/01150_gan_fuse.jpg", r)
