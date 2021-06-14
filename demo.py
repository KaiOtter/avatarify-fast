import sys
import os

sys.path.append(os.getcwd())
from afy.predictor_local import PredictorLocal
import cv2

from facial_landmark.onnx_detector import Detector as LandmarkDetector
from facial_landmark.face_onnx.detector import Detector as FaceDetector
import onnxruntime as ort

# close useless warning
ort.set_default_logger_severity(3)
from face_swap import Swap
import torch
import numpy as np
import yaml
from tqdm import tqdm
import time
import imageio


class Pipeline:
    def __init__(self, config_path, image_path):
        assert os.path.exists(image_path)
        with open(config_path) as f:
            self.path_config = yaml.load(f, Loader=yaml.FullLoader)['path_params']
        self.ckpt_path = self.path_config['vox_ckpt']
        self.predictor = PredictorLocal(config_path, self.ckpt_path, device='cuda')

        self.face_detector = FaceDetector()
        self.lmk_detector = LandmarkDetector()
        self.new_face = self.set_face(image_path)
        self.predictor.set_source_image(self.new_face)

        swap_ckpt = self.path_config['swap_ckpt']
        swap_config = self.path_config['swap_config']
        swap_index = self.path_config['swap_index']
        # swap_ckpt = '/workspace/face/code/motion-cosegmentation/weights/vox-10segments.pth.tar'
        # swap_config = './swap_config/vox-256-sem-10segments.yaml'
        # swap_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # 输入输出均为 [0, 1]之间的float类型，RGB图像
        self.face_swap = Swap(swap_config, swap_ckpt, swap_index)

        print('init successfully')

    def set_face(self, image_path):
        face_img = cv2.imread(image_path)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (256, 256))
        return face_img

    def frame_forward(self, bgr_image: np.array([1])):
        if bgr_image is None:
            print('bgr_image is empty')
            return None

        H, W, _ = bgr_image.shape
        if H == 0 or W == 0:
            print('bgr_image is empty', H, W)
            return None

        result, _ = self.face_detector.detect(bgr_image)
        if result is None or len(result) < 1:
            print('do not find a face!')
            return None
        face_bbox = result[0]

        # bgr_image = cv2.rectangle(bgr_image, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0, 0, 255), thickness=2)
        bw = abs(face_bbox[2] - face_bbox[0])
        bh = abs(face_bbox[3] - face_bbox[1])

        if bh < 60:
            if result is None:
                print('face too small')
                return None

        rect_side = int(max(bw, bh) * 1.5)
        # rect_side = int(max(bw, bh) * 1.0)
        shift_x0 = (rect_side - bw) // 2
        shift_x1 = (rect_side - bw) - shift_x0
        shift_y0 = (rect_side - bh) // 2
        shift_y1 = (rect_side - bh) - shift_y0
        crop_x0 = face_bbox[0] - shift_x0
        crop_y0 = face_bbox[1] - shift_y0
        crop_x1 = face_bbox[2] + shift_x1
        crop_y1 = face_bbox[3] + shift_y1

        if crop_x0 < 0:
            crop_x0 = 0
            crop_x1 = rect_side
        if crop_y0 < 0:
            crop_y0 = 0
            crop_y1 = rect_side
        if crop_x1 >= W:
            crop_x1 = W - 1
            crop_x0 = crop_x1 - rect_side
        if crop_y1 >= H:
            crop_y1 = H - 1
            crop_y0 = crop_y1 - rect_side

        assert crop_x1 - crop_x0 == crop_y1 - crop_y0 == rect_side

        crop_image = bgr_image[crop_y0:crop_y1, crop_x0:crop_x1]
        try:
            crop_image = cv2.resize(crop_image, (256, 256))
        except cv2.error as e:
            print(e)
            return None

        # cv2.imwrite("/workspace/face/data/my_golden_wheel/01150_crop.jpg", crop_image)

        crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)

        motion_output = self.predictor.predict(crop_rgb)

        swap_src = motion_output.astype(np.float64) / 255

        swap_tar = crop_rgb.astype(np.float64) / 255

        swap_output = self.face_swap.forward(swap_src, swap_tar)
        swap_output = cv2.cvtColor(swap_output, cv2.COLOR_RGB2BGR)

        # cv2.imwrite("/workspace/face/data/my_golden_wheel/01150_gan.jpg", swap_output)

        swap_output = cv2.resize(swap_output, (crop_x1 - crop_x0, crop_y1 - crop_y0))

        old_patch = bgr_image[crop_y0:crop_y1, crop_x0:crop_x1]

        lmks = self.lmk_detector.detect_full(old_patch)
        half_nose = abs(lmks[28][1] - lmks[29][1])

        lmks_part1 = lmks[:17]

        lmks_part2 = lmks[17:27]
        _top = min(lmks[19, 1] - half_nose, lmks[24, 1] - half_nose)
        lmks_part2[:, 1] = _top

        lmks_part2 = lmks_part2.tolist()
        lmks_part2.reverse()

        left_idx = lmks_part1[:, 0].argmin()
        left, bottom0 = lmks_part1[left_idx]
        right_idx = lmks_part1[:, 0].argmax()
        right, bottom1 = lmks_part1[right_idx]

        new_lmks = lmks_part1.tolist() + lmks_part2
        # lmks_part1[:7][0] -= half_nose
        # lmks_part1[8:][0] += half_nose
        # new_lmks = new_lmks.tolist()
        # lmks_part1.append([right, 0])
        # lmks_part1.append([left, 0])

        # new_lmks = lmks_part1

        new_lmks = np.array(new_lmks).astype(np.int)
        mask = np.zeros([crop_y1 - crop_y0, crop_x1 - crop_x0], dtype=np.uint8)

        old_patch = old_patch.astype(np.float32)

        above_eye_area = [[left, 0], [right, 0], [right, bottom1],
                          [left, bottom0]]
        above_eye_area = np.array(above_eye_area, dtype=np.int)
        cv2.fillPoly(mask, [new_lmks], 255)
        mask2 = mask.copy()
        cv2.fillPoly(mask2, [above_eye_area], 255)
        mask2 = mask2 - mask
        # cv2.imwrite("/workspace/face/data/my_golden_wheel/debug6_mask.jpg", mask2)

        new_patch = old_patch.copy()
        thre1 = mask > 127
        # new_patch[thre] = swap_output[thre]

        H_old = old_patch[thre1]
        old_median = []
        old_mean = []
        for t in range(3):
            zz = H_old[..., t]
            old_mean.append(zz.mean())
            # zz.sort()
            # old_median.append(zz[len(zz) // 2])

        H_new = swap_output[thre1]
        new_median = []
        new_mean = []
        for t in range(3):
            zz = H_new[..., t]
            new_mean.append(zz.mean())
            # zz.sort()
            # new_median.append(zz[len(zz) // 2])

        # better than using
        mean_diff_bgr = [(a - b) for a, b in zip(new_mean, old_mean)]
        mean_diff_bgr = np.array(mean_diff_bgr, dtype=np.int)

        # median_dif_bgr = [a - b for a, b in zip(new_median, old_median)]
        # median_diff_bgr = np.array(mid_dif_bgr, dtype=np.int)

        tmp = swap_output.copy()
        tmp = tmp.astype(np.int)
        tmp -= mean_diff_bgr
        tmp = np.clip(tmp, 0, 255).astype(np.uint8)
        new_patch[thre1] = tmp[thre1]
        thre2 = mask2 > 127
        new_patch[thre2] = old_patch[thre2]

        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # edge_mask = np.zeros([crop_y1 - crop_y0, crop_x1 - crop_x0], dtype=np.uint8)
        # cv2.drawContours(edge_mask, contours, -1, 255, 5)
        # edge_mask = cv2.blur(edge_mask, ksize=(7, 7))
        # edge_area = edge_mask > 0
        # edge_mask = edge_mask.astype(np.float32) / 255
        #
        # fused_edge = edge_mask[..., np.newaxis] * new_patch + (1 - edge_mask)[..., np.newaxis] * old_patch
        # new_patch[edge_area] = fused_edge[edge_area]

        # cv2.imwrite("/workspace/face/data/my_golden_wheel/debug6_patch2_mean.jpg", new_patch.astype(np.uint8))
        bgr_image[crop_y0:crop_y1, crop_x0:crop_x1] = new_patch.astype(np.uint8)

        # bgr_image[crop_y0:crop_y1, crop_x0:crop_x1] = swap_output

        return bgr_image


def make_video(fps, src_dir, work_dir, save):
    '''
    :param fps:
    :param src_dir: use origin frames when missing face
    :param work_dir: the root of swapped face frames
    :param save:
    :return:
    '''
    from skimage import img_as_ubyte

    ubyte_list = []
    miss = 0

    image_list = os.listdir(src_dir)
    image_list = sorted(image_list)
    for img in tqdm(image_list):
        dst = os.path.join(work_dir, img)
        if os.path.exists(dst):
            frame = imageio.imread(dst)
        else:
            frame = imageio.imread(os.path.join(src_dir, img))
        if frame is None:
            miss += 1
            frame = ubyte_list[-1]

        ubyte_list.append(img_as_ubyte(frame))
    print('miss', miss)
    imageio.mimsave(save, ubyte_list, fps=fps)


if __name__ == '__main__':
    image_root = ""
    target_face_image = ''

    worker = Pipeline('./config.yaml', target_face_image)

    save_dir = ""
    os.makedirs(save_dir, exist_ok=True)
    #
    log = ""

    image_list = os.listdir(image_root)
    image_list = sorted(image_list)

    from tqdm import tqdm

    with open(log, 'w+') as f:
        for file in tqdm(image_list):
            print(file)
            path = os.path.join(image_root, file)
            assert os.path.exists(path)
            image = cv2.imread(path)
            output = worker.frame_forward(image)
            if output is None:
                f.write("{} fail!\n".format(file))
            else:
                cv2.imwrite(os.path.join(save_dir, file), output)


    target_video = "xxx.mp4"
    reader = imageio.get_reader(target_video)
    fps = reader.get_meta_data()['fps']
    reader.close()

    save_video_path = 'xxx.mp4'
    make_video(fps, image_root, save_dir, save_video_path)
