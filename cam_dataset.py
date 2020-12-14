from torch.utils.data import Dataset
import numpy as np
import glob
import random
import logging
import cv2
import os
import time
from skimage import img_as_float32

LOG = logging.getLogger(__name__)



def points_from_landmark(land):
    points = []
    points.append(land[0])
    points.append(land[8])
    points.append(land[16])

    points.append(land[36])#y left
    points.append(land[45])#yright

    points.append(land[60])#mouse left
    points.append(land[64])#mouse right

    points.append(land[29])#nose


    points.append(land[19])#brow
    points.append(land[24])#brow

    return np.array(points,np.float32)

class CamDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.width = 256
        self.data = []
        self.kf = 2

        def _close(f, frames_out, boxes_out, lands_out):
            lm = 1
            best_land = -1
            for i in range(len(lands_out)):
                l = lands_out[i]
                v = np.max(np.abs(l[61:64, 1] - l[65:68, 1][::-1]))
                if v < lm:
                    lm = v
                    best_land = i
            if best_land >= 0:
                self.data.append((f, best_land, frames_out, boxes_out, lands_out))

        for src in glob.glob(os.path.join(root_dir, "*")):
            LOG.info("Parse: {}".format(src))
            for f in glob.glob(os.path.join(src, "*")):
                data_file = os.path.join(f, 'data.npz')
                if not os.path.exists(data_file):
                    continue
                npzfile = np.load(data_file)
                lands = npzfile['landmarks3']
                if len(lands) < 3:
                    continue
                lands = lands[:, :, 0:2]
                frames = npzfile['frames']
                boxes = npzfile['boxes']
                lands_out = []
                frames_out = []
                boxes_out = []
                first = None
                fps = npzfile['fps'][0]
                for i, frame in enumerate(frames):
                    box = boxes[i]
                    x1 = max(0, box[0] - (box[2] - box[0]) / self.kf)
                    x2 = min(1, box[2] + (box[2] - box[0]) / self.kf)
                    y1 = max(0, box[1] - (box[3] - box[1]) / self.kf)
                    y2 = min(1, box[3] + (box[3] - box[1]) / self.kf)
                    if y1 >= y2 or x1 >= x2:
                        continue
                    l = lands[i]
                    if first is None:
                        first = l
                    else:
                        d = np.abs(first[0:16] - l[0:16])
                        d = np.sum(d)
                        if d > 2.5 or len(frames_out) > fps*4:
                            if len(frames_out) > 2:
                                _close(f, frames_out, boxes_out, lands_out)
                            lands_out = []
                            frames_out = []
                            boxes_out = []
                            first = l
                    frames_out.append(frame)
                    lands_out.append(lands[i])
                    boxes_out.append(boxes[i])
                if len(frames_out) > 2:
                    _close(f, frames_out, boxes_out, lands_out)

        LOG.info("Samples: {}".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        kf = 2
        f, best_land, frames, boxes, lands = self.data[idx]
        f1 = best_land
        f2 = random.randint(1, len(frames) - 1)
        box = boxes[f2]
        land_out = lands[f2][:,:2].copy()
        x1 = max(0, box[0] - (box[2] - box[0]) / kf)
        x2 = min(1, box[2] + (box[2] - box[0]) / kf)
        y1 = max(0, box[1] - (box[3] - box[1]) / kf)
        y2 = min(1, box[3] + (box[3] - box[1]) / kf)
        land_out[:, 0] = (land_out[:, 0] - x1) / (x2 - x1)
        land_out[:, 1] = (land_out[:, 1] - y1) / (y2 - y1)
        land_out = np.clip(land_out, 0, 1)
        land_out = points_from_landmark(land_out)
        land_out = land_out - 0.5

        img_out = cv2.imread(os.path.join(f, f'{frames[f2]}.jpg'))
        x1 = int(x1 * img_out.shape[1])
        x2 = int(x2 * img_out.shape[1])
        y1 = int(y1 * img_out.shape[0])
        y2 = int(y2 * img_out.shape[0])
        img_out = img_out[y1:y2, x1:x2, ::-1]
        img_out = cv2.resize(img_out, (256, 256))
        img_out = img_as_float32(img_out)
        img_in = cv2.imread(os.path.join(f, f'{frames[f1]}.jpg'))
        box = boxes[f1]
        x1 = max(0, box[0] - (box[2] - box[0]) / kf)
        x2 = min(1, box[2] + (box[2] - box[0]) / kf)
        y1 = max(0, box[1] - (box[3] - box[1]) / kf)
        y2 = min(1, box[3] + (box[3] - box[1]) / kf)

        land_in = lands[f1][:,:2].copy()
        land_in[:, 0] = (land_in[:, 0] - x1) / (x2 - x1)
        land_in[:, 1] = (land_in[:, 1] - y1) / (y2 - y1)
        land_in = np.clip(land_in, 0, 1)
        land_in = points_from_landmark(land_in)
        land_in = land_in - 0.5

        x1 = int(x1 * img_in.shape[1])
        x2 = int(x2 * img_in.shape[1])
        y1 = int(y1 * img_in.shape[0])
        y2 = int(y2 * img_in.shape[0])
        img_in = img_in[y1:y2, x1:x2, ::-1]
        img_in = cv2.resize(img_in, (256, 256))
        img_in = img_as_float32(img_in)
        out = {
            'driving': img_out.transpose((2, 0, 1)),
            'driving_upoints': land_out,
            'source': img_in.transpose((2, 0, 1)),
            'source_upoints': land_in,
        }
        return out

    def __iter__(self):
        random.seed(time.time())
        random.shuffle(self.data)
        kf = 2
        for f, best_land, frames, boxes, lands in self.data:
            f1 = best_land
            f2 = random.randint(1, len(frames) - 1)
            if abs(f2 - f1) < 2:
                continue
            box = boxes[f2]
            x1 = max(0, box[0] - (box[2] - box[0]) / kf)
            x2 = min(1, box[2] + (box[2] - box[0]) / kf)
            y1 = max(0, box[1] - (box[3] - box[1]) / kf)
            y2 = min(1, box[3] + (box[3] - box[1]) / kf)
            img_out = cv2.imread(os.path.join(f, f'{frames[f2]}.jpg'))
            if img_out is None:
                continue
            x1 = int(x1 * img_out.shape[1])
            x2 = int(x2 * img_out.shape[1])
            y1 = int(y1 * img_out.shape[0])
            y2 = int(y2 * img_out.shape[0])
            if y1 >= y2 or x1 >= x2:
                continue
            img_out = img_out[y1:y2, x1:x2, ::-1]
            img_out = cv2.resize(img_out, (256, 256))
            img_out = img_as_float32(img_out)
            img_in = cv2.imread(os.path.join(f, f'{frames[f1]}.jpg'))
            if img_in is None:
                continue
            box = boxes[f1]
            x1 = max(0, box[0] - (box[2] - box[0]) / kf)
            x2 = min(1, box[2] + (box[2] - box[0]) / kf)
            y1 = max(0, box[1] - (box[3] - box[1]) / kf)
            y2 = min(1, box[3] + (box[3] - box[1]) / kf)
            x1 = int(x1 * img_in.shape[1])
            x2 = int(x2 * img_in.shape[1])
            y1 = int(y1 * img_in.shape[0])
            y2 = int(y2 * img_in.shape[0])
            if y1 >= y2 or x1 >= x2:
                continue
            img_in = img_in[y1:y2, x1:x2, ::-1]
            img_in = cv2.resize(img_in, (256, 256))
            img_in = img_as_float32(img_in)
            out = {
                'driving': img_out.transpose((2, 0, 1)),
                'source': img_in.transpose((2, 0, 1)),
            }
            yield out
