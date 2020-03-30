import cv2
import matplotlib
matplotlib.use('Agg')
import yaml
from argparse import ArgumentParser

import imageio
import numpy as np
from skimage.transform import resize

import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp


def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if cpu:
        generator.cpu()
    else:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if cpu:
        kp_detector.cpu()
    else:
        kp_detector.cuda()

    checkpoint = torch.load(checkpoint_path, map_location='cpu' if cpu else None)
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")

    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="CPU only (no CUDA used)")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)
    source_image = resize(source_image, (256, 256))[..., :3]

    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    kp_driving_initial = None

    with torch.no_grad():
        spm = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = spm.cpu() if opt.cpu else spm.cuda()
        kp_source = kp_detector(source)
        if kp_driving_initial is None:
            kp_driving_initial = kp_source

        video = cv2.VideoCapture(0)

        while True:
            frame = video.read()
            if isinstance(frame, tuple):
                frame = frame[1]
            if frame is None:
                print("Oops frame is None. Possibly camera or display does not work")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            y, x, _ = frame.shape
            min_dim = min(y, x)
            startx = x // 2 - (min_dim // 2)
            starty = y // 2 - (min_dim // 2)
            frame = frame[starty:starty + min_dim, startx:startx + min_dim, :]

            frame = resize(frame, (256, 256))[..., :3]
            frame = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

            kp_driving = kp_detector(frame)

            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=opt.relative,
                                   use_relative_jacobian=opt.relative, adapt_movement_scale=opt.adapt_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            p = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            cv2.imshow('Video', p)
            key = cv2.waitKey(1)
            if key in [ord('q'), 202, 27]:
                break
