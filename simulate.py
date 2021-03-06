import cv2


import imageio
import yaml
from argparse import ArgumentParser

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


def process(opt,img_orig, generator, kp_detector):
    img = resize(img_orig, (256, 256))[..., :3]
    kp_driving_initial = None
    with torch.no_grad():
        spm = torch.tensor(img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = spm.cpu() if opt.cpu else spm.cuda()
        kp_source = kp_detector(source)

        video = cv2.VideoCapture(opt.video)
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        vout = cv2.VideoWriter(opt.out, fourcc, fps, (256, 256))
        while True:
            frame_img = video.read()
            if isinstance(frame_img, tuple):
                frame_img = frame_img[1]
            if frame_img is None:
                print("Oops frame is None. Possibly camera or display does not work")
                break

            #frame_img = cv2.rotate(frame_img,cv2.ROTATE_180)
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

            y, x, _ = frame_img.shape
            min_dim = min(y, x)
            startx = x // 2 - (min_dim // 2)
            starty = y // 2 - (min_dim // 2)
            frame_img = frame_img[starty:starty + min_dim, startx:startx + min_dim, :]

            frame_img = resize(frame_img, (256, 256))[..., :3]
            frame = torch.tensor(frame_img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

            kp_driving = kp_detector(frame)
            if kp_driving_initial is None:
                kp_driving_initial = kp_driving
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=opt.relative,
                                   use_relative_jacobian=opt.relative, adapt_movement_scale=opt.adapt_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            p = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            p = p*255
            p = p.astype(np.uint8)
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            vout.write(p)

        video.release()
        vout.release()

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--sources", default='source')
    parser.add_argument("--video", default='')
    parser.add_argument("--out", default='result.mp4', type=str)

    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="CPU only (no CUDA used)")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    process(opt,imageio.imread(opt.sources), generator, kp_detector)


if __name__ == "__main__":
    main()
