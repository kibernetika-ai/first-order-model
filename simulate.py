import cv2
import matplotlib

matplotlib.use('Agg')
import yaml
from argparse import ArgumentParser
import glob
import os
import random
import numpy as np
from skimage.transform import resize
import shutil

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


def process(opt, name, img_orig, generator, kp_detector, backgrounds):
    img = resize(img_orig, (256, 256))[..., :3]
    kp_driving_initial = None
    with torch.no_grad():
        spm = torch.tensor(img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = spm.cpu() if opt.cpu else spm.cuda()
        kp_source = kp_detector(source)
        if kp_driving_initial is None:
            kp_driving_initial = kp_source

        video = cv2.VideoCapture(opt.video)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fist_frame = int(fps)
        step = (frame_count - fist_frame) // opt.count
        if step < 1:
            step = 1
        print(f'Step: {step}')
        k = -fist_frame
        k = 0
        n = 1
        while True:
            print(k)
            frame_img = video.read()
            if isinstance(frame_img, tuple):
                frame_img = frame_img[1]
            if frame_img is None:
                print("Oops frame is None. Possibly camera or display does not work")
                break
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

            y, x, _ = frame_img.shape
            min_dim = min(y, x)
            startx = x // 2 - (min_dim // 2)
            starty = y // 2 - (min_dim // 2)
            frame_img = frame_img[starty:starty + min_dim, startx:startx + min_dim, :]

            frame_img = resize(frame_img, (256, 256))[..., :3]
            frame = torch.tensor(frame_img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

            kp_driving = kp_detector(frame)

            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=opt.relative,
                                   use_relative_jacobian=opt.relative, adapt_movement_scale=opt.adapt_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            if k < 0:
                k += 1
                continue
            if k % step < -1:
                k += 1
                continue

            k += 1
            p = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            p = p*255
            p = p.astype(np.float32)
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            res_img = cv2.resize(p, (160, 160))
            mask = np.copy(res_img)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[mask <= 20] = 0
            mask[mask > 20] = 255
            back = cv2.imread(random.choice(backgrounds))
            back_width = max(back.shape[1], 160)
            back_height = max(back.shape[0], 160)
            back = cv2.resize(back, (back_width, back_height))
            x0 = int(np.random.uniform(0, back_width - 160))
            y0 = int(np.random.uniform(0, back_height - 160))
            back = back[y0:y0 + 160, x0:x0 + 160, 0:3]
            fmask = np.reshape(mask, (160, 160, 1)).astype(np.float32) / 255
            # res_img = res_img.astype(np.float32)*fmask+back.astype(np.float32)*(1-fmask)
            # res_img = res_img.astype(np.float32)
            print(f'{name}-{n}.png')
            res_img = res_img.astype(np.uint8)
            thresh = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            print(thresh.shape)
            print(thresh.dtype)
            thresh = cv2.Canny(thresh, 20, 225)
            #thresh = cv2.medianBlur(thresh,3)
            #thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
            cv2.imwrite(os.path.join(opt.out_dir, 'images', f'{name}-{n}-c.png'), thresh)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            res_img = cv2.drawContours(res_img, contours, -1, (0, 255, 0), 3)
            cv2.imwrite(os.path.join(opt.out_dir, 'images', f'{name}-{n}.png'), res_img)
            cv2.imwrite(os.path.join(opt.out_dir, 'masks', f'{name}-{n}.png'), mask)
            #cv2.findContours(res_img,)
            n += 1
            if n==4:
                break
        video.release()

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--sources", default='backgrounds')
    parser.add_argument("--video", default='')
    parser.add_argument("--count", default=10, type=int)
    parser.add_argument("--backgrounds", default='backgrounds/backgrounds')
    parser.add_argument("--out_dir", default='backgrounds/new')

    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="CPU only (no CUDA used)")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()
    print(opt.out_dir)
    shutil.rmtree(opt.out_dir, ignore_errors=True)
    os.makedirs(opt.out_dir, exist_ok=True)
    os.mkdir(os.path.join(opt.out_dir, 'images'))
    os.mkdir(os.path.join(opt.out_dir, 'masks'))
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    backgrounds = []
    for b in glob.glob(opt.backgrounds + '/*'):
        backgrounds.append(b)
    for i in glob.glob(opt.sources + "/images/*"):
        name = os.path.basename(i)
        img = cv2.imread(i)[:, :, ::-1]
        mask = cv2.imread(opt.sources + "/masks/" + name)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        name = name.replace('.png', '')
        img = img.astype(np.float32)
        mask = mask.astype(np.float32) / 255
        bg = np.zeros((img.shape[0],img.shape[1],3),np.float32)
        bg[:,:,1] = 255
        img = mask * img
        img = img.astype(np.uint8)
        # img[:,:,0:1][mask == 0] = 0
        process(opt, name, img, generator, kp_detector, backgrounds)


if __name__ == "__main__":
    main()
