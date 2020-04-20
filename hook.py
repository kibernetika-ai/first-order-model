import logging
import cv2
from ml_serving.utils import helpers
import yaml
import os
import numpy as np
from skimage.transform import resize
import imageio

import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp

LOG = logging.getLogger(__name__)


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


generator = None
kp_detector = None
is_cpu = True
source_image = None


class Worker:
    def __init__(self, img_orig):
        img = resize(img_orig, (256, 256))[..., :3]
        with torch.no_grad():
            spm = torch.tensor(img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            self.source = spm.cpu() if is_cpu else spm.cuda()
            self.kp_source = kp_detector(self.source)
            self.kp_driving_initial = None
            self.count = 0

    def process(self, frame):
        with torch.no_grad():
            y, x, _ = frame.shape
            min_dim = min(y, x)
            startx = x // 2 - (min_dim // 2)
            starty = y // 2 - (min_dim // 2)
            frame = frame[starty:starty + min_dim, startx:startx + min_dim, :]
            frame = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)


            kp_driving = kp_detector(frame)
            if self.kp_driving_initial is None:
                self.kp_driving_initial = kp_driving

            kp_norm = normalize_kp(kp_source=self.kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=self.kp_driving_initial, use_relative_movement=True,
                                   use_relative_jacobian=True, adapt_movement_scale=True)
            out = generator(self.source, kp_source=self.kp_source, kp_driving=kp_norm)

            p = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

            p = p * 255
            p = p.astype(np.uint8)
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            self.count += 1
            return p


def init_hook(**params):
    LOG.info("Got params:")
    LOG.info(params)
    global generator, kp_detector, is_cpu, source_image
    logging.info('GPU_COUNT: {}'.format(os.environ.get('GPU_COUNT', 0)))
    is_cpu = True if int(os.environ.get('GPU_COUNT', 0)) == 0 else False
    source_image = imageio.imread(params.get('source', './backgrounds/images/10000.png'))
    generator, kp_detector = load_checkpoints(config_path=params.get('config'),
                                              checkpoint_path=params.get('checkpoint'), cpu=is_cpu)


trackers = {}


def on_complete(meta, _):
    key = meta.get('key', None)
    LOG.info('Remove stream: {}'.format(key))
    if key is not None:
        t = trackers.get(key, None)
        if t is not None:
            del trackers[key]


def process(inputs, ctx, **kwargs):
    frame, is_video = helpers.load_image(inputs, 'image')
    frame = cv2.resize(frame, (256, 256))
    #if frame is not None:
    #    return {'output': frame}
    key = '1'#kwargs.get('metadata', {}).get('stream_id', None)
    if key is None:
        return {'output': frame}
    track = trackers.get(key, None)
    if track is None:
        track = Worker(source_image)
        trackers[key] = track
    frame = track.process(frame)
    return {'output': frame[:,:,::-1]}
