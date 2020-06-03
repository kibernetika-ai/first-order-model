import cv2


import imageio
import yaml
from argparse import ArgumentParser

import numpy as np
from skimage.transform import resize
import os

import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp

import torch.onnx

from onnx_coreml import convert

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

    #generator = DataParallelWithCallback(generator)
    #kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

generator, kp_detector = load_checkpoints('./config/vox-adv-256.yaml','./models/vox-adv-cpk.pth.tar',cpu=True)



def f3():
    x1 = torch.randn(1, 10, 1, 58, 58)
    x2 = torch.randn(1, 10, 2,2)
    kp_detector.fmode = 2
    traced_kp = torch.jit.trace(kp_detector, (x1,x2))
    traced_kp.save('kp_detector2_jit.zip')
def f1():
    x1 = torch.randn(1, 3, 256+12, 256+12)
    kp_detector.fmode = 1
    x2,_ = kp_detector(x1)
    print(x2.shape)
    kp_detector.fmode = 2
    traced_kp = torch.jit.trace(kp_detector,(x2))
    traced_kp.save('kp_detector2_jit.zip')
    print(kp_detector(x2).shape)

def f2():
    img = torch.randn(1, 3, 256, 256)
    kp = kp_detector(img)
    out = torch.randn(1, 256, 64, 64)
    traced = torch.jit.trace(generator,(out,img,kp,kp,kp))
    traced.save('generator1_jit.zip')

f3()