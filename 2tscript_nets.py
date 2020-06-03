import cv2


import yaml
import torch
import torch.nn as nn
import numpy as np

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector

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

class Net1(nn.Module):
    def __init__(self,generator):
        super(Net1, self).__init__()
        self.generator = generator
    def forward(self,img,img_padded):
        return generator.forward_prepare(img,img_padded)

class Net2(nn.Module):
    def __init__(self,generator):
        super(Net2, self).__init__()
        self.generator = generator
    def forward(self,source_image_small,kp_driving, kp_source, kp_initial):
        return generator.forward_prepare_optical(source_image_small,kp_driving, kp_source, kp_initial)

class Net3(nn.Module):
    def __init__(self,generator):
        super(Net3, self).__init__()
        self.generator = generator
    def forward(self,input,sparse_motion):
        return generator.forward_optical(input,sparse_motion)

class Net4(nn.Module):
    def __init__(self,generator):
        super(Net4, self).__init__()
        self.generator = generator
    def forward(self,out,deformation,occlusion_map):
        return generator.forward_prepare_last(out,deformation,occlusion_map)

class Net5(nn.Module):
    def __init__(self,generator):
        super(Net5, self).__init__()
        self.generator = generator
    def forward(self,out):
        return generator.forward_last(out)

x = torch.randn(1, 3, 256, 256, requires_grad=True)
x_padded = torch.randn(1, 3, 256+12, 256+12, requires_grad=True)

kp_res = kp_detector(x)

n1 = Net1(generator)
n1.eval()
out,small = n1(x,x_padded)
print('out: {}'.format(out.shape))
print('small: {}'.format(small.shape))
n2 = Net2(generator)
n2.eval()
optical_input,sparse_motion = n2(small,kp_res, kp_res, kp_res)
print('optical_input: {}'.format(optical_input.shape))
print('sparse_motion: {}'.format(sparse_motion.shape))
n3 = Net3(generator)
n3.eval()
deformation,occlusion_map = n3(optical_input,sparse_motion)
print('deformation: {}'.format(deformation.shape))
print('occlusion_map: {}'.format(sparse_motion.shape))
n4 = Net4(generator)
n4.eval()
n4_out = n4(out,deformation,occlusion_map)
print('n4_out: {}'.format(n4_out.shape))
n5 = Net5(generator)
n5.eval()
res = n5(n4_out)
print('res: {}'.format(res.shape))

print("generate *****************************")

g2_traced = torch.jit.trace(n2,(small,kp_res, kp_res, kp_res))
g2_traced.save('gnet2.zip')

g4_traced = torch.jit.trace(n4,(out,deformation,occlusion_map))
g4_traced.save('gnet4.zip')