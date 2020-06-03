import cv2


import imageio
import yaml
import torch
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

kp_detector.fmode = 1
x = torch.randn(1, 3, 256+12, 256+12, requires_grad=True)
#kp_out = kp_detector(x)
#print(kp_out)

# Export the model
torch.onnx.export(kp_detector,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "kp_detector.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['heatmap','jacobian'], # the model's output names
                  )

#custom_conversion_functions={"Slice": _convert_slice_v9}
#mlmodel = convert(model='./kp_detector.onnx',add_custom_layers=True, minimum_ios_deployment_target='13')
#mlmodel = convert(model='./kp_detector.onnx', minimum_ios_deployment_target='13')
# Save converted CoreML model
#mlmodel.save('kp_detector.mlmodel')

img = cv2.imread('/Users/agunin/Projects/Salut2Yoy/Salut2You/faces/obama.jpg')[:,:,::-1]
img = cv2.resize(img,(256+12, 256+12))
img = img.astype(np.float32)/255
img = torch.tensor(img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
_, j = kp_detector(img)
print(j.flatten())