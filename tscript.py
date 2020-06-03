import torch
import torch.nn as nn

class TNet(nn.Module):
   def __init__(self):
       super(TNet, self).__init__()

   def forward(self,x):
       _, x2 = torch.split(x, [1, 2], 3)
       x2 = x2.reshape((10,2,2))
       #x2 = torch.inverse(x2)
       x2 = x2.reshape((1,10, 2, 2))
       return x2


net = TNet()
x1 = torch.tensor([0.35159385, 1.4026177, -0.16491778, 0.15036172, 0.14590532, 1.7214202, -0.01648209, 2.0667543, -0.54575884, 0.58275443, 0.31624478, 1.3999747, 0.20542753, 1.8475686, 0.2493963, -0.39412174, 0.3819594, 2.0253587, -0.47573334, 0.9894327, 1.0208408, 0.27658972, -0.26601127, 2.4485419, -0.033901345, 2.305436, -0.27099714, -0.013389754, 0.39519912, 1.9000958, 0.3378937, 1.2235969, 1.176776, 0.604899, -1.0020723, 2.2966354, 0.20534195, 1.7340744, -0.19135174, -0.065742806, 0.4940952, 1.8216875, -0.11525583, 1.0760419, 0.25169486, 0.8061476, -0.3411195, 1.12281, -0.37333015, 1.8477993, 0.03703611, -0.24712409, 0.037584323, 2.2576797, -0.114279166, 1.245715, 0.8529224, -0.34079376, -0.64367706, 1.7498884],dtype=torch.float32)
x1 = x1.reshape((1,10,2,3))
traced = torch.jit.trace(net, (x1))

traced.save('tnet_jit.zip')

print(traced(x1))


def forward(self, source_image, kp_driving, kp_source, kp_initial):
    # source_image = self.quant(source_image)
    # kp_driving = self.quant(kp_driving)
    # kp_source = self.quant(kp_source)
    # kp_initial = self.quant(kp_initial)
    kp_driving_value, kp_driving_jacobian = torch.split(kp_driving, [1, 2], 3)
    kp_driving_value = torch.squeeze(kp_driving_value, 3)
    kp_source_value, kp_source_jacobian = torch.split(kp_source, [1, 2], 3)
    kp_source_value = torch.squeeze(kp_source_value, 3)
    kp_initial_value, kp_initial_jacobian = torch.split(kp_initial, [1, 2], 3)
    kp_initial_value = torch.squeeze(kp_initial_value, 3)

    kp_value_diff = (kp_driving_value - kp_initial_value)
    kp_np_value = kp_value_diff + kp_source_value

    jacobian_diff = torch.matmul(kp_driving_jacobian, torch.inverse(kp_initial_jacobian))
    # jacobian_diff = torch.matmul(kp_driving_jacobian, kp_initial_jacobian)
    kp_np_jacobian = torch.matmul(jacobian_diff, kp_source_jacobian)

