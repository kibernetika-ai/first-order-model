import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork, DenseMotionNetworkFake


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def clean_up(self):
        self.final = None
        self.up_blocks = None
        self.bottleneck = None
        self.down_blocks = None
        self.first = None
        self.dense_motion_network.clean_up()

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:

                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            output_dict["deformed"] = self.deform_input(source_image, deformation)

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict

    def forward_setup(self, source_image):
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        scaled_image = self.dense_motion_network.forward_setup(source_image)
        return out, scaled_image

    def forward_step(self, out, source_image, kp_new_value, kp_new_jacobian, kp_source_value, kp_source_jacobian):
        kp_driving = {'value': kp_new_value, 'jacobian': kp_new_jacobian}
        kp_source = {'value': kp_source_value, 'jacobian': kp_source_jacobian}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network.forward_step(source_image, kp_driving, kp_source)
            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)
        return out

    def forward_g1(self, source_image, source_image_padded):
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        self.dense_motion_network.down.skip_padding = True
        self.dense_motion_network.down.skip_interpolate = True
        source_image_padded = self.dense_motion_network.down(source_image_padded)
        return out, source_image_padded

    def forward_g2(self, source_image, kp_driving, kp_source, kp_initial, adaptive_scale):
        kp_driving_value, kp_driving_jacobian = torch.split(kp_driving, [1, 2], 3)
        kp_driving_value = torch.squeeze(kp_driving_value, 3)

        kp_source_value, kp_source_jacobian = torch.split(kp_source, [1, 2], 3)
        kp_source_value = torch.squeeze(kp_source_value, 3)

        kp_initial_value, kp_initial_jacobian = torch.split(kp_initial, [1, 2], 3)
        kp_initial_value = torch.squeeze(kp_initial_value, 3)

        kp_value_diff = (kp_driving_value - kp_initial_value)
        kp_new = kp_value_diff * adaptive_scale + kp_source_value

        jacobian_diff = torch.matmul(kp_driving_jacobian, torch.inverse(kp_initial_jacobian))
        kp_new_jacobian = torch.matmul(jacobian_diff, kp_source_jacobian)

        kp_driving = {'value': kp_new, 'jacobian': kp_new_jacobian}
        kp_source = {'value': kp_source_value, 'jacobian': kp_source_jacobian}
        return self.dense_motion_network.forward_g2(source_image, kp_driving, kp_source)

    def forward_g3(self, input, sparse_motion):
        return self.dense_motion_network.forward_g3(input, sparse_motion)

    def forward_g4(self, out, deformation, occlusion_map):
        out = self.deform_input(out, deformation)
        if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
        out = out * occlusion_map
        return out

    def forward_g5(self, out):
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)
        return out


class OcclusionAwareGeneratorFake(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False):
        super(OcclusionAwareGeneratorFake, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetworkFake(num_kp=num_kp, num_channels=num_channels,
                                                               estimate_occlusion_map=estimate_occlusion_map,
                                                               **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        return source_image

    def forward_g2(self, source_image, kp_driving, kp_source, kp_initial, adaptive_scale):
        kp_driving_value, kp_driving_jacobian = torch.split(kp_driving, [1, 2], 3)
        kp_driving_value = torch.squeeze(kp_driving_value, 3)

        kp_source_value, kp_source_jacobian = torch.split(kp_source, [1, 2], 3)
        kp_source_value = torch.squeeze(kp_source_value, 3)

        kp_initial_value, kp_initial_jacobian = torch.split(kp_initial, [1, 2], 3)
        kp_initial_value = torch.squeeze(kp_initial_value, 3)

        kp_value_diff = (kp_driving_value - kp_initial_value)
        kp_new = kp_value_diff * adaptive_scale + kp_source_value

        jacobian_diff = torch.matmul(kp_driving_jacobian, torch.inverse(kp_initial_jacobian))
        kp_new_jacobian = torch.matmul(jacobian_diff, kp_source_jacobian)

        kp_driving = {'value': kp_new, 'jacobian': kp_new_jacobian}
        kp_source = {'value': kp_source_value, 'jacobian': kp_source_jacobian}
        return self.dense_motion_network.forward_g2(source_image, kp_driving, kp_source)

    def forward_g4(self, out, deformation, occlusion_map):
        out = self.deform_input(out, deformation)
        if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
        out = out * occlusion_map
        return out