import os

import face_alignment
from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d


class Landmarks(nn.Module):
    def __init__(self):
        super(Landmarks, self).__init__()

        network_size = 4
        fan = face_alignment.api.FAN(network_size)
        base_path = os.path.join(face_alignment.api.appdata_dir('face_alignment'), "data")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        network_name = f'2DFAN-{str(network_size)}.pth.tar'
        fan_path = os.path.join(base_path, network_name)
        if not os.path.isfile(fan_path):
            print("Downloading the Face Alignment Network(FAN). Please wait...")

            fan_temp_path = os.path.join(base_path, network_name + '.download')

            if os.path.isfile(fan_temp_path):
                os.remove(os.path.join(fan_temp_path))

            face_alignment.api.request_file.urlretrieve(
                "https://www.adrianbulat.com/downloads/python-fan/" +
                network_name, os.path.join(fan_temp_path))

            os.rename(os.path.join(fan_temp_path), os.path.join(fan_path))
        fan_weights = torch.load(
            fan_path,
            map_location=torch.device('cpu')
        )
        fan.load_state_dict(fan_weights)
        fan.eval()

        if torch.cuda.is_available():
            fan.to(torch.device('cuda'))

        for p in fan.parameters():
            p.requires_grad_(False)
        self.fan = fan  # .requires_grad_(False)
        # self.requires_grad_(False)

    def forward(self, x):
        heatmap = self.fan(x)[-1]

        points, _ = face_alignment.api.get_preds_fromhm(heatmap.cpu())

        points = points * 4
        points = (points - 127.5) / 127.5
        return heatmap, points.to(heatmap.device)


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0, use_landmarks=False):
        super(KPDetector, self).__init__()

        self.use_landmarks = use_landmarks
        if use_landmarks:
            num_kp = 68
            self.fan = Landmarks()#.requires_grad_(False)

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            if self.use_landmarks:
                self.jacobian = nn.Conv2d(
                    in_channels=self.predictor.out_filters,
                    out_channels=4 * self.num_jacobian_maps,
                    kernel_size=(7, 7),
                    padding=3
                )
            else:
                self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                          out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape  # B, 10, 58, 58
        heatmap = heatmap.unsqueeze(-1)  # B, 10, 58, 58, 1
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)  # 1, 1, 58, 58, 2
        value = (heatmap * grid).sum(dim=(2, 3))  # B, 10, 2
        kp = {'value': value}

        return kp

    def forward(self, x):
        x_orig = x
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)  # B, 35, 64, 64
        if not self.use_landmarks:
            prediction = self.kp(feature_map)  # B, 10, 58, 58

            final_shape = prediction.shape
            heatmap = prediction.view(final_shape[0], final_shape[1], -1)
            heatmap = F.softmax(heatmap / self.temperature, dim=2)
            heatmap = heatmap.view(*final_shape)  # B, 10, 1, 58, 58
            out = self.gaussian2kp(heatmap)
        else:
            heatmap, points = self.fan(x_orig)  # heatmap should be [0, 1], points [-1, 1]

            out = {'value': points}
            #  heatmap B, 68, 64, 64
            #  points B, 68, 2
            # feature_map = heatmap
            final_shape = heatmap.shape
            # final_shape = torch.Size([heatmap.shape[0], self.num_jacobian_maps, 58, 58])
        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])  # B, num_kp, 4, 58, 58
            heatmap = heatmap.unsqueeze(2)  # B, num_kp, 1, 58, 58

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)  # B, num_kp, 2, 2
            out['jacobian'] = jacobian

        return out
