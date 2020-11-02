import tensorflow as tf
from tensorflow.keras import layers
from torch import nn
import torch.nn.functional as F
import torch
from modules_tf.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian


class DenseMotionNetwork(tf.keras.Model):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(
            block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
            max_features=max_features, num_blocks=num_blocks
        )

        self.mask = layers.Conv2D(num_kp + 1, kernel_size=(7, 7), padding='SAME')

        if estimate_occlusion_map:
            self.occlusion = layers.Conv2D(1, kernel_size=(7, 7), padding='SAME')
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def clean_up(self):
        self.hourglass = None
        self.mask = None
        self.occlusion = None
        self.down = None

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[1:3]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = tf.zeros([heatmap.shape[0], spatial_size[0], spatial_size[1], 1], dtype=heatmap.dtype)
        heatmap = tf.concat([zeros, heatmap], axis=-1)
        heatmap = tf.expand_dims(heatmap, -1)  # B, H, W, C, 1
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving_value, kp_driving_jacobian,
                              kp_source_value, kp_source_jacobian):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, h, w, _ = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source_value.dtype)
        # identity_grid = tf.expand_dims(tf.expand_dims(identity_grid, 0), 3)  # 1, h, w, 1, 2
        identity_grid = tf.reshape(identity_grid, [1, 1, h, w, 2])
        coordinate_grid = identity_grid - tf.reshape(kp_driving_value, [bs, self.num_kp, 1, 1, 2])

        jacobian = tf.matmul(kp_source_jacobian, tf.linalg.inv(kp_driving_jacobian))
        jacobian = tf.expand_dims(tf.expand_dims(jacobian, -3), -3)  # B, 10, 1, 1, 2, 2
        jacobian = jacobian.repeat(1, 1, h, w, 1, 1)  # B, 10, 64, 64, 2, 2
        coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))  # B, 10, 64, 64, 2, 1
        coordinate_grid = coordinate_grid.squeeze(-1)  # B, 10, 64, 64, 2

        driving_to_source = coordinate_grid + kp_source_value.view(bs, self.num_kp, 1, 1, 2)  # B, 10, 64, 64, 2

        # adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)  # B, 1, 64, 64, 2
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)  # B, 11, 64, 64, 2
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape  # B, 3, 64, 64
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1,
                                                                      1)  # B, 11, 1, 3, 64, 64
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving,
                                                                     kp_source)  # B, 11, 1, 64, 64
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)  # B, 11, 64, 64, 2
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)  # B, 11, 3, 64, 64
        out_dict['sparse_deformed'] = deformed_source

        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)  # B, 44, 64, 64

        prediction = self.hourglass(input)  # B, 108, 64, 64

        mask = self.mask(prediction)  # B, 11, 64, 64
        mask = F.softmax(mask, dim=1)  # B, 11, 64, 64
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)  # B, 11, 1, 64, 64
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)  # B, 11, 2, 64, 64
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)  # B, 64, 64, 2

        out_dict['deformation'] = deformation

        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))  # B, 1, 64, 64
            out_dict['occlusion_map'] = occlusion_map

        return out_dict

    def forward_setup(self, source_image):
        if self.scale_factor != 1:
            source_image = self.down(source_image)
        return source_image

    def forward_step(self, source_image, kp_driving, kp_source):
        bs, _, h, w = source_image.shape
        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        # out_dict['sparse_deformed'] = deformed_source

        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        # out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation

        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict

    def forward_g2(self, source_image, kp_driving, kp_source):
        bs, _, h, w = source_image.shape
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)
        return input, sparse_motion

    def forward_g3(self, input, sparse_motion):
        prediction = self.hourglass(input)
        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)
        occlusion_map = torch.sigmoid(self.occlusion(prediction))
        return deformation, occlusion_map
