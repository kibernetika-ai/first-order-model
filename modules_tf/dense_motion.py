from stn import transformer
import tensorflow as tf
from tensorflow.keras import layers

from modules_tf.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian


class DenseMotionNetwork(layers.Layer):
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

        self.occlusion = layers.Conv2D(1, kernel_size=(7, 7), padding='SAME')

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

    @tf.function
    def create_heatmap_representations(self, source_image, kp_driving_value, kp_source_value):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = tf.shape(source_image)[1:3]

        gaussian_driving = kp2gaussian(kp_driving_value, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source_value, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = tf.zeros([tf.shape(heatmap)[0], spatial_size[0], spatial_size[1], 1], dtype=heatmap.dtype)
        heatmap = tf.concat([zeros, heatmap], axis=-1)
        heatmap = tf.expand_dims(heatmap, -1)  # B, H, W, C, 1
        return heatmap

    @tf.function
    def create_sparse_motions(self, source_image, kp_driving_value, kp_driving_jacobian_inv,
                              kp_source_value, kp_source_jacobian):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        shape = tf.shape(source_image)
        bs, h, w = shape[0], shape[1], shape[2]
        identity_grid = make_coordinate_grid((h, w), type=kp_source_value.dtype)
        # identity_grid = tf.expand_dims(tf.expand_dims(identity_grid, 0), 3)  # 1, h, w, 1, 2
        identity_grid = tf.reshape(identity_grid, [1, h, w, 1, 2])  # 1, 64, 64, 1, 2
        coordinate_grid = identity_grid - tf.reshape(kp_driving_value, [bs, 1, 1, self.num_kp, 2])  # B, 64, 64, 10, 2

        jacobian = tf.matmul(kp_source_jacobian, kp_driving_jacobian_inv)
        jacobian = tf.expand_dims(tf.expand_dims(jacobian, 1), 1)  # B, 1, 1, 10, 2, 2
        jacobian = tf.tile(jacobian, [1, h, w, 1, 1, 1])  # B, 64, 64, 10, 2, 2
        coordinate_grid = tf.matmul(jacobian, tf.expand_dims(coordinate_grid, -1))  # B, 64, 64, 10, 2, 1
        coordinate_grid = tf.squeeze(coordinate_grid, axis=-1)  # B, 64, 64, 10, 2

        driving_to_source = coordinate_grid + tf.reshape(kp_source_value, [bs, 1, 1, self.num_kp, 2])  # B,64,64,10,2

        # adding background feature
        identity_grid = tf.tile(identity_grid, [bs, 1, 1, 1, 1])  # B, 64, 64, 1, 2
        sparse_motions = tf.concat([identity_grid, driving_to_source], axis=3)  # B, 64, 64, 11, 2
        return sparse_motions

    @tf.function
    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        shape = tf.shape(source_image)  # B, 64, 64, 3
        bs, h, w = shape[0], shape[1], shape[2]
        source_repeat = tf.tile(source_image, [self.num_kp + 1, 1, 1, 1])  # B*11, 64, 64, 3

        # source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)  # B*11, 64, 64, 3
        # sparse_motions B, 64, 64, 11, 2
        sparse_motions = tf.reshape(
            tf.transpose(sparse_motions, [0, 3, 4, 1, 2]),  # B, 11, 2, 64, 64
            [bs * (self.num_kp + 1), 2, h, w]
        )  # B*11, 2, 64, 64
        sparse_deformed = transformer.bilinear_sampler(
            source_repeat, sparse_motions[:, 0, :, :], sparse_motions[:, 1, :, :]
        )  # B*11, 64, 64, 3
        # sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = tf.reshape(sparse_deformed, [bs, self.num_kp + 1, h, w, -1])  # B, 11, 64, 64, 3
        sparse_deformed = tf.transpose(sparse_deformed, [0, 2, 3, 1, 4])  # B, 64, 64, 11, 3
        return sparse_deformed

    def call(self, inputs, **kwargs):
        source_image, kp_driving_value, kp_driving_jacobian, kp_source_value, kp_source_jacobian = inputs
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        shape = tf.shape(source_image)
        bs, h, w = shape[0], shape[1], shape[2]

        # out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(
            source_image, kp_driving_value, kp_source_value
        )  # B, 64, 64, 11, 1
        sparse_motion = self.create_sparse_motions(
            source_image,
            kp_driving_value, kp_driving_jacobian,
            kp_source_value, kp_source_jacobian
        )  # B, 64, 64, 11, 2
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)  # B, 64, 64, 11, 3
        # out_dict['sparse_deformed'] = deformed_source

        input = tf.concat([heatmap_representation, deformed_source], axis=-1)  # B, 64, 64, 11, 4
        input = tf.reshape(input, [bs, h, w, -1])  # B, 64, 64, 44

        prediction = self.hourglass(input)  # B, 64, 64, 108

        mask = self.mask(prediction)  # B, 64, 64, 11
        mask = tf.nn.softmax(mask, axis=-1)  # B, 64, 64, 11
        # out_dict['mask'] = mask
        mask = tf.expand_dims(mask, -1)  # B, 64, 64, 11, 1
        deformation = tf.reduce_sum(sparse_motion * mask, axis=3)  # B, 64, 64, 2
        # deformation = deformation.permute(0, 2, 3, 1)  # B, 64, 64, 2

        # out_dict['deformation'] = deformation

        # Sec. 3.2 in the paper
        occlusion_map = tf.keras.activations.sigmoid(self.occlusion(prediction))  # B, 64, 64, 1
        # out_dict['occlusion_map'] = occlusion_map

        return deformation, occlusion_map
