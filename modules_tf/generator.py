from stn.transformer import bilinear_sampler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops.gen_image_ops import resize_bilinear

from modules_tf.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules_tf.dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(tf.keras.Model):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp, num_channels=num_channels,
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params
            )
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(block_expansion, kernel_size=(7, 7))

        self.num_down_blocks = num_down_blocks
        for i in range(num_down_blocks):
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            block = DownBlock2d(out_features, kernel_size=(3, 3))
            setattr(self, f'down_block{i}', block)

        self.num_up_blocks = num_down_blocks
        for i in range(num_down_blocks):
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            block = UpBlock2d(out_features, kernel_size=(3, 3))
            setattr(self, f'up_block{i}', block)

        self.bottleneck = tf.keras.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add(ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = layers.Conv2D(num_channels, kernel_size=(7, 7), padding='SAME')
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
        _, h_old, w_old, _ = deformation.shape  # B, 64, 64, 2
        _, h, w, _ = inp.shape
        if h_old != h or w_old != w:
            # deformation = deformation.permute(0, 3, 1, 2)
            deformation = resize_bilinear(deformation, size=(h, w))
        deformation = tf.transpose(deformation, [0, 3, 1, 2])
        return bilinear_sampler(inp, deformation[:, 0, :, :], deformation[:, 1, :, :])

    def call(self, inputs, training=None, mask=None):
        source_image, kp_driving_value, kp_driving_jacobian, kp_source_value, kp_source_jacobian = inputs
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(self.num_down_blocks):
            down_block = getattr(self, f'down_block{i}')
            out = down_block(out)

        # Transforming feature representation according to deformation and occlusion
        # output_dict = {}
        deformation, occlusion_map = self.dense_motion_network(
            (source_image, kp_driving_value, kp_driving_jacobian, kp_source_value, kp_source_jacobian)
        )
        # output_dict['mask'] = dense_motion['mask']
        # output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

        # occlusion_map = dense_motion['occlusion_map']
        # output_dict['occlusion_map'] = occlusion_map

        # deformation = dense_motion['deformation']
        # out [B, 64, 64, 256]
        # deformation [B, 64, 64, 2]
        out = self.deform_input(out, deformation)  # B, 64, 64, 256

        if out.shape[1] != occlusion_map.shape[1] or out.shape[2] != occlusion_map.shape[2]:
            occlusion_map = resize_bilinear(occlusion_map, (out.shape[1:3]))
        out = out * occlusion_map

        # output_dict["deformed"] = self.deform_input(source_image, deformation)

        # Decoding part
        out = self.bottleneck(out)  # B, 64, 64, 256
        for i in range(self.num_up_blocks):
            up_block = getattr(self, f'up_block{i}')
            out = up_block(out)
        # B, 256, 256, 64
        out = self.final(out)  # B, 256, 256, 3
        out = tf.nn.sigmoid(out)

        # output_dict["prediction"] = out

        return out

    def forward_setup(self, source_image):
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        scaled_image = self.dense_motion_network.forward_setup(source_image)
        return out, scaled_image

    def forward_step(self, out, source_image, kp_new_value, kp_new_jacobian, kp_source_value, kp_source_jacobian):
        dense_motion = self.dense_motion_network.forward_step(
            source_image, kp_new_value, kp_new_jacobian, kp_source_value, kp_source_jacobian
        )
        occlusion_map = dense_motion['occlusion_map']

        deformation = dense_motion['deformation']
        out = self.deform_input(out, deformation)

        occlusion_map = resize_bilinear(occlusion_map, (out.shape[1:3]))
        out = out * occlusion_map

        # Decoding part
        out = self.bottleneck(out)  # B, 64, 64, 256
        for i in range(len(self.num_up_blocks)):
            up_block = getattr(self, f'up_block{i}')
            out = up_block(out)
        # B, 256, 256, 64
        out = self.final(out)  # B, 256, 256, 3
        out = tf.nn.sigmoid(out)
        return out
