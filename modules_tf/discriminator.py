import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons import layers as addons_layers

from modules_tf.util import kp2gaussian


class DownBlock2d(layers.Layer):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = layers.Conv2D(out_features, kernel_size=kernel_size, padding='SAME')

        if sn:
            self.conv = addons_layers.SpectralNormalization(self.conv)

        if norm:
            self.norm = addons_layers.InstanceNormalization()
        else:
            self.norm = None
        self.pool = pool
        if self.pool:
            self.avg_pool = layers.AvgPool2D(pool_size=(2, 2))

    def call(self, x, **kwargs):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = tf.keras.activations.relu(out, 0.2)
        if self.pool:
            out = self.avg_pool(out)
        return out


class Discriminator(layers.Layer):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(Discriminator, self).__init__()

        self.num_blocks = num_blocks
        for i in range(num_blocks):
            block = DownBlock2d(
                min(max_features, block_expansion * (2 ** (i + 1))),
                norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn
            )
            setattr(self, f'down_block{i}', block)

        self.conv = layers.Conv2D(1, kernel_size=1, padding='SAME')
        if sn:
            self.conv = addons_layers.SpectralNormalization(self.conv)
        self.use_kp = use_kp
        self.kp_variance = kp_variance

    def call(self, inputs, training=None, mask=None):
        x, kp_value = inputs
        feature_maps = []
        out = x
        heatmap = kp2gaussian(kp_value, x.shape[1:3], self.kp_variance)
        out = tf.concat([out, heatmap], axis=-1)

        for i in range(self.num_blocks):
            res = getattr(self, f'down_block{i}')(out)
            feature_maps.append(res)
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map


class MultiScaleDiscriminator(tf.keras.Model):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        for scale in scales:
            setattr(self, f'scale_{str(scale).replace(".", "-")}', Discriminator(**kwargs))

    def call(self, inputs, is_training=True, **kwargs):
        x, kp_value = inputs
        out_dict = {}
        for scale in self.scales:
            disc = getattr(self, f'scale_{str(scale).replace(".", "-")}')
            key = f'prediction_{scale}'
            feature_maps, prediction_map = disc((x[key], kp_value))
            out_dict[f'feature_maps_{scale}'] = feature_maps
            out_dict[f'prediction_map_{scale}'] = prediction_map
        return out_dict
