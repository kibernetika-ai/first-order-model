import tensorflow as tf
from tensorflow.keras import layers
from modules_tf.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d


class KPDetector(tf.keras.Model):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = layers.Conv2D(num_kp, kernel_size=(7, 7), padding='same')

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = layers.Conv2D(
                4 * self.num_jacobian_maps, kernel_size=(7, 7), padding='same', kernel_initializer='zeros'
            )
            # self.jacobian.weight.data.zero_()
            # self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        # if self.scale_factor != 1:
        self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = tf.expand_dims(heatmap, -1)
        grid = tf.expand_dims(make_coordinate_grid(shape[2:], heatmap.dtype), 0)
        grid = tf.expand_dims(grid, -1)
        value = tf.reduce_sum(heatmap * grid, axis=[1, 2])
        kp = {'value': value}

        return kp

    def call(self, x, training=None, mask=None):
        # if self.scale_factor != 1:
        x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = tf.reshape(prediction, [final_shape[0], -1, final_shape[1]])  # [N, HxW, C]
        heatmap = tf.nn.softmax(heatmap / self.temperature, axis=1)  # [N, HxW, C]
        heatmap = tf.reshape(heatmap, [*final_shape])  # [N, H, W, C]

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = tf.reshape(
                jacobian_map,
                [final_shape[0], self.num_jacobian_maps, 4, final_shape[2], final_shape[3]]
            )
            heatmap = tf.expand_dims(heatmap, 2)

            jacobian = heatmap * jacobian_map
            jacobian = tf.reshape(jacobian, [final_shape[0], final_shape[1], 4, -1])
            jacobian = tf.reduce_sum(jacobian, axis=-1)
            jacobian = tf.reshape(jacobian, [jacobian.shape[0], jacobian.shape[1], 2, 2])
            out['jacobian'] = jacobian

        return out

    def forward_script(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = tf.reshape(prediction, [final_shape[0], -1, final_shape[1]])  # [N, HxW, C]
        heatmap = tf.nn.softmax(heatmap / self.temperature, axis=1)  # [N, HxW, C]
        heatmap = tf.reshape(heatmap, [*final_shape])  # [N, H, W, C]

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = tf.reshape(
                jacobian_map,
                [final_shape[0], self.num_jacobian_maps, 4, final_shape[2], final_shape[3]]
            )
            heatmap = tf.expand_dims(heatmap, 2)

            jacobian = heatmap * jacobian_map
            jacobian = tf.reshape(jacobian, [final_shape[0], final_shape[1], 4, -1])
            jacobian = tf.reduce_sum(jacobian, axis=-1)
            jacobian = tf.reshape(jacobian, [jacobian.shape[0], jacobian.shape[1], 2, 2])

        return out['value'], jacobian
