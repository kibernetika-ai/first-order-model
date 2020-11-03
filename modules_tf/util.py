import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops.gen_image_ops import resize_nearest_neighbor


def kp2gaussian(kp_value, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp_value  # B, 10, 2

    coordinate_grid = make_coordinate_grid(spatial_size, mean.dtype)  # 64, 64, 2
    coordinate_grid = tf.reshape(
        coordinate_grid,
        [1, coordinate_grid.shape[0], coordinate_grid.shape[1], 1, 2]
    )  # 1, 64, 64, 1, 2
    repeats = [mean.shape[0], 1, 1, mean.shape[1], 1]  # B, 1, 1, 10, 1
    coordinate_grid = tf.tile(coordinate_grid, [*repeats])  # B, 64, 64, 10, 2

    # Preprocess kp shape
    mean = tf.reshape(mean, [mean.shape[0], 1, 1, mean.shape[1], mean.shape[2]])  # B, 1, 1, 10, 2

    mean_sub = (coordinate_grid - mean)  # B, 64, 64, 10, 2

    out = tf.exp(-0.5 * tf.reduce_sum(mean_sub ** 2, -1) / kp_variance)  # B, 64, 64, 10

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = tf.range(w, dtype=type)
    y = tf.range(h, dtype=type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = tf.repeat(tf.reshape(y, [-1, 1]), w, 1)
    xx = tf.repeat(tf.reshape(x, [1, -1]), h, 0)
    # yy = y.view(-1, 1).repeat(1, w)
    # xx = x.view(1, -1).repeat(h, 1)

    meshed = tf.concat([tf.expand_dims(xx, 2), tf.expand_dims(yy, 2)], 2)

    return meshed


class ResBlock2d(layers.Layer):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = layers.Conv2D(in_features, kernel_size=kernel_size, padding='same')
        self.conv2 = layers.Conv2D(in_features, kernel_size=kernel_size, padding='same')
        self.norm1 = layers.BatchNormalization()
        self.norm2 = layers.BatchNormalization()

    def call(self, x, **kwargs):
        out = self.norm1(x)
        out = tf.keras.activations.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = tf.keras.activations.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(layers.Layer):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, out_features, kernel_size=(3, 3), groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = layers.Conv2D(out_features, kernel_size=kernel_size, padding='same')
        self.norm = layers.BatchNormalization()
        self.up = layers.UpSampling2D()

    def call(self, x, **kwargs):
        out = self.up(x)
        out = self.conv(out)
        out = self.norm(out)
        out = tf.keras.activations.relu(out)
        return out


def build_model(block_expansion=64,
                max_features=512, num_down_blocks=2, num_bottleneck_blocks=6, num_channels=3):
    n4_out = layers.Input((64, 64, 256))
    inputs = n4_out

    bottleneck = tf.keras.Sequential()
    in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
    for i in range(num_bottleneck_blocks):
        bottleneck.add(ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
    up_blocks = []
    for i in range(num_down_blocks):
        # in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
        out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
        up_blocks.append(UpBlock2d(out_features, kernel_size=(3, 3)))
    up_blocks = tf.keras.Sequential(layers=up_blocks)

    final = layers.Conv2D(num_channels, kernel_size=(7, 7), padding='same')

    out = bottleneck(inputs)
    out = up_blocks(out)
    out = final(out)
    out = tf.keras.activations.sigmoid(out)

    # outputs = tf.clip_by_value(outputs, 0.0, 1.0)
    model = tf.keras.Model(inputs=[inputs], outputs=[out])
    return model


class DownBlock2d(layers.Layer):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, out_features, kernel_size=3):
        super(DownBlock2d, self).__init__()
        self.conv = layers.Conv2D(out_features, kernel_size=kernel_size, padding='same')
        self.norm = layers.BatchNormalization()
        self.pool = layers.AvgPool2D(pool_size=(2, 2))

    def call(self, x, **kwargs):
        out = self.conv(x)
        out = self.norm(out)
        out = tf.keras.activations.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(layers.Layer):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, out_features, kernel_size=3):
        super(SameBlock2d, self).__init__()
        self.conv = layers.Conv2D(out_features, kernel_size=kernel_size, padding='same')
        self.norm = layers.BatchNormalization()

    def call(self, x, **kwargs):
        out = self.conv(x)
        out = self.norm(out)
        out = tf.keras.activations.relu(out)
        return out


class Encoder(layers.Layer):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        self.num_blocks = num_blocks
        # down_blocks = []
        for i in range(num_blocks):
            block = DownBlock2d(
                min(max_features, block_expansion * (2 ** (i + 1))),
                kernel_size=3
            )
            setattr(self, f'down_block{i}', block)
            # down_blocks.append(block)
        # self.down_blocks = tf.keras.Sequential(down_blocks)

    def call(self, x, **kwargs):
        outs = [x]
        for i in range(self.num_blocks):
            res = getattr(self, f'down_block{i}')(outs[-1])
            outs.append(res)
        return outs


class Decoder(layers.Layer):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        # up_blocks = []

        self.num_blocks = num_blocks
        for i in range(num_blocks)[::-1]:
            out_filters = min(max_features, block_expansion * (2 ** i))
            block = UpBlock2d(out_filters, kernel_size=3)
            setattr(self, f'up_block{i}', block)

            # up_blocks.append()

        # self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def call(self, x, **kwargs):
        out = x.pop()
        for i in range(self.num_blocks)[::-1]:
            out = getattr(self, f'up_block{i}')(out)
            skip = x.pop()
            out = tf.concat([out, skip], axis=3)
        return out


class Hourglass(layers.Layer):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def call(self, x, **kwargs):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(layers.Layer):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = tf.meshgrid(
            [tf.range(kernel_size[0], dtype=tf.float32)],
            [tf.range(kernel_size[1], dtype=tf.float32)],
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= tf.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / tf.reduce_sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = tf.reshape(kernel, [1, 1, *kernel.shape])
        kernel = tf.repeat(kernel, channels, 0)

        self.kernel = tf.transpose(tf.constant(kernel, name='kernel'), [2, 3, 1, 0])
        # self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        self.skip_padding = False
        self.skip_interpolate = False

    def call(self, input, **kwargs):
        if self.scale == 1.0:
            return input
        if self.skip_padding:
            out = input
        else:
            out = tf.keras.backend.spatial_2d_padding(input, ((self.ka, self.kb), (self.ka, self.kb)))
        out = tf.nn.conv2d(out, self.kernel, strides=1, padding='VALID')
        if self.skip_interpolate:
            return out

        size = (tf.cast(out.shape[1] * self.scale, tf.int32), tf.cast(out.shape[2] * self.scale, tf.int32))
        # out = tf.image.resize(out, (64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        out = resize_nearest_neighbor(out, size)
        # out = tf.keras.backend.resize_images(out, 0.25, 0.25, 'channels_last', interpolation='nearest')
        # out = self.interpolate(out)

        return out
