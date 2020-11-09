from stn.transformer import bilinear_sampler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.losses import metric_learning

from modules_tf.util import AntiAliasInterpolation2d, make_coordinate_grid


vgg19_feat_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']


def vgg_19(input_tensor=None, input_shape=(None, None, 3)):
    vgg19 = tf.keras.applications.VGG19(input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    outputs = []
    for layer_name in vgg19_feat_layers:
        outputs.append(vgg19.get_layer(layer_name).output)
    model = tf.keras.Model(inputs=vgg19.input, outputs=outputs, trainable=False)

    return model


class ImagePyramid(layers.Layer):
    """
    Create image pyramid for computing pyramid perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramid, self).__init__()
        self.scales = scales
        for scale in scales:
            setattr(self, f'scale_{str(scale).replace(".", "-")}', AntiAliasInterpolation2d(num_channels, scale))

    def call(self, x, **kwargs):
        out_dict = {}
        for scale in self.scales:
            down = getattr(self, f'scale_{str(scale).replace(".", "-")}')
            out_dict['prediction_' + str(scale).replace('-', '.')] = down(x)
        return out_dict


class Transform(layers.Layer):
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        super().__init__()
        noise = tf.random.normal(shape=[bs, 2, 3], mean=0, stddev=kwargs['sigma_affine'] * tf.ones([bs, 2, 3]))
        self.theta = noise + tf.reshape(tf.eye(2, 3), [1, 2, 3])
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.dtype)
            self.control_points = tf.expand_dims(self.control_points, 0)
            self.control_params = tf.random.normal(
                shape=[bs, 1, kwargs['points_tps'] ** 2],
                mean=0,
                stddev=kwargs['sigma_tps'] * tf.ones([bs, 1, kwargs['points_tps'] ** 2])
            )
        else:
            self.tps = False

    @tf.function
    def transform_frame(self, frame):
        grid = tf.expand_dims(make_coordinate_grid(frame.shape[1:3], type=frame.dtype), 0)  # 1, H, W, 2
        grid = tf.reshape(grid, (1, frame.shape[1] * frame.shape[2], 2))
        grid = tf.reshape(self.warp_coordinates(grid), (self.bs, frame.shape[1], frame.shape[2], 2))
        grid = tf.transpose(grid, [0, 3, 1, 2])
        return bilinear_sampler(frame, grid[:, 0, :, :], grid[:, 1, :, :])
        # return F.grid_sample(frame, grid, padding_mode="reflection")

    @tf.function
    def warp_coordinates(self, coordinates):  # 1, H*W, 2
        theta = self.theta  # B, 2, 3
        theta = tf.expand_dims(theta, 1)  # B, 1, 2, 3
        transformed = tf.matmul(theta[:, :, :, :2], tf.expand_dims(coordinates, -1)) + theta[:, :, :, 2:]
        transformed = tf.squeeze(transformed, -1)

        if self.tps:
            control_points = self.control_points
            control_params = self.control_params
            distances = tf.reshape(coordinates, (coordinates.shape[0], -1, 1, 2)) - tf.reshape(control_points, (1, 1, -1, 2))
            distances = tf.reduce_sum(tf.abs(distances), -1)

            result = distances ** 2
            result = result * tf.math.log(distances + 1e-6)
            result = result * control_params
            result = tf.reshape(tf.reduce_sum(result, axis=2), (self.bs, coordinates.shape[1], 1))
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates, grad_tape):
        new_coordinates = self.warp_coordinates(coordinates)
        # test_jacobian = grad_tape.jacobian(new_coordinates, coordinates)
        grad_x = grad_tape.gradient(tf.reduce_sum(new_coordinates[..., 0]), coordinates)  # B, 10, 2
        grad_y = grad_tape.gradient(tf.reduce_sum(new_coordinates[..., 1]), coordinates)  # B, 10, 2

        # grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        # grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = tf.concat([tf.expand_dims(grad_x, -2), tf.expand_dims(grad_y, -2)], axis=-2)  # B, 10, 2, 2
        return jacobian


class GeneratorFullModel(layers.Layer):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramid(self.scales, generator.num_channels)
        self.grad_tape = None
        self.kp_loss_weight = 1.0
        self.bs = train_params['batch_size']
        self.use_kp_loss = train_params['use_kp_loss']

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = vgg_19()

        # self.transform =

    def call(self, inputs, **kwargs):
        x_source, x_driving = inputs
        kp_source_value, kp_source_jacobian = self.kp_extractor(x_source)
        kp_driving_value, kp_driving_jacobian = self.kp_extractor(x_driving)

        generated = {}
        generated_prediction = self.generator(
            (x_source, kp_driving_value, kp_driving_jacobian, kp_source_value, kp_source_jacobian)
        )
        generated.update({
            'kp_source_value': kp_source_value,
            'kp_driving_value': kp_driving_value,
            'prediction': generated_prediction,
        })

        loss_values = {}

        pyramide_real = self.pyramid(x_driving)
        pyramide_generated = self.pyramid(generated_prediction)

        # kp detector normalize loss
        if self.use_kp_loss:
            kp_source_loss = 0.
            kp_driving_loss = 0.
            kp_loss_koef = 0.7
            for kp in kp_source_value:
                distances = metric_learning.pairwise_distance(kp)
                v, idx = tf.nn.top_k(-distances, 2)
                mins = -v[:, 1]  # 10
                # tf.print(mins)
                kp_source_loss += tf.reduce_sum(kp_loss_koef - mins)

            for kp in kp_driving_value:
                distances = metric_learning.pairwise_distance(kp)
                v, idx = tf.nn.top_k(-distances, 2)
                mins = -v[:, 1]  # 10
                # tf.print(mins)
                kp_driving_loss += tf.reduce_sum(kp_loss_koef - mins)

            kp_loss = (kp_source_loss + kp_driving_loss) / self.bs
            loss_values['kp_loss'] = kp_loss * self.kp_loss_weight

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = tf.reduce_mean(tf.abs(x_vgg[i] - tf.stop_gradient(y_vgg[i])))
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator((pyramide_generated, tf.stop_gradient(kp_driving_value)))
            discriminator_maps_real = self.discriminator((pyramide_real, tf.stop_gradient(kp_driving_value)))
            value_total = 0
            for scale in self.disc_scales:
                key = f'prediction_map_{scale}'
                value = tf.reduce_mean((1 - discriminator_maps_generated[key]) ** 2)
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = f'feature_maps_{scale}'
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = tf.reduce_mean(tf.abs(a - b))
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            # if self.transform is None:
            #     self.transform = Transform(x_driving.shape[0], **self.train_params['transform_params'])

            transform = Transform(self.train_params['batch_size'], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x_driving)
            transformed_kp_value, transformed_kp_jacobian = self.kp_extractor(transformed_frame)

            # generated['transformed_frame'] = transformed_frame
            # generated['transformed_kp_value'] = transformed_kp_value
            # generated['transformed_kp_jacobian'] = transformed_kp_jacobian

            # Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = tf.reduce_mean(tf.abs(kp_driving_value - transform.warp_coordinates(transformed_kp_value)))
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            # jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = tf.matmul(
                    transform.jacobian(transformed_kp_value, self.grad_tape),
                    transformed_kp_jacobian
                )

                normed_driving = tf.linalg.inv(kp_driving_jacobian)
                normed_transformed = jacobian_transformed
                value = tf.matmul(normed_driving, normed_transformed)

                eye = tf.reshape(tf.eye(2), [1, 1, 2, 2])

                value = tf.reduce_mean(tf.abs(eye - value))
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return loss_values, generated


class DiscriminatorFullModel(layers.Layer):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramid(self.scales, generator.num_channels)

        self.loss_weights = train_params['loss_weights']

    def call(self, inputs, **kwargs):
        x_driving, kp_driving_value, generated_prediction = inputs
        pyramide_real = self.pyramid(x_driving)
        pyramide_generated = self.pyramid(tf.stop_gradient(generated_prediction))

        discriminator_maps_generated = self.discriminator((pyramide_generated, tf.stop_gradient(kp_driving_value)))
        discriminator_maps_real = self.discriminator((pyramide_real, tf.stop_gradient(kp_driving_value)))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = f'prediction_map_{scale}'
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * tf.reduce_mean(value)
        loss_values['disc_gan'] = value_total

        return loss_values
