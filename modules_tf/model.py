from stn.transformer import bilinear_sampler
import tensorflow as tf
from tensorflow.keras import layers
import torch
from modules_tf.util import AntiAliasInterpolation2d, make_coordinate_grid
import numpy as np
from torch.autograd import grad


vgg19_feat_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']


def VGG19(input_tensor=None, input_shape=(256, 256, 3)):
    vgg19 = tf.keras.applications.VGG19(input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    outputs = []
    for layer_name in vgg19_feat_layers:
        outputs.append(vgg19.get_layer(layer_name).output)
    model = tf.keras.Model(inputs=vgg19.input, outputs=outputs)

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


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
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

    def transform_frame(self, frame):
        grid = tf.expand_dims(make_coordinate_grid(frame.shape[1:3], type=frame.dtype), 0)
        grid = tf.reshape(grid, (1, frame.shape[1] * frame.shape[2], 2))
        grid = tf.reshape(self.warp_coordinates(grid), (self.bs, frame.shape[1], frame.shape[2], 2))
        return bilinear_sampler(frame, grid[:, 0, :, :], grid[:, 1, :, :])
        # return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):  # 1, H*W, 2
        theta = self.theta  # B, 2, 3
        theta = tf.expand_dims(theta, 1)  # B, 1, 2, 3
        transformed = torch.matmul(theta[:, :, :, :2], tf.expand_dims(coordinates, -1)) + theta[:, :, :, 2:]
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

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        # grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_x = tf.gradients(tf.reduce_sum(new_coordinates[..., 0]), coordinates)  # B, 10, 2
        # grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        grad_y = tf.gradients(tf.reduce_sum(new_coordinates[..., 1]), coordinates)  # B, 10, 2
        jacobian = tf.concat([tf.expand_dims(grad_x[0], -2), tf.expand_dims(grad_y[0], -2)], axis=-2)  # B, 10, 2, 2
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
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
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, x):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
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
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
