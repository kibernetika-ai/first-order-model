import argparse
import logging
import os
from shutil import copy
import time
import yaml

import cv2
import numpy as np
import tensorflow as tf

from modules_tf.discriminator import MultiScaleDiscriminator
from modules_tf.frames_dataset import FramesDataset
from modules_tf.generator import OcclusionAwareGenerator
from modules_tf.keypoint_detector import KPDetector
from modules_tf import model

# from reconstruction import reconstruction
# from animate import animate


LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate", "export"])
    parser.add_argument("--data-dir")
    parser.add_argument("--train-dir", default='train')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument("--enable-jacobian", action='store_true')
    parser.add_argument("--disable-jacobian", action='store_true')
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--disable-id-sampling", action='store_true')
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-kp", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument('--resolution', default="256x256")
    parser.set_defaults(verbose=False)

    return parser.parse_args()


def parse_resolution(res):
    splitted = res.split('x')
    if len(splitted) != 2:
        raise RuntimeError("Resolution must be in form WxH")

    return int(splitted[0]), int(splitted[1])


def main():
    opt = parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
        level='INFO'
    )
    logging.root.setLevel(logging.INFO)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        LOG.info("=" * 50)
        LOG.info(f"Set memory growth to {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
        LOG.info("=" * 50)

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    log_dir = opt.train_dir

    if opt.data_dir:
        config['dataset_params']['root_dir'] = opt.data_dir
    if opt.epochs != 0:
        config['train_params']['num_epochs'] = opt.epochs
    if opt.repeats != 0:
        config['train_params']['num_repeats'] = opt.repeats
    if opt.disable_id_sampling:
        config['dataset_params']['id_sampling'] = False

    config['train_params']['batch_size'] = opt.batch_size
    # config['model_params']['kp_detector_params']['use_landmarks'] = opt.use_landmarks
    # if not opt.use_landmarks:
    config['train_params']['loss_weights']['equivariance_jacobian'] = 10
    config['model_params']['common_params']['estimate_jacobian'] = True
    if opt.disable_jacobian:
        config['model_params']['common_params']['estimate_jacobian'] = False
    if opt.enable_jacobian:
        config['model_params']['common_params']['estimate_jacobian'] = True

    # if opt.use_landmarks:
    #     config['model_params']['common_params']['num_kp'] = 68
    # elif opt.num_kp:
    #     config['model_params']['common_params']['num_kp'] = opt.num_kp

    generator = OcclusionAwareGenerator(
        **config['model_params']['generator_params'],
        **config['model_params']['common_params']
    )

    if opt.verbose:
        LOG.info(generator.summary())

    discriminator = MultiScaleDiscriminator(
        **config['model_params']['discriminator_params'],
        **config['model_params']['common_params']
    )
    if opt.verbose:
        LOG.info(discriminator.summary())

    kp_detector = KPDetector(
        **config['model_params']['kp_detector_params'],
        **config['model_params']['common_params']
    )

    if opt.verbose:
        LOG.info(kp_detector.summary())

    dataset = FramesDataset(
        is_train=True,
        repeats=opt.repeats,
        **config['dataset_params'],
    )
    if len(dataset) < 10 and opt.repeats < 10:
        dataset.repeats = 100 // len(dataset)

    LOG.info(f'Dataset length: {len(dataset) * dataset.repeats}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        LOG.info("Training...")
        train(config, generator, discriminator, kp_detector, log_dir, dataset)
    elif opt.mode == 'export':
        export(generator, discriminator, kp_detector, log_dir, dataset.get_input_fn(1))
    elif opt.mode == 'reconstruction':
        LOG.info("Reconstruction...")
        # reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        LOG.info("Animate...")
        # animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)


def train(config, generator, discriminator, kp_detector, log_dir, dataset):
    train_params = config['train_params']

    step_var = tf.Variable(initial_value=0, dtype=tf.int32)

    input_fn = dataset.get_input_fn(train_params['batch_size'])
    optimizer_generator = tf.keras.optimizers.Adam(
        learning_rate=train_params['lr_generator'], beta_1=0.5, beta_2=0.999
    )
    optimizer_discriminator = tf.keras.optimizers.Adam(
        learning_rate=train_params['lr_discriminator'], beta_1=0.5, beta_2=0.999
    )
    optimizer_kp_detector = tf.keras.optimizers.Adam(
        learning_rate=train_params['lr_kp_detector'], beta_1=0.5, beta_2=0.999
    )

    checkpoint = tf.train.Checkpoint(
        kp_detector=kp_detector,
        generator=generator,
        discriminator=discriminator,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        optimizer_kp_detector=optimizer_kp_detector,
        step_var=step_var
    )
    manager = tf.train.CheckpointManager(checkpoint, log_dir, max_to_keep=3)
    prev_step = 0
    if manager.restore_or_initialize():
        prev_step = manager.checkpoint.step_var.numpy()

    if prev_step != 0:
        LOG.info(f'Initialized at {prev_step} step.')

    start_epoch = 0

    tf.keras.optimizers.schedules.LearningRateSchedule()
    # scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
    #                                   last_epoch=start_epoch - 1)
    # scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
    #                                       last_epoch=start_epoch - 1)
    # scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
    #                                     last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    generator_full = model.GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = model.DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    # writer = tensorboardX.SummaryWriter(log_dir, flush_secs=60)
    summary_writer = tf.summary.create_file_writer(log_dir, flush_millis=60000)

    @tf.function
    def train_step(x_source, x_driving, epoch, step):
        with tf.GradientTape(persistent=True) as tape:
            generator_full.grad_tape = tape
            losses_generator, generated = generator_full((x_source, x_driving), training=True)

            loss_values = [tf.reduce_mean(val) for val in losses_generator.values()]
            loss = tf.reduce_sum(loss_values)

            if train_params['loss_weights']['generator_gan'] != 0:
                losses_discriminator = discriminator_full(
                    (x_driving, generated['kp_driving_value'], generated['prediction'])
                )
                loss_values = [tf.reduce_mean(val) for val in losses_discriminator.values()]
                disc_loss = tf.reduce_sum(loss_values)

            else:
                losses_discriminator = {}

            losses_generator.update(losses_discriminator)
            losses = {key: tf.reduce_mean(value) for key, value in losses_generator.items()}
            # logger.log_iter(losses=losses)

        generator_gradients = tape.gradient(loss, generator.trainable_variables)
        kp_detector_gradients = tape.gradient(loss, kp_detector.trainable_variables)
        optimizer_generator.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )
        optimizer_kp_detector.apply_gradients(
            zip(kp_detector_gradients, kp_detector.trainable_variables)
        )

        disc_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)
        optimizer_discriminator.apply_gradients(
            zip(disc_gradients, discriminator.trainable_variables)
        )

        return losses, generated

    for epoch in range(start_epoch, train_params['num_epochs']):
        for i, (x_source, x_driving) in input_fn.enumerate():
            step = prev_step + i.numpy() + int(epoch * len(dataset) * dataset.repeats / train_params['batch_size'])
            losses, generated = train_step(x_source, x_driving, tf.constant(epoch), tf.constant(step))

            if step % 100 == 0:
                LOG.info(
                    f'Epoch {epoch + 1}, global step {step}: {", ".join([f"{k}={v}" for k, v in losses.items()])}'
                )
                for k, loss in losses.items():
                    with summary_writer.as_default():
                        tf.summary.scalar(k, loss, step=step)
                source = x_source[0].numpy()
                driving = x_driving[0].numpy()
                kp_source = generated['kp_source_value'][0].numpy()
                kp_driving = generated['kp_driving_value'][0].numpy()
                pred = generated['prediction'][0].numpy()
                kp_source = kp_source * 127.5 + 127.5
                kp_driving = kp_driving * 127.5 + 127.5
                source = cv2.UMat((source * 255.).clip(0, 255).astype(np.uint8)).get()
                driving = cv2.UMat((driving * 255.).clip(0, 255).astype(np.uint8)).get()
                pred = (pred * 255.).clip(0, 255).astype(np.uint8)
                for x1, y1 in kp_source:
                    cv2.circle(source, (int(x1), int(y1)), 2, (250, 250, 250), thickness=cv2.FILLED)
                for x1, y1 in kp_driving:
                    cv2.circle(driving, (int(x1), int(y1)), 2, (250, 250, 250), thickness=cv2.FILLED)

                with summary_writer.as_default():
                    tf.summary.image(
                        'SourceDrivingPred', np.expand_dims(np.hstack((source, driving, pred)), axis=0),
                        step=step,
                    )
                    summary_writer.flush()

            step_var.assign_add(1)
            if step % 100 == 0:
                manager.save()


def export(generator, discriminator, kp_detector, log_dir, input_fn):
    x_source, x_driving = next(iter(input_fn))
    kp_source_value, kp_source_jacobian = kp_detector(x_source)
    kp_driving_value, kp_driving_jacobian = kp_detector(x_driving)
    generator(
        (x_source, kp_driving_value, kp_driving_jacobian, kp_source_value, kp_source_jacobian)
    )
    kp_detector.save(os.path.join(log_dir, 'kp_detector'))
    generator.save(os.path.join(log_dir, 'generator'))


if __name__ == '__main__':
    main()

