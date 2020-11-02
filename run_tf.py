import argparse
import logging
import os
from shutil import copy
import time
import yaml

import cv2
import numpy as np
import tensorflow as tf

from modules_tf.frames_dataset import FramesDataset

from modules_tf.generator import OcclusionAwareGenerator
from modules_tf.discriminator import MultiScaleDiscriminator
from modules_tf.keypoint_detector import KPDetector

from reconstruction import reconstruction
from animate import animate


LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--data-dir")
    parser.add_argument("--train-dir", default='train')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument("--enable-jacobian", action='store_true')
    parser.add_argument("--disable-jacobian", action='store_true')
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=0)
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

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if opt.verbose:
        LOG.info(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if opt.verbose:
        LOG.info(discriminator)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if opt.verbose:
        LOG.info(kp_detector)

    dataset = FramesDataset(
        is_train=(opt.mode == 'train'),
        **config['dataset_params']
    )
    # dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    LOG.info(f'Dataset length: {len(dataset)}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        LOG.info("Training...")
        train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == 'reconstruction':
        LOG.info("Reconstruction...")
        reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        LOG.info("Animate...")
        animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)


def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    input_fn = dataset.get_input_fn(train_params['batch_size'])
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'],
                                               betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'],
                                             betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(
        dataset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    LOG.info(f'Full dataset length (with repeats): {len(dataset)}')

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    writer = tensorboardX.SummaryWriter(log_dir, flush_secs=60)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs'], disable=None):
            for i, x in enumerate(dataloader):
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

                step = i + int(epoch * len(dataset) / dataloader.batch_size)
                if step % 20 == 0:
                    LOG.info(
                        f'Epoch {epoch + 1}, global step {step}: {", ".join([f"{k}={v}" for k, v in losses.items()])}'
                    )

                if step != 0 and step % 50 == 0:
                    for k, loss in losses.items():
                        writer.add_scalar(k, float(loss), global_step=step)
                    # add images
                    source = x['source'][0].detach().cpu().numpy().transpose([1, 2, 0])
                    driving = x['driving'][0].detach().cpu().numpy().transpose([1, 2, 0])
                    kp_source = generated['kp_source']['value'][0].detach().cpu().numpy()
                    kp_driving = generated['kp_driving']['value'][0].detach().cpu().numpy()
                    pred = generated['prediction'][0].detach().cpu().numpy().transpose([1, 2, 0])
                    kp_source = kp_source * 127.5 + 127.5
                    kp_driving = kp_driving * 127.5 + 127.5
                    source = cv2.UMat((source * 255.).clip(0, 255).astype(np.uint8)).get()
                    driving = cv2.UMat((driving * 255.).clip(0, 255).astype(np.uint8)).get()
                    pred = (pred * 255.).clip(0, 255).astype(np.uint8)
                    for x1, y1 in kp_source:
                        cv2.circle(source, (int(x1), int(y1)), 2, (250, 250, 250), thickness=cv2.FILLED)
                    for x1, y1 in kp_driving:
                        cv2.circle(driving, (int(x1), int(y1)), 2, (250, 250, 250), thickness=cv2.FILLED)

                    writer.add_image(
                        'SourceDrivingPred', np.hstack((source, driving, pred)),
                        global_step=step,
                        dataformats='HWC'
                    )
                    writer.flush()

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()

            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector})


if __name__ == '__main__':
    main()

