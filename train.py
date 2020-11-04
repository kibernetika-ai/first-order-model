import sys

import cv2
import numpy as np
import tensorboardX
import torch
from tqdm import trange
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel
from sync_batchnorm import DataParallelWithCallback
import logging

def print_fun(s):
    print(s)
    sys.stdout.flush()


def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

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

    dataloader = DataLoader(
        dataset,
        batch_size=train_params['batch_size'],
        drop_last=True
    )

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    writer = tensorboardX.SummaryWriter(log_dir, flush_secs=60)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs'], disable=None):
            i = 0
            for x in dataloader:
                logging.info(x['source'].shape)
                logging.info(x['driving'].shape)
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

                step = i
                if step % 20 == 0:
                    logging.info(f'Epoch {epoch + 1}, step {step}: {", ".join([f"{k}={v}" for k, v in losses.items()])}')

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

                    i += 1

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector})
