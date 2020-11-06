import glob
import logging
import os

import cv2
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

from augmentation import AllAugmentationTransform


LOG = logging.getLogger(__name__)


def read_video(name, frame_shape, is_train=True):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        vc = cv2.VideoCapture(name)
        n_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = np.sort(np.random.choice(n_frames, replace=True, size=2)) if is_train else range(n_frames)

        video = []
        for idx in frame_idx:
            vc.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = vc.read()
            if not ret:
                break

            video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        video = np.stack(video)
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(object):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, repeats=1):
        self.id_sampling = id_sampling
        self.frame_shape = tuple(frame_shape)
        self.repeats = repeats
        self.is_train = is_train

        data_dir = os.environ.get("DATA_DIR")
        if data_dir is not None:
            root_dir = data_dir

        LOG.info(f'Dataset root dir {root_dir}.')

        self.root_dir = os.path.join(root_dir)
        self.videos = os.listdir(root_dir)
        if all(os.path.isdir(os.path.join(root_dir, v)) for v in self.videos):
            LOG.info('Detected 2-level videos dataset.')
            if not self.id_sampling:
                self.videos = glob.glob(os.path.join(root_dir, '*/*'))

        if pairs_list:
            self.pairs_list = os.path.join(self.root_dir, pairs_list)
        else:
            pairs_list = None

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            LOG.info("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            LOG.info("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.1)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        name = self.videos[0]
        path = glob.glob(os.path.join(self.root_dir, name, '**/*.jpg'), recursive=True)
        if len(path) > 0:
            LOG.info(f'Detected frame dataset by {path[0]}.')
            self.video_dataset = False
        else:
            LOG.info(f'Detected video dataset.')
            self.video_dataset = True

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
            # self.transform = None
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling and self.video_dataset:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name, '**/*.mp4'), recursive=True))
        elif self.is_train and self.id_sampling and not self.video_dataset:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name, '*'), recursive=True))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = glob.glob(os.path.join(path, '*.jpg'))
            num_frames = len(frames)
            if num_frames < 2:
                new_idx = np.random.randint(0, len(self))
                return self[new_idx]
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [
                img_as_float32(cv2.cvtColor(cv2.imread(os.path.join(path, frames[idx])), cv2.COLOR_BGR2RGB))
                for idx in frame_idx
            ]
            new_video = np.zeros([len(video_array), *self.frame_shape])
            for i, im in enumerate(video_array):
                new_video[i] = cv2.resize(im, (self.frame_shape[1], self.frame_shape[0]))
            video_array = new_video
        else:
            video_array = read_video(path, frame_shape=self.frame_shape, is_train=self.is_train)
            if video_array.shape[1:3] != self.frame_shape[:2]:
                new_video = np.zeros([len(video_array), *self.frame_shape])
                for i, im in enumerate(video_array):
                    new_video[i] = cv2.resize(im, (self.frame_shape[1], self.frame_shape[0]))
                video_array = new_video

            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        # out = {}
        # if self.is_train:
        source = np.array(video_array[0], dtype='float32')
        driving = np.array(video_array[1], dtype='float32')

        # else:
        #     video = np.array(video_array, dtype='float32')
        #     out['video'] = video.transpose((3, 0, 1, 2))

        # out['name'] = video_name

        return source, driving

    def generator(self):
        for i in range(len(self)):
            yield self[i]

    def get_input_fn(self, batch_size):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            (tf.float32, tf.float32),
            (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3]))
        )
        return dataset.repeat(self.repeats).padded_batch(batch_size, drop_remainder=True).prefetch(batch_size * 2)
