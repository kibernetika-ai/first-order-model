import os

import cv2
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob


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


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        data_dir = os.environ.get("DATA_DIR")
        if data_dir is not None:
            root_dir = data_dir

        print(f'Dataset root dir {root_dir}.')

        self.root_dir = os.path.join(root_dir)
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        if pairs_list:
            self.pairs_list = os.path.join(self.root_dir, pairs_list)
        else:
            pairs_list = None
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        name = self.videos[0]
        path = glob.glob(os.path.join(self.root_dir, name, '**/*.jpg'), recursive=True)
        if len(path) > 0:
            print(f'Detected frame dataset by {path[0]}.')
            self.video_dataset = False
        else:
            print(f'Detected video dataset.')
            self.video_dataset = True

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
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

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
