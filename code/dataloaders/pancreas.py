"""PancreasCT dataset (preprocessed .h5 volumes), training transforms and the two-stream batch sampler."""
import itertools

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class Pancreas(Dataset):
    """Volumes listed in train.list (split='train') or test.list (split='test' or 'val')."""

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test' or split == 'val':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/Pancreas_data/{}".format(image_name), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomCrop(object):
    """
    Random crop of `output_size`. With probability 0.5 the crop is centred on a
    random pancreas voxel, so that small patches still contain the organ
    (it occupies a tiny fraction of an abdominal CT volume).
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Pad volumes that are not larger than the patch.
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)

            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        if np.random.uniform() < 0.5:
            # Foreground-aware crop: centre the patch on a random pancreas voxel.
            pancreas_voxels = np.where(label == 1)

            if len(pancreas_voxels[0]) > 0:
                rand_idx = np.random.randint(0, len(pancreas_voxels[0]))
                center_w = pancreas_voxels[0][rand_idx]
                center_h = pancreas_voxels[1][rand_idx]
                center_d = pancreas_voxels[2][rand_idx]

                # Clamp the start so that the patch stays inside the volume.
                w1 = max(0, min(w - self.output_size[0], center_w - self.output_size[0] // 2))
                h1 = max(0, min(h - self.output_size[1], center_h - self.output_size[1] // 2))
                d1 = max(0, min(d - self.output_size[2], center_d - self.output_size[2] // 2))
            else:
                # No pancreas in the volume: fall back to a random crop.
                w1 = np.random.randint(0, max(1, w - self.output_size[0] + 1))
                h1 = np.random.randint(0, max(1, h - self.output_size[1] + 1))
                d1 = np.random.randint(0, max(1, d - self.output_size[2] + 1))
        else:
            # Random crop anywhere in the volume (max(1, ...) avoids an empty range when w == output_size).
            w1 = np.random.randint(0, max(1, w - self.output_size[0] + 1))
            h1 = np.random.randint(0, max(1, h - self.output_size[1] + 1))
            d1 = np.random.randint(0, max(1, d - self.output_size[2] + 1))

        label = label[w1:w1 + self.output_size[0],
                      h1:h1 + self.output_size[1],
                      d1:d1 + self.output_size[2]]

        image = image[w1:w1 + self.output_size[0],
                      h1:h1 + self.output_size[1],
                      d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """Random rotation by a multiple of 90 degrees in the first two axes, then a random flip."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class ToTensor(object):
    """Add the channel axis to the image and convert both arrays to tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """
    Batches made of `batch_size - secondary_batch_size` labeled indices and
    `secondary_batch_size` unlabeled ones. One epoch is one pass over the
    labeled indices; the unlabeled ones are cycled through indefinitely.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)
