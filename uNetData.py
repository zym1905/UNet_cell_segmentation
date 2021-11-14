import logging
import torch
import sys
import numpy as np

from os import listdir
from os.path import splitext
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from skimage.io import imread

class BasicDataset(Dataset):
    def __init__(self, root_dir: str, target_dir: str, scale: float = 1.0):
        self.dataFromat(root_dir, target_dir)
        self.images_dir = Path(target_dir + '/images')
        self.masks_idr = Path(target_dir + '/masks')
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.ids = [listdir(self.images_dir)]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def dataFromat(cls, root_dir, target_dir):
        dirs = listdir(root_dir)
        print('original image number:' + str(len(dirs)))
        count = 0
        for dir_name in dirs:
            if dir_name.startswith('.'):
                continue
            image_dir = root_dir + '/' + dir_name + '/images'
            images = listdir(image_dir)
            mask_dir = root_dir + '/' + dir_name + '/masks'
            masks = listdir(mask_dir)

            if len(images) > 1:
                sys.exit(-1)
            image_ndarrary = imread(image_dir + '/' + images[0], as_gray=True)
            if image_ndarrary.shape != (256, 256):
                continue

            image_name = images[0][:-4]
            # print(image_name)
            mask_ndarrary = np.ndarray(shape=image_ndarrary.shape)
            for mask in masks:
                mask_ndarrary += imread(mask_dir + '/' + mask, as_gray=True)
            count += 1
            np.savez(target_dir + '/images/image_' + image_name + '.npz', image_ndarrary, 'image_' + image_name)
            np.savez(target_dir + '/masks/mask_' + image_name + '.npz', mask_ndarrary, 'mask_' + image_name)

        print('final image number:' + str(count))

    @classmethod
    def preprocess(cls, cell_img, scale, is_mask):
        w, h = cell_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        cell_img = cell_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(cell_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, root_dir, target_dir, scale=1):
        super().__init__(root_dir, target_dir, scale)