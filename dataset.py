import os
import random
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import functional as TF

#method to prepare the folders with train/val/test
def prepare_paths(train_dir, map_dir, fix_dir):
    #Prepare training data paths
    image_paths_train = []
    map_paths_train = []
    fix_paths_train = []

    map_dict = {}
    fix_dict = {}
    
    for f in sorted(os.listdir(map_dir / 'train')):
        stem = Path(f).stem
        map_dict[stem] = f

    for f in sorted(os.listdir(fix_dir / 'train')):
        stem = Path(f).stem
        fix_dict[stem] = f

    for f in sorted(os.listdir(train_dir / 'train')):
        stem = Path(f).stem
        if stem in map_dict and stem in fix_dict:
            image_paths_train.append(train_dir / 'train' / f)
            map_paths_train.append(map_dir / 'train' / map_dict[stem])
            fix_paths_train.append(fix_dir / 'train' / fix_dict[stem])

    print(f"Train size of images: {len(image_paths_train)}")
    print(f"Train size of maps: {len(map_paths_train)}")
    print(f"Train size of fixations: {len(fix_paths_train)}")

    #Prepare val/test; notice test is from part of the dataset val
    #because the original test does not have labels
    image_paths_val_full = []
    map_paths_val_full = []
    fix_paths_val_full = []

    map_dict = {}
    fix_dict = {}

    for f in sorted(os.listdir(map_dir / 'val')):
        stem = Path(f).stem
        map_dict[stem] = f

    for f in sorted(os.listdir(fix_dir / 'val')):
        stem = Path(f).stem
        fix_dict[stem] = f

    for f in sorted(os.listdir(train_dir / 'val')):
        stem = Path(f).stem
        if stem in map_dict and stem in fix_dict:
            image_paths_val_full.append(train_dir / 'val' / f)
            map_paths_val_full.append(map_dir / 'val' / map_dict[stem])
            fix_paths_val_full.append(fix_dir / 'val' / fix_dict[stem])

    idx = list(range(len(image_paths_val_full)))
    random.shuffle(idx)

    val_idx, test_idx = idx[:3500], idx[3500:]

    image_paths_val = [image_paths_val_full[i] for i in val_idx]
    image_paths_test = [image_paths_val_full[i] for i in test_idx]
    map_paths_val = [map_paths_val_full[i] for i in val_idx]
    map_paths_test = [map_paths_val_full[i] for i in test_idx]
    fix_paths_val = [fix_paths_val_full[i] for i in val_idx]
    fix_paths_test = [fix_paths_val_full[i] for i in test_idx]

    print(f"Val size: {len(image_paths_val)}")
    print(f"Test size: {len(image_paths_test)}")

    return {
        "train": (image_paths_train, map_paths_train, fix_paths_train),
        "val": (image_paths_val, map_paths_val, fix_paths_val),
        "test": (image_paths_test, map_paths_test, fix_paths_test),
    }

#method to load fixation maps
def load_fixation_map(mat_path, size):
    """SALICON fixations are stored as .mat files (gaze of ~60 observers,
    each with fixation coordinates [x, y]). Build the binary fixation map
    DIRECTLY at the target resolution by scaling the coordinates: resizing a
    sparse binary map afterwards would discard ~80% of the points."""
    m = sio.loadmat(mat_path)
    H, W = int(m['resolution'][0, 0]), int(m['resolution'][0, 1])
    th, tw = size
    fmap = np.zeros((th, tw), dtype=np.uint8)
    gaze = m['gaze']
    for i in range(gaze.shape[0]):
        pts = gaze[i, 0]['fixations']
        if pts is None or pts.size == 0:
            continue
        xs = np.clip((pts[:, 0].astype(np.float64) * tw / W).astype(int), 0, tw - 1)  # x -> column
        ys = np.clip((pts[:, 1].astype(np.float64) * th / H).astype(int), 0, th - 1)  # y -> row
        fmap[ys, xs] = 255
    return Image.fromarray(fmap, mode='L')


class ImageDataset(Dataset):

    def __init__(self, image_paths, map_paths, fix_paths, image_size=(256, 256), train=False):

        self.train = train

        if len(image_paths) != len(map_paths):
            raise ValueError("Number of images and maps must be the same")

        self.image_paths = image_paths
        self.map_paths = map_paths
        self.fix_paths = fix_paths
        self.image_size = image_size

        #resize the images and apply normalization to standard values
        self.image_transform = Compose([
            Resize(self.image_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        #same transformation for maps, but no normalization
        self.map_transform = Compose([
            Resize(self.image_size),
            ToTensor()
        ])

        # fix is already built at image_size (coordinates scaled), so only ToTensor
        self.fix_transform = ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        map_path = self.map_paths[index]
        fix_path = self.fix_paths[index]

        image = Image.open(image_path).convert("RGB")
        sal = Image.open(map_path).convert("L")
        fix = load_fixation_map(fix_path, self.image_size)  # binary map already at image_size

        #data augmentation, applied to both the image, saliency and fixation
        if self.train:

            #Flipping of the image
            r = random.random()
            if r < 0.5:
                image = TF.hflip(image)
                sal = TF.hflip(sal)
                fix = TF.hflip(fix)

            #Random rotation. Interpolation is used, for fixations nearest is needed, otherwise most of the values are lost
            r = random.random()
            if r < 0.3:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                sal = TF.rotate(sal, angle, interpolation=TF.InterpolationMode.BILINEAR)
                fix = TF.rotate(fix, angle, interpolation=TF.InterpolationMode.NEAREST)

            #Random change in brightness, contrast and saturation
            r = random.random()
            if r < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        return(self.image_transform(image), self.map_transform(sal), self.fix_transform(fix))
