from torch.utils.data import Dataset
import torch
from configs import *
import logging
from torchvision.transforms import Compose, Resize
from pathlib import Path
from typing import Literal
from utils.img_read import img_read
import os
from utils.saliency import Saliency
import cv2
import numpy as np

def read_image_ycbcr(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    y = ycbcr[..., 0:1]
    cbcr = ycbcr[..., 1:3]

    y = torch.from_numpy(y).permute(2, 0, 1).float()
    cbcr = torch.from_numpy(cbcr).permute(2, 0, 1).float()

    return y, cbcr


def read_image_gray(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).float()
    return img


def check_mask(root: Path, img_list, config: ConfigDict):
    mask_cache = True
    if (root / 'mask').exists():
        for img_name in img_list:
            if not (root / 'mask' / img_name).exists():
                mask_cache = False
                break
    else:
        mask_cache = False
    if mask_cache:
        logging.info('find mask cache in folder, skip saliency detection')
    else:
        logging.info('find no mask cache in folder, start saliency detection')
        saliency = Saliency()
        saliency.inference(src=root / 'ir', dst=root / 'mask', suffix='png')

class MSRS(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = os.listdir(os.path.join(cfg.dataset_root, self.mode, 'ir'))
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        self.ir_path = Path(Path(self.cfg.dataset_root) / self.mode / 'ir')
        self.vi_path = Path(Path(self.cfg.dataset_root) / self.mode / 'vi')
        if self.mode == 'train' and cfg.have_seg_label == False:
            check_mask(Path(cfg.dataset_root) / 'train', self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'labels')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]

        if self.mode == 'train':

            ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
            vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')


            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)


            return ir_img, vi_img, mask, img_name

        else:
            ir_img = read_image_gray(os.path.join(self.ir_path, img_name))
            vi_img_y, vi_img_cbcr = read_image_ycbcr(os.path.join(self.vi_path, img_name))
            mask = None

            _, h, w = ir_img.shape
            if h % 2 != 0 or w % 2 != 0:
                ir_img = ir_img[:, :h - (h % 2), :w - (w % 2)]
                vi_img_y = vi_img_y[:, :h - (h % 2), :w - (w % 2)]
                vi_img_cbcr = vi_img_cbcr[:, :h - (h % 2), :w - (w % 2)]

            return ir_img, vi_img_y, vi_img_cbcr, mask, img_name

    def __collate_fn__(self, batch):

        if self.mode == 'train':
            ir_img_batch, vi_img_batch, mask_batch, img_name_batch = zip(*batch)
            ir_img_batch = torch.stack(ir_img_batch, dim=0)
            vi_img_batch = torch.stack(vi_img_batch, dim=0)
            mask_batch = torch.stack(mask_batch, dim=0)
            return ir_img_batch, vi_img_batch, mask_batch, img_name_batch


        else:
            ir_img_batch, vi_img_y_batch, vi_img_cbcr_batch, mask_batch, img_name_batch = zip(*batch)
            ir_img_batch = torch.stack(ir_img_batch, dim=0)
            vi_img_y_batch = torch.stack(vi_img_y_batch, dim=0)
            vi_img_cbcr_batch = torch.stack(vi_img_cbcr_batch, dim=0)
            return ir_img_batch, vi_img_y_batch, vi_img_cbcr_batch, mask_batch, img_name_batch


class RoadScene(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = Path(Path(self.cfg.dataset_root) / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        if self.mode == 'train' and cfg.have_seg_label == False:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            check_mask(Path(cfg.dataset_root), self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'labels')
        if self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'vi')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]

        if self.mode == 'train':

            ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
            vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')

            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)

            return ir_img, vi_img, mask, img_name

        else:

            ir_img = read_image_gray(os.path.join(self.ir_path, img_name))
            vi_img_y, vi_img_cbcr = read_image_ycbcr(os.path.join(self.vi_path, img_name))
            mask = None

            _, h, w = ir_img.shape
            if h % 2 != 0 or w % 2 != 0:
                ir_img = ir_img[:, :h - (h % 2), :w - (w % 2)]
                vi_img_y = vi_img_y[:, :h - (h % 2), :w - (w % 2)]
                vi_img_cbcr = vi_img_cbcr[:, :h - (h % 2), :w - (w % 2)]

            return ir_img, vi_img_y, vi_img_cbcr, mask, img_name

    def __collate_fn__(self, batch):

        if self.mode == 'train':
            ir_img_batch, vi_img_batch, mask_batch, img_name_batch = zip(*batch)
            ir_img_batch = torch.stack(ir_img_batch, dim=0)
            vi_img_batch = torch.stack(vi_img_batch, dim=0)
            mask_batch = torch.stack(mask_batch, dim=0)
            return ir_img_batch, vi_img_batch, mask_batch, img_name_batch

        else:
            ir_img_batch, vi_img_y_batch, vi_img_cbcr_batch, mask_batch, img_name_batch = zip(*batch)
            ir_img_batch = torch.stack(ir_img_batch, dim=0)
            vi_img_y_batch = torch.stack(vi_img_y_batch, dim=0)
            vi_img_cbcr_batch = torch.stack(vi_img_cbcr_batch, dim=0)
            return ir_img_batch, vi_img_y_batch, vi_img_cbcr_batch, mask_batch, img_name_batch

class TNO(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        self.img_list = Path(Path(self.cfg.dataset_root) / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(self.img_list)} images from TNO dataset')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        if self.mode == 'train' and cfg.have_seg_label == False:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            check_mask(Path(cfg.dataset_root), self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'labels')
        if self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'vi')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]

        if self.mode == 'train':

            ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
            vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')

            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)

            return ir_img, vi_img, mask, img_name

        else:

            ir_img = read_image_gray(os.path.join(self.ir_path, img_name))
            vi_img_y, vi_img_cbcr = read_image_ycbcr(os.path.join(self.vi_path, img_name))
            mask = None

            _, h, w = ir_img.shape
            if h % 2 != 0 or w % 2 != 0:
                ir_img = ir_img[:, :h - (h % 2), :w - (w % 2)]
                vi_img_y = vi_img_y[:, :h - (h % 2), :w - (w % 2)]
                vi_img_cbcr = vi_img_cbcr[:, :h - (h % 2), :w - (w % 2)]

            return ir_img, vi_img_y, vi_img_cbcr, mask, img_name

    def __collate_fn__(self, batch):

        if self.mode == 'train':
            ir_img_batch, vi_img_batch, mask_batch, img_name_batch = zip(*batch)
            ir_img_batch = torch.stack(ir_img_batch, dim=0)
            vi_img_batch = torch.stack(vi_img_batch, dim=0)
            mask_batch = torch.stack(mask_batch, dim=0)
            return ir_img_batch, vi_img_batch, mask_batch, img_name_batch

        else:
            ir_img_batch, vi_img_y_batch, vi_img_cbcr_batch, mask_batch, img_name_batch = zip(*batch)
            ir_img_batch = torch.stack(ir_img_batch, dim=0)
            vi_img_y_batch = torch.stack(vi_img_y_batch, dim=0)
            vi_img_cbcr_batch = torch.stack(vi_img_cbcr_batch, dim=0)
            return ir_img_batch, vi_img_y_batch, vi_img_cbcr_batch, mask_batch, img_name_batch

class M3FD(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        self.img_list = Path(Path(self.cfg.dataset_root) / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(self.img_list)} images from M3FD dataset')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        if self.mode == 'train' and cfg.have_seg_label == False:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            check_mask(Path(cfg.dataset_root), self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'labels')
        if self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'vi')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]

        if self.mode == 'train':

            ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
            vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')

            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)

            return ir_img, vi_img, mask, img_name

        else:

            ir_img = read_image_gray(os.path.join(self.ir_path, img_name))
            vi_img_y, vi_img_cbcr = read_image_ycbcr(os.path.join(self.vi_path, img_name))
            mask = None

            _, h, w = ir_img.shape
            if h % 2 != 0 or w % 2 != 0:
                ir_img = ir_img[:, :h - (h % 2), :w - (w % 2)]
                vi_img_y = vi_img_y[:, :h - (h % 2), :w - (w % 2)]
                vi_img_cbcr = vi_img_cbcr[:, :h - (h % 2), :w - (w % 2)]

            return ir_img, vi_img_y, vi_img_cbcr, mask, img_name

    def __collate_fn__(self, batch):

        if self.mode == 'train':
            ir_img_batch, vi_img_batch, mask_batch, img_name_batch = zip(*batch)
            ir_img_batch = torch.stack(ir_img_batch, dim=0)
            vi_img_batch = torch.stack(vi_img_batch, dim=0)
            mask_batch = torch.stack(mask_batch, dim=0)
            return ir_img_batch, vi_img_batch, mask_batch, img_name_batch

        else:
            ir_img_batch, vi_img_y_batch, vi_img_cbcr_batch, mask_batch, img_name_batch = zip(*batch)
            ir_img_batch = torch.stack(ir_img_batch, dim=0)
            vi_img_y_batch = torch.stack(vi_img_y_batch, dim=0)
            vi_img_cbcr_batch = torch.stack(vi_img_cbcr_batch, dim=0)
            return ir_img_batch, vi_img_y_batch, vi_img_cbcr_batch, mask_batch, img_name_batch
class MRI_PET(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        self.img_list = Path(Path(self.cfg.dataset_root) / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(self.img_list)} images from MRI-PET dataset')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        if self.mode == 'train' and cfg.have_seg_label == False:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'MRI')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'PET')
            check_mask(Path(cfg.dataset_root), self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'MRI')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'PET')
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'labels')
        if self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'MRI')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'PET')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]

        if self.mode == 'train':

            ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
            vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')

            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)

            return ir_img, vi_img, mask, img_name

        else:

            ir_img = read_image_gray(os.path.join(self.ir_path, img_name))
            vi_img_y, vi_img_cbcr = read_image_ycbcr(os.path.join(self.vi_path, img_name))
            mask = None

            _, h, w = ir_img.shape
            if h % 2 != 0 or w % 2 != 0:
                ir_img = ir_img[:, :h - (h % 2), :w - (w % 2)]
                vi_img_y = vi_img_y[:, :h - (h % 2), :w - (w % 2)]
                vi_img_cbcr = vi_img_cbcr[:, :h - (h % 2), :w - (w % 2)]

            return ir_img, vi_img_y, vi_img_cbcr, mask, img_name

    def __collate_fn__(self, batch):

        if self.mode == 'train':
            ir_img_batch, vi_img_batch, mask_batch, img_name_batch = zip(*batch)
            ir_img_batch = torch.stack(ir_img_batch, dim=0)
            vi_img_batch = torch.stack(vi_img_batch, dim=0)
            mask_batch = torch.stack(mask_batch, dim=0)
            return ir_img_batch, vi_img_batch, mask_batch, img_name_batch

        else:
            ir_img_batch, vi_img_y_batch, vi_img_cbcr_batch, mask_batch, img_name_batch = zip(*batch)
            ir_img_batch = torch.stack(ir_img_batch, dim=0)
            vi_img_y_batch = torch.stack(vi_img_y_batch, dim=0)
            vi_img_cbcr_batch = torch.stack(vi_img_cbcr_batch, dim=0)
            return ir_img_batch, vi_img_y_batch, vi_img_cbcr_batch, mask_batch, img_name_batch


if __name__ == '__main__':
    import yaml

    config = yaml.safe_load(open('./configs/cfg.yaml'))
    cfg = from_dict(config)
    train_dataset = MSRS(cfg, 'train')

    for i in range(3):
        ir, vi, mask, img_name = train_dataset[i]
        print(f"【training mode】Image: {img_name}")
        print(f"IR shape: {ir.shape}, VI shape: {vi.shape}, Mask shape: {mask.shape}")

    test_dataset = MSRS(cfg, 'test')
    for i in range(3):
        ir, vi_y, vi_cbcr, mask, img_name = test_dataset[i]
        print(f"\n【Verification mode】Image: {img_name}")
        print(f"IR shape: {ir.shape}, VI Y shape: {vi_y.shape}, VI CbCr shape: {vi_cbcr.shape}")

        import matplotlib.pyplot as plt

        ir_np = ir.squeeze().numpy()
        vi_y_np = vi_y.squeeze().numpy()

        vi_cbcr_np = vi_cbcr.permute(1, 2, 0).numpy()
        ycbcr = np.concatenate([vi_y_np[..., np.newaxis], vi_cbcr_np], axis=-1)
        ycbcr = (ycbcr * 255).astype(np.uint8)
        vi_rgb = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)

        plt.subplot(131)
        plt.imshow(ir_np, cmap='gray')
        plt.title('IR')
        plt.subplot(132)
        plt.imshow(vi_y_np, cmap='gray')
        plt.title('VI Y')
        plt.subplot(133)
        plt.imshow(vi_rgb)
        plt.title('VI RGB')
        plt.savefig(f'./test_{img_name}.png')
        plt.close()