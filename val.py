from modules import *
import os
import numpy as np
from utils.evaluator import Evaluator
import torch
from utils.img_read import *
import argparse
import logging
from kornia.metrics import AverageMeter
from tqdm import tqdm
import warnings
import yaml
from configs import from_dict
import dataset
from torch.utils.data import DataLoader
from thop import profile, clever_format
import time
import cv2
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
# =======================================

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(level='INFO', format=log_f)


def test(args):
    test_d = getattr(dataset, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')

    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0,
        collate_fn=test_dataset.__collate_fn__, pin_memory=True
    )

    fuse_out_folder = args.out_dir
    rgb_out_folder = os.path.join(fuse_out_folder, "RGB_results")
    os.makedirs(fuse_out_folder, exist_ok=True)
    os.makedirs(rgb_out_folder, exist_ok=True)
    logging.info(f"Grayscale image save path: {fuse_out_folder}")
    logging.info(f"RGB image save path: {rgb_out_folder}")

    fuse_net = Fuse()
    ckpt = torch.load(args.ckpt_path, map_location=device)
    fuse_net.load_state_dict(ckpt['fuse_net'])
    fuse_net.to(device)
    fuse_net.eval()

    time_list = []
    with torch.no_grad():
        logging.info(f'Start fusing the image ...')
        iter_bar = tqdm(testloader, total=len(testloader), ncols=80)

        for batch_data in iter_bar:
            data_ir, data_vi_y, data_vi_cbcr, _, img_name = batch_data

            data_ir = data_ir.to(device)
            data_vi_y = data_vi_y.to(device)
            data_vi_cbcr = data_vi_cbcr.to(device)

            ts = time.time()
            fus_data, _, _ = fuse_net(data_ir, data_vi_y)  # fus_data: [1,1,H,W]
            te = time.time()
            time_list.append(te - ts)

            try:
                fi_gray = fus_data.squeeze().cpu().numpy()
                fi_gray = (fi_gray * 255).astype(np.uint8).clip(0, 255)
                img_save(fi_gray, img_name[0], fuse_out_folder)
            except Exception as e:
                logging.error(f"Failed to save the grayscale image {img_name[0]}: {str(e)}")
                continue

            if args.mode == "RGB":
                try:
                    y_fuse = fus_data
                    cbcr = data_vi_cbcr

                    ycbcr = torch.cat([y_fuse, cbcr], dim=1)

                    ycbcr_np = ycbcr.squeeze().permute(1, 2, 0).cpu().numpy()

                    ycbcr_np = (ycbcr_np * 255).astype(np.uint8).clip(0, 255)

                    rgb_img = cv2.cvtColor(ycbcr_np, cv2.COLOR_YCrCb2RGB)

                    img_save(rgb_img, img_name[0], rgb_out_folder, mode='RGB')

                except Exception as e:
                    logging.error(f"Failed to save the RGB image {img_name[0]}: {str(e)}")
                    continue

    if len(time_list) > 1:
        avg_time = np.round(np.mean(time_list[1:]), 6)
    else:
        avg_time = 0
    logging.info(f'Image fusion completed!')
    logging.info(f'Average fusion time per image: {avg_time}s')

    evaluate(fuse_out_folder)


def evaluate(fuse_out_folder):
    test_d = getattr(dataset, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')
    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0,
        collate_fn=test_dataset.__collate_fn__, pin_memory=True
    )

    metric_result = [AverageMeter() for _ in range(7)]
    metric_names = ['EN', 'SD', 'SF', 'MI', 'VIFF', 'Qabf', 'AG']

    logging.info(f'Start evaluating the image ...')
    iter_bar = tqdm(testloader, total=len(testloader), ncols=80)

    for batch_data in iter_bar:
        try:
            data_ir, data_vi_y, _, _, img_name = batch_data
            data_vi = data_vi_y

            ir = data_ir.numpy().squeeze() * 255
            vi = data_vi.numpy().squeeze() * 255
            fi = img_read(os.path.join(fuse_out_folder, img_name[0]), 'L').numpy().squeeze() * 255

            h, w = fi.shape
            if h % 2 != 0 or w % 2 != 0:
                fi = fi[:h - (h % 2), :w - (w % 2)]
            if fi.shape != ir.shape or fi.shape != vi.shape:
                fi = cv2.resize(fi, (ir.shape[1], ir.shape[0]))

            metric_result[0].update(Evaluator.EN(fi))
            metric_result[1].update(Evaluator.SD(fi))
            metric_result[2].update(Evaluator.SF(fi))
            metric_result[3].update(Evaluator.MI(fi, ir, vi))
            metric_result[4].update(Evaluator.VIFF(fi, ir, vi))
            metric_result[5].update(Evaluator.Qabf(fi, ir, vi))
            metric_result[6].update(Evaluator.AG(fi))

        except Exception as e:
            logging.error(f"Image evaluation failed {img_name[0]}: {str(e)}")
            continue

    # 保存评估结果
    result_file = f'{fuse_out_folder}_result.txt'
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            for i, name in enumerate(metric_names):
                value = np.round(metric_result[i].avg, 3)
                f.write(f'{name}: {value}\n')
        logging.info(f"The evaluation results have been saved to: {result_file}")
    except Exception as e:
        logging.error(f"Failed to save evaluation results: {str(e)}")

    # 打印评估结果
    print("\n" * 2 + "=" * 80)
    print("Summary of Evaluation Results:")
    print("\t\t EN\t SD\t SF\t MI\tVIF\tQabf\tAG")
    print(f'result:\t'
          f'{np.ceil(metric_result[0].avg * 100) / 100:>6.2f}\t'
          f'{np.ceil(metric_result[1].avg * 100) / 100:>4.2f}\t'
          f'{np.ceil(metric_result[2].avg * 100) / 100:>4.2f}\t'
          f'{np.ceil(metric_result[3].avg * 100) / 100:>4.2f}\t'
          f'{np.ceil(metric_result[4].avg * 100) / 100:>4.2f}\t'
          f'{np.ceil(metric_result[5].avg * 100) / 100:>4.2f}\t'
          f'{np.ceil(metric_result[6].avg * 100) / 100:>4.2f}')
    print("=" * 80)


if __name__ == "__main__":
    try:
        config = yaml.safe_load(open('configs/cfg.yaml', encoding='utf-8'))
        cfg = from_dict(config)
    except Exception as e:
        logging.error(f"Failed to load the configuration file: {str(e)}")
        exit(1)

    parser = argparse.ArgumentParser(description='Infrared-Visible Image Fusion Test')
    parser.add_argument('--ckpt_path', type=str, default=f'models/{cfg.exp_name}.pth',
                        help='Model weight path')
    parser.add_argument('--dataset_name', type=str, default=cfg.dataset_name,
                        help='Dataset name (M3FD/MSRS/RoadScene)')
    parser.add_argument('--out_dir', type=str, default=f'test_result/{cfg.dataset_name}/{cfg.exp_name}',
                        help='output directory')
    parser.add_argument('--mode', type=str, default='RGB', choices=['gray', 'RGB'],
                        help='output mode：gray(Gray scale only) / RGB(gray+RGB)')
    args = parser.parse_args()

    logging.info("=" * 50)
    logging.info("Runtime Configuration:")
    logging.info(f"  dataset: {args.dataset_name}")
    logging.info(f"  output directory: {args.out_dir}")
    logging.info(f"  output mode: {args.mode}")
    logging.info(f"  model weights: {args.ckpt_path}")
    logging.info(f"  equipment: {device}")
    logging.info("=" * 50)

    test(args)