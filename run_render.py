import argparse
import random

import numpy as np
import torch, cv2

from src import config
from src.NICE_SLAM import NICE_SLAM
from src.tools.eval_recon import render_image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')

    slam = NICE_SLAM(cfg, args)
    idx = 1530
    depth, uncertainty, rgb = slam.render(idx)
    color = rgb.detach().cpu().numpy()
    color_np = np.clip(color, 0, 1)
    cv2.imwrite("rgb.png", cv2.cvtColor(color_np * 255, cv2.COLOR_BGR2RGB))
    depth = depth.detach().cpu().numpy()
    max_depth = np.max(depth)
    cv2.imwrite("depth.png", depth / max_depth * 255)
    # slam.run()

    # c2w = slam.gt_c2w_list[idx].detach().cpu().numpy()
    # render_image("./output/Replica/office2_imap/mesh/final_mesh_eval_rec.ply", "/root/Dataset/Replica/office2_mesh.ply", c2w, False)

if __name__ == '__main__':
    main()
