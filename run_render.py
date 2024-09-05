import argparse
import random

import numpy as np
import torch, cv2, os

from src import config
from src.NICE_SLAM import NICE_SLAM
from src.tools.eval_recon import render_image
from localfeature.LocalFeature import CVisualLocLocal
from localfeature.lcore.hal import eSettingCmd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def __BGR2GRAY(img):
    oGrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.expand_dims(np.asarray(oGrayImg), axis=0)

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
    parser.add_argument('--idx', type=int, help='index of the image to render')
    parser.add_argument('--render_style', type=str, help='render style: mesh | nerf', default='mesh')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')

    slam = NICE_SLAM(cfg, args)
    if(args.idx == -1):
        from_idx = 0
        to_idx = 2000
    else:
        from_idx = args.idx
        to_idx = args.idx + 1
    strInputFolder =cfg['data']['input_folder']
    strOutputFolder = cfg['data']['output'] + "/render/" + str(args.render_style)
    if(not os.path.exists(strOutputFolder)):
        os.makedirs(strOutputFolder)
    for idx in range(from_idx, to_idx):
        oSourceImg = cv2.imread(strInputFolder + "/results/frame" + str(idx).zfill(6) + ".jpg")
        oSourceDepth = cv2.imread(strInputFolder + "/results/depth" + str(idx).zfill(6) + ".png", cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['cam']['png_depth_scale']
        strDataName = cfg['data']['input_folder'].split("/")[-1]
        if args.render_style == 'mesh':
            c2w = slam.get_gt_c2w_list(idx).detach().cpu().numpy()
            strFinalMeshEvalRecPath = cfg['data']['output'] + "/mesh/final_mesh_eval_rec.ply"
            strGtMeshFile = "cull_replica_mesh/" + strDataName + ".ply"
            color_np, depth = render_image(strFinalMeshEvalRecPath, strGtMeshFile, c2w, True)
        else:
            depth, uncertainty, rgb = slam.render(idx, oSourceDepth)
            color = rgb.detach().cpu().numpy()
            depth = depth.detach().cpu().numpy()
            color_np = np.clip(color, 0, 1) * 255

        oLocalFeature = CVisualLocLocal(str("eventpointnet"))
        oLocalFeature.Open("match", True)
        oMatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        oLocalFeature.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY, __BGR2GRAY(color_np))
        vTargetKpt, vTargetDesc, _ = oLocalFeature.Read()
        oLocalFeature.Reset()

        oLocalFeature.Setting(eSettingCmd.eSettingCmd_IMAGE_DATA_GRAY, __BGR2GRAY(oSourceImg))
        vKpSetSource, vSourceDesc, _ = oLocalFeature.Read()
        oLocalFeature.Reset()

        cv2.imwrite(strOutputFolder + "/rgb_" + str(idx).zfill(6) + ".png", cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
        max_depth = np.max(depth)
        cv2.imwrite(strOutputFolder + "/depth_" + str(idx).zfill(6) + ".png", depth / max_depth * 255)
        if(len(vTargetDesc) > 0 and len(vSourceDesc) > 0):
            vMatches = oMatcher.match(vTargetDesc, vSourceDesc)

            oMatchesMask = []
            vKpSetQuery = np.float32([vKpSetSource[m.trainIdx].pt for m in vMatches]).reshape(-1, 1, 2)
            vKpSetRender = np.float32([vTargetKpt[m.queryIdx].pt for m in vMatches]).reshape(-1, 1, 2)
            if(len(vKpSetRender) > 5):
                _, oMatchesMask = cv2.findHomography(vKpSetQuery, vKpSetRender, cv2.RANSAC, 3.0)

            vMatchesMask = []
            for _, mask in enumerate(oMatchesMask):
                if(mask[0] == 1):
                    vMatchesMask.append(1)
                else:
                    vMatchesMask.append(0)
        else:
            vMatches = None
        oImgMatch = cv2.drawMatches(cv2.convertScaleAbs(color_np), 
                                            vTargetKpt, 
                                            cv2.convertScaleAbs(oSourceImg), 
                                            vKpSetSource,
                                            vMatches, 
                                            None, 
                                            matchColor=(0, 255, 0), 
                                            singlePointColor=(0, 0, 255), 
                                            matchesMask=vMatchesMask, flags=0)
        cv2.imwrite(strOutputFolder + "/match_" + str(idx).zfill(6) + ".png", oImgMatch)


if __name__ == '__main__':
    main()
