import argparse
import glob
import multiprocessing as mp
import os
import time

import cv2
import tqdm
from Dictguided.adet.config import get_cfg
from Dictguided.detectron2.detectron2.data.detection_utils import read_image
from Dictguided.detectron2.detectron2.utils.logger import setup_logger
from Dictguided.demo.predictor import VisualizationDemo


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg



def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="./configs/BAText/VinText/attn_R_50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for path in tqdm.tqdm(args.input):
        img = read_image(path, format="BGR")
        start_time = time.time()
        file_name = path.split('/')[-1]
        file_name = file_name.split('.')[0]

        path_out = os.path.join(args.output, f'res_{file_name}.txt')
        predictions = demo.run_on_image(img, path, path_out, args.confidence_threshold)

