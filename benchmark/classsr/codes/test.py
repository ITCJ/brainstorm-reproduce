import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import numpy as np

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)
util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

opt_net = opt["network_G"]
which_model = opt_net["which_model_G"]

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []

    for data in test_loader:
        need_GT = True
        model.feed_data(data, need_GT=need_GT)
        img_path = data["GT_path"][0] if need_GT else data["LQ_path"][0]
        img_name = osp.splitext(osp.basename(img_path))[0]
        model.test()
        visuals = model.get_current_visuals(need_GT=need_GT)

        if which_model == "RCAN":
            sr_img = util.tensor2img(
                visuals["rlt"], out_type=np.uint8, min_max=(0, 255)
            )  # uint8
        else:
            sr_img = util.tensor2img(visuals["rlt"])  # uint8

        # save images
        suffix = opt["suffix"]
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = osp.join(dataset_dir, img_name + ".png")
        util.save_img(sr_img, save_img_path)

        if need_GT:
            if which_model == "RCAN":
                gt_img = util.tensor2img(
                    visuals["GT"], out_type=np.uint8, min_max=(0, 255)
                )  # uint8
            else:
                gt_img = util.tensor2img(visuals["GT"])

            sr_img, gt_img = util.crop_border([sr_img, gt_img], opt["scale"])
            psnr = util.calculate_psnr(sr_img, gt_img)
            ssim = util.calculate_ssim(sr_img, gt_img)
            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)

            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img / 255.0, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img / 255.0, only_y=True)

                psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
                ssim_y = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
                test_results["psnr_y"].append(psnr_y)
                test_results["ssim_y"].append(ssim_y)
                logger.info(
                    "{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                        img_name, psnr, ssim, psnr_y, ssim_y
                    )
                )
            else:
                logger.info("{:20s} - PSNR: {:.6f} dB;".format(img_name, psnr))
        else:
            logger.info(img_name)

    if need_GT:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
        ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
        logger.info(
            "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB\n".format(
                test_set_name, ave_psnr
            )
        )
        if test_results["psnr_y"] and test_results["ssim_y"]:
            ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
            ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
            logger.info(
                "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                    ave_psnr_y, ave_ssim_y
                )
            )
