import math
import torch
from tqdm import tqdm
import utils
import argparse
import os
import yaml
import datasets.loader as dloader
import models.loader as mloader
from torch.utils.data import DataLoader
import cv2


def eval_model(loader, model, model_name, out_path=None, mtsutil=None):
    if out_path is not None:
        utils.ensure_path(out_path, remove=True)
    print(f"out path {out_path}")

    model.eval()

    pbar = tqdm(loader, leave=False, desc="val")
    index = 0
    gt_exr_paths = []
    pred_exr_paths = []

    if model_name == "ShadeNet":
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda()

            brdf_color_batch = batch["brdf_color"].squeeze(0)
            dmdl_frame_batch = batch["dmdl_frame"].squeeze(0)
            rmv_batch = batch["rmv"].squeeze(0)
            gt_batch = batch["gt"].squeeze(0)
            base_color_batch = batch["base_color"].squeeze(0)
            batch_size = batch["batch_size"]
            frame_offset = batch["frame_offset"]

            shape = gt_batch[0:1].shape
            h, w = shape[2], shape[3]

            g_state, d_state = model.beginState(gt_batch.device, w, h)
            for idx in range(batch_size):
                idx += frame_offset
                brdf_color = brdf_color_batch[idx : idx + 1]
                base_color = base_color_batch[idx : idx + 1]
                dmdl_frame0 = dmdl_frame_batch[idx : idx + 1]
                dmdl_frame1 = dmdl_frame_batch[idx - 1 : idx]
                dmdl_frame2 = dmdl_frame_batch[idx - 2 : idx - 1]
                rmv0 = rmv_batch[idx + 1 : idx + 2]
                rmv1 = rmv_batch[idx + 0 : idx + 1]
                rmv2 = rmv_batch[idx - 1 : idx + 0]
                gt = gt_batch[idx + 1]

                L, g_s, d_s, smv_list, smv_resi_list = model(
                    brdf_color,
                    dmdl_frame0,
                    dmdl_frame1,
                    dmdl_frame2,
                    rmv0,
                    rmv1,
                    rmv2,
                    g_state,
                    d_state,
                )

                g_state = [g_s, g_state[1], g_state[2]]
                d_state = [d_s, d_state[1], d_state[2]]
                I = L * base_color
                gt_L = dmdl_frame_batch[idx + 1 : idx + 2]

                if out_path is not None:
                    p = os.path.join(save_path, "pred_" + str(index) + ".exr")
                    utils.write_exr(I[0].clone().detach().cpu(), p)
                    pred_exr_paths.append(p)
                    p = os.path.join(save_path, "gt_" + str(index) + ".exr")
                    utils.write_exr(gt[0].clone().detach().cpu(), p)
                    gt_exr_paths.append(p)
                    index += 1

    def exr_to_png(pin, pout):
        os.system("{} -q tonemap -o {} {}".format(mtsutil, pout, pin))

    if mtsutil is not None:
        print(f"using {mtsutil} for png tone mapping")
        gt_png_paths = []
        infer_png_paths = []
        print("converting gt")
        for gt_exr in tqdm(gt_exr_paths):
            gt_png = gt_exr[:-3:] + "png"
            exr_to_png(gt_exr, gt_png)
            gt_png_paths.append(gt_png)
        print("converting pred")
        for pred_exr in tqdm(pred_exr_paths):
            infer_png = pred_exr[:-3:] + "png"
            exr_to_png(pred_exr, infer_png)
            infer_png_paths.append(infer_png)
        ssim_res = utils.Averager()
        psnr_res = utils.Averager()
        with open(f"./metric/{model_name}.metric.log", "w") as log:
            print("evaluating")
            for i in tqdm(range(len(gt_png_paths))):
                gt = cv2.imread(gt_png_paths[i])
                infer = cv2.imread(infer_png_paths[i])
                ssim = utils.caculate_ssim(gt, infer)
                psnr = utils.calculate_psnr(gt, infer)
                log.write(
                    "gt {}, infer {}, ssim {:.4f}, psnr {:.4f}\n".format(
                        gt_png_paths[i], infer_png_paths[i], ssim, psnr
                    )
                )

                ssim_res.add(ssim)
                psnr_res.add(psnr)
            log.write(
                "ssim {:.4f}, psnr {:.4f} over {} frames\n".format(
                    ssim_res.item(), psnr_res.item(), len(gt_png_paths)
                )
            )
        print(
            "ssim {:.4f}, psnr {:.4f} over {} frames\n".format(
                ssim_res.item(), psnr_res.item(), len(gt_png_paths)
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 读取配置文件
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 获取保存路径
    save_name = config["save_name"]
    if args.name is not None:
        save_name = os.path.join(args.name, save_name)
    save_path = os.path.join(os.path.join(".", "result"), save_name)
    # if args.name is None:
    #     save_path = None

    # 加载数据集
    spec = config["test_dataset"]
    dataset = dloader.loadDataset(spec["dataset"])
    dataset = dloader.loadDataset(spec["wrapper"], args={"dataset": dataset})
    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # 加载模型
    model_spec = torch.load(args.model)["model"]
    model = mloader.loadModel(model_spec, load_sd=True).cuda()

    # 测试
    utils.ensure_path(save_path)
    eval_model(
        loader,
        model,
        model_spec["name"],
        out_path=save_path,
        mtsutil=config["mtsutil_path"],
    )
