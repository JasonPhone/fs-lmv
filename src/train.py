import argparse
import os
import yaml
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import datasets.loader as dloader
import models.loader as mloader
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils import warp
import utils
import torchvision.utils as tu
from torchvision.transforms import ToPILImage
from loss import mLoss
import numpy as np


def make_data_loader(spec, tag=""):
    if spec is None:
        return None

    dataset = dloader.loadDataset(spec["dataset"])
    dataset = dloader.loadDataset(spec["wrapper"], args={"dataset": dataset})

    log("{} dataset: size={}".format(tag, len(dataset)))

    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get("train_dataset"), tag="train")
    return train_loader


def prepare_training():
    if config.get("resume") is not None:
        sv_file = torch.load(config["resume"])
        model = mloader.loadModel(sv_file["model"], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file["optimizer"], load_sd=True
        )
        epoch_start = sv_file["epoch"] + 1
        if config.get("multi_step_lr") is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config["multi_step_lr"])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = mloader.loadModel(config["model"]).cuda()
        optimizer = utils.make_optimizer(model.parameters(), config["optimizer"])
        epoch_start = 1
        if config.get("multi_step_lr") is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config["multi_step_lr"])

    log("model: #params={}".format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train_model(train_loader, model, model_name, optimizer):
    model.train()
    train_loss = utils.Averager()
    loss_fn = mLoss()

    if model_name == "ShadeNet":
        for batch in tqdm(train_loader, leave=False, desc="train"):
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
                dmdl_frame0 = dmdl_frame_batch[idx - 1 : idx - 0]
                dmdl_frame1 = dmdl_frame_batch[idx - 2 : idx - 1]
                dmdl_frame2 = dmdl_frame_batch[idx - 3 : idx - 2]
                rmv0 = rmv_batch[idx - 0 : idx + 1]
                rmv1 = rmv_batch[idx - 1 : idx - 0]
                rmv2 = rmv_batch[idx - 2 : idx - 1]
                gt = gt_batch[idx : idx + 1]

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
                gt_L = dmdl_frame_batch[idx : idx + 1]

                loss = loss_fn(I, gt, L, gt_L, smv_list, smv_resi_list)
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                train_loss.add(loss.item())

    return train_loss.item()


def main(_config, save_path):
    global config, log, writer
    config = _config
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader = make_data_loaders()
    if config.get("data_norm") is None:
        config["data_norm"] = {
            "inp": {"sub": [0], "div": [1]},
            "gt": {"sub": [0], "div": [1]},
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    epoch_max = config["epoch_max"]
    epoch_save = config.get("epoch_save")
    loss_min = 10

    # cnt = 0
    # for batch in train_loader:
    #     bs = batch["batch_size"]
    #     nf = batch["next_frame"]
    #     bi = batch["batch_index"]
    #     cnt += 1
    #     print(f"b {cnt}")
    #     print(f"bi {bi}")
    #     print(f"bs {bs}")
    #     print(f"next frame {nf}")
    # return

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ["epoch {}/{}".format(epoch, epoch_max)]

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        torch.autograd.anomaly_mode.set_detect_anomaly(False)
        train_loss = train_model(
            train_loader, model, config["model"]["name"], optimizer
        )
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append("train: loss={:.4f}".format(train_loss))
        writer.add_scalars("loss", {"train": train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config["model"]
        model_spec["sd"] = model_.state_dict()
        optimizer_spec = config["optimizer"]
        optimizer_spec["sd"] = optimizer.state_dict()
        sv_file = {"model": model_spec, "optimizer": optimizer_spec, "epoch": epoch}

        torch.save(sv_file, os.path.join(save_path, "epoch-last.pth"))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path, "epoch-{}.pth".format(epoch)))

        if train_loss < loss_min:
            loss_min = train_loss
            torch.save(sv_file, os.path.join(save_path, "epoch-best.pth"))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append("{} {}/{}".format(t_epoch, t_elapsed, t_all))

        log(", ".join(log_info))
        writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print("config loaded.")

    save_name = args.name
    if save_name is None:
        save_name = "_" + args.config.split(os.sep)[-1][: -len(".yaml")]
    save_path = os.path.join(os.path.join(".", "save"), save_name)

    torch.manual_seed(3407)
    main(config, save_path)
