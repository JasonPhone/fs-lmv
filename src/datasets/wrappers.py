"""
Process data for model.
"""

from .loader import RegisterDataset
from torch.utils.data import Dataset
from utils import (
    make_coord,
    warp,
    warpCHW,
    write_exr,
    simpleDeToneMapping,
    simpleToneMapping,
    create_brdf_color,
)
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tu
from torchvision.transforms import ToPILImage
import sys


@RegisterDataset("DmdlWrapper")
class DmdlWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 从Dataset中获取一个batch的数据
        batch_data = self.dataset[index]

        render_mv_batch = batch_data["render_mv"]
        metallic_batch = batch_data["metallic"]
        roughness_batch = batch_data["roughness"]
        specular_batch = batch_data["specular"]
        nov_batch = batch_data["nov"]
        base_color_batch = batch_data["base_color"]
        frame_dmdl_batch = batch_data["frame_dmdl"]
        gt_batch = batch_data["gt"]

        batch_size = batch_data["batch_size"]
        frame_offset = batch_data["frame_offset"]

        brdf_color_b = []
        dmdl_frame_b = []
        rmv_b = []
        gt_b = []
        base_color_b = []

        for i in range(len(gt_batch)):
            rmv = render_mv_batch[i]
            metallic = metallic_batch[i]
            roughness = roughness_batch[i]
            specular = specular_batch[i]
            nov = nov_batch[i]
            base_color = base_color_batch[i]
            frame_dmdl = frame_dmdl_batch[i]
            gt = gt_batch[i]

            brdf_color = create_brdf_color(
                roughness, nov, base_color, metallic, specular
            )

            crop_h, crop_w = brdf_color.shape[1], brdf_color.shape[2]
            crop_h = int(crop_h // 16 * 16)
            crop_w = int(crop_w // 16 * 16)

            # write_exr(brdf_color, "./brdf.exr")
            # write_exr(frame_dmdl, "./dmdl.exr")
            # write_exr(rmv, "./rmv.exr")
            # write_exr(gt, "./gt.exr")
            # write_exr(base_color, "./base_color.exr")
            # assert False

            brdf_color_b.append(brdf_color[:, :crop_h, :crop_w])
            dmdl_frame_b.append(frame_dmdl[:, :crop_h, :crop_w])
            rmv_b.append(rmv[:, :crop_h, :crop_w])
            gt_b.append(gt[:, :crop_h, :crop_w])
            base_color_b.append(base_color[:, :crop_h, :crop_w])

        return {
            "brdf_color": torch.stack(brdf_color_b, dim=0),
            "dmdl_frame": torch.stack(dmdl_frame_b, dim=0),
            "rmv": torch.stack(rmv_b, dim=0),
            "gt": torch.stack(gt_b, dim=0),
            "base_color": torch.stack(base_color_b, dim=0),
            "batch_size": batch_size,
            "frame_offset": frame_offset,
        }
