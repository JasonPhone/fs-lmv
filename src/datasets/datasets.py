"""
Load raw data.
"""

import torch
from torch.utils.data import Dataset
from .loader import RegisterDataset
import os
import sys
from utils import read_exr
import random


@RegisterDataset("DmdlDatasetLocal")
class DmdlDatasetLocal(Dataset):
    def __init__(
        self,
        root_path,
        working_resolution,
        batch_size,
        frame_offset,
        frame_st,
        frame_ed,
    ):
        self.path_root = root_path
        print(f"dataset\n\tbatch_size {batch_size}\n\troot {root_path}")
        self.batch_size = batch_size
        """
        need:
            i: rough, nov, base_color, metallic, spec -> g_encoder
            i, i-1, i-2: frame_dmdl -> his_encoder
            i, i-1, i-2: rmv
        """

        path_prefix = "MedievalDocks_ForRecord"
        self.path_base_color = path_prefix + "BaseColor.{}.exr"
        self.path_mv_metal_rough = (
            path_prefix + "MotionVectorAndMetallicAndRoughness.{}.exr"
        )
        self.path_nov = path_prefix + "NoV.{}.exr"
        self.path_specular = path_prefix + "Specular.{}.exr"
        self.path_gt = path_prefix + "PreTonemapHDRColor.{}.exr"

        self.frame_st, self.frame_ed = frame_st, frame_ed
        self.frame_offset = frame_offset

        print(f"\ttotal frames {self.frame_ed- self.frame_st}")
        print(f"\tbatch start from frame {self.frame_st}")

    def __len__(self):
        return (self.frame_ed - self.frame_st) // self.batch_size

    def __getitem__(self, index):
        def loadFrame(temp, index, channel=3):
            path = temp.format(str(index).rjust(4, "0"))
            path = os.path.join(self.path_root, path)
            return read_exr(path, channel=channel)

        batch_frame_offset = index * self.batch_size + self.frame_st

        render_mv = []  # channel: 2
        metallic = []  # channel: 1
        roughness = []  # channel: 1
        specular = []  # channel: 1
        nov = []  # channel: 1
        base_color = []  # channel: 3
        frame_dmdl = []  # channel: 3
        gt = []  # channel: 3

        for i in range(0, self.batch_size + self.frame_offset+1):
            """
            frame: i-2, i-1, i+0, i+1, i+2, ...
            index:   0,   1,   2,   3,   4, ...
            """
            idx = batch_frame_offset + i - self.frame_offset
            idx = self.frame_ed - 1 if idx >= self.frame_ed else idx
            mv_metal_rough = loadFrame(self.path_mv_metal_rough, idx, channel=4)
            render_mv.append(mv_metal_rough[0:2])
            metallic.append(mv_metal_rough[2:3])
            roughness.append(mv_metal_rough[3:4])

            specular.append(loadFrame(self.path_specular, idx)[0:1])
            nov.append(loadFrame(self.path_nov, idx)[0:1])

            base_color.append(loadFrame(self.path_base_color, idx))
            gt.append(loadFrame(self.path_gt, idx))

            bc = base_color[-1]
            bc[bc <= 0] = 1.0  # Avoid div0.
            frame_dmdl.append(gt[-1] / bc)

        return {
            "render_mv": render_mv,
            "metallic": metallic,
            "roughness": roughness,
            "specular": specular,
            "nov": nov,
            "base_color": base_color,
            "frame_dmdl": frame_dmdl,
            "gt": gt,
            "batch_size": self.batch_size,
            "frame_offset": self.frame_offset,
        }
