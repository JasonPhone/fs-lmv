import shutil
import time
import torch
import os
import numpy as np
from torch.optim import SGD, Adam

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import torch.nn.functional as F
import sys
import lpips
import math
from skimage.metrics import structural_similarity
from tensorboardX import SummaryWriter


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def caculate_ssim(img1, img2):
    img1_g = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return structural_similarity(img1_g, img2_g)


def read_exr(path, channel=3):
    """
    读取exr格式图片
    :param path: 图片路径
    :return: Tensor, 格式为[3, H, W]
    """
    # exr格式图片有四通道，所以需要指定读取模式
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"fail reading {path}")
    image = torch.from_numpy(image[:, :, :channel]).permute(2, 0, 1)
    # openvc读取的图片通道为BGR，将其更改为RGB通道
    if channel == 3:
        image = torch.stack([image[2], image[1], image[0]], dim=0)
    elif channel == 4:
        image = torch.stack([image[2], image[1], image[0], image[3]], dim=0)
    else:
        print("Error: Invalid channel for EXR read")
    return image


def write_exr(tensor, path):
    """
    保存exr格式图片
    :param tensor: 图片数据
    :param path: 想要保存的图片路径
    """
    C, H, W = tensor.shape
    image = tensor
    while C < 3:
        image = torch.cat([image, torch.zeros((1, H, W))], dim=0)
        C = C + 1
    if C == 3:
        image = torch.cat([image, torch.ones((1, H, W))], dim=0)
    image = torch.stack([image[2], image[1], image[0], image[3]], dim=0)
    cv2.imwrite(path, image.permute(1, 2, 0).numpy())


def simpleToneMapping(img):
    errors = img == -1.0
    result = torch.log(torch.ones_like(img) + img)
    result[errors] = 0.0
    return result


def simpleDeToneMapping(img):
    result = torch.exp(img) - torch.ones_like(img)
    result[result < 0.0] = 0.0
    return result


def warp(previous_frame, motion_vector):
    """
    将前一帧根据运动矢量映射到当前帧
    :param previous_frame: 前一帧数据
    :param motion_vector: 当前帧的运动矢量
    """
    # print(f"prev f size {previous_frame.size()}, mv size {motion_vector.size()}")
    B, C, H, W = previous_frame.size()
    xx = (
        torch.arange(0, W, dtype=previous_frame.dtype, device=previous_frame.device)
        .view(1, -1)
        .repeat(H, 1)
    )
    yy = (
        torch.arange(0, H, dtype=previous_frame.dtype, device=previous_frame.device)
        .view(-1, 1)
        .repeat(1, W)
    )
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).to(motion_vector.device)
    grid[:, 0, :, :] = grid[:, 0, :, :] - motion_vector[:, 0, :, :]  # X 坐标
    grid[:, 1, :, :] = grid[:, 1, :, :] + motion_vector[:, 1, :, :]  # Y 坐标
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / max(W - 1, 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / max(H - 1, 1) - 1
    return F.grid_sample(
        previous_frame,
        grid.permute(0, 2, 3, 1).to(previous_frame.dtype),
        mode="nearest",
        align_corners=True,
    )


def warpCHW(previous_frame, motion_vector):
    res = warp(previous_frame.unsqueeze(0), motion_vector.unsqueeze(0))
    return res.squeeze(0)


def make_coord(shape, ranges=None, flatten=True):
    """
    制作坐标图
    :param shape: 形状
    :param ranges: 坐标范围，默认为[-1,1]
    :param flatten: 是否展平
    :return: 坐标图
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename="log.txt"):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), "a") as f:
            print(obj, file=f)


def ensure_path(path, remove=False):
    """
    若路径存在，则删除并新建，否则直接新建，确保路径存在且为空
    :param path:
    :param remove:
    :return:
    """
    basename = os.path.basename(path.rstrip("/"))
    if os.path.exists(path):
        if remove or (
            basename.startswith("_")
            or input("{} exists, remove? (y/[n]): ".format(path)) == "y"
        ):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            sys.exit()
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=False):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return "{:.1f}M".format(tot / 1e6)
        else:
            return "{:.1f}K".format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {"sgd": SGD, "adam": Adam}[optimizer_spec["name"]]
    optimizer = Optimizer(param_list, **optimizer_spec["args"])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec["sd"])
    return optimizer


class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    elif t >= 60:
        return "{:.1f}m".format(t / 60)
    else:
        return "{:.1f}s".format(t)


class Averager:
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        # self.v = (self.v * self.n + v * n) / (self.n + n)
        self.v += n * v
        self.n += n

    def item(self):
        # return self.v
        return self.v / self.n


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == "benchmark":
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == "div2k":
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def calc_ssim(sr, hr):
    # tensor转numpy
    prediction = sr.detach().clone().squeeze(0).cpu().numpy().transpose([1, 2, 0])
    target = hr.detach().clone().squeeze(0).cpu().numpy().transpose([1, 2, 0])
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calc_lpips(prediction, target):
    prediction = prediction.cuda()
    target = target.cuda()
    loss_net = lpips.LPIPS(net="vgg").cuda()
    loss_net.eval()
    return loss_net(prediction * 2 - 1, target * 2 - 1).mean()


lut = read_exr("./assets/lut.exr")[0:2].unsqueeze(0)


def create_brdf_color(roughness, nov, albedo, metallic, specular, skybox_mask=None):
    global lut
    if lut.device != roughness.device:
        lut = lut.to(roughness.device)
    if len(nov.shape) == 4:
        op_dim = 1
    else:
        op_dim = 0
    uv = torch.cat([nov, roughness], dim=op_dim) * 2 - 1
    if op_dim == 0:
        uv = uv.unsqueeze(0)
    uv = uv.permute(0, 2, 3, 1)
    if lut.dtype != uv.dtype:
        lut = lut.type(uv.dtype)
    # log.debug(dict_to_string([lut, uv]))
    if op_dim == 0:
        input_lut = lut
    else:
        input_lut = lut.repeat(uv.shape[0], 1, 1, 1)
    precomputed = torch.nn.functional.grid_sample(
        input=input_lut,
        grid=uv,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    specular_color = 0.08 * specular * albedo + (1 - 0.08 * specular) * metallic
    # log.debug(dict_to_string(data, mmm=True))
    if op_dim == 1:
        brdf_color = (
            albedo * (1 - metallic)
            + precomputed[:, :1, ...] * specular_color
            + precomputed[:, 1:, ...]
        )
    else:
        brdf_color = (
            albedo * (1 - metallic)
            + precomputed[0, :1, ...] * specular_color
            + precomputed[0, 1:, ...]
        )

    if skybox_mask is not None:
        brdf_color = torch.ones_like(brdf_color) * skybox_mask + brdf_color * (
            1 - skybox_mask
        )
    # arr = brdf_color.reshape((-1, 720, 1280))
    # ToPILImage()(arr).show()
    # input()
    return brdf_color
