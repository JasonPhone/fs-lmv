# TODO Make configurable.
import torch
import torch.nn as nn

from enum import Enum


class ShadeNetKey(Enum):
    BufferBrdfColor = "brdf_color"
    SkipGEncoder = "gco_skip_layers"
    SkipHEncoder = "hco_skip_layers"
    SkipDecoder = "dec_skip_layers"


def midOutChannels(last_c, channels):
    if len(channels) == 1:
        mid_c = channels[0]
        out_c = channels[0]
    elif len(channels) == 2:
        mid_c = channels[0]
        out_c = channels[1]
    elif len(channels) == 3:
        last_c = channels[0]
        mid_c = channels[1]
        out_c = channels[2]
    return last_c, mid_c, out_c


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, mid_c=None, strides=[1, 1], device="cuda"):

        super().__init__()
        if mid_c is None:
            mid_c = out_c
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=mid_c,
                kernel_size=3,
                stride=strides[0],
                padding=1,
                bias=True,
            ),
            nn.PReLU(num_parameters=1),
            nn.Conv2d(
                in_channels=mid_c,
                out_channels=out_c,
                kernel_size=3,
                stride=strides[1],
                padding=1,
                bias=True,
            ),
            nn.PReLU(num_parameters=1),
        ).to(device)

    def forward(self, x):
        return self.double_conv(x)


class SingleConv(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        ks=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        single_act_c=False,
    ):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(
                in_c,
                out_c,
                ks,
                stride,
                padding,
                dilation,
                groups,
                bias,
            )
        )
        n_param = 1 if single_act_c else out_c
        self.single_conv.add_module("act", nn.PReLU(num_parameters=n_param))

    def forward(self, x):
        return self.single_conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_c, mid_c, side_c, bias=True):
        super().__init__()
        self.side_c = side_c
        self.conv1 = SingleConv(in_c=in_c, out_c=mid_c)
        self.conv5 = nn.Conv2d(
            mid_c, mid_c, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.act = nn.PReLU(mid_c)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(x + self.conv5(out))
        return out


class RecurrentEncoder(nn.Module):
    """
    Used to compress decoder skip layers.
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.compress_conv = nn.Conv2d(
            in_channels=in_c, out_channels=out_c, kernel_size=1
        )
        self.tanh = nn.Tanh()

    def forward(self, data):
        ret = self.compress_conv(data)
        ret = self.tanh(ret)
        return ret


class ConvLSTMCell(nn.Module):
    """
    Recurrent unit.
    """

    def __init__(self, in_c, hidden_c, out_c, ks=1, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__()

        self.in_channel = in_c
        self.hidden_channel = hidden_c
        self.out_channel = out_c

        self.kernel_size = ks
        self.padding = ks // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.hidden_channel,
            out_channels=self.out_channel * 3,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state, c_list=None):
        """
        input_tensor:   history code from this run.
        cur_state:      [
                            1. hidden state of recurrent unit,
                                output of recurrent compress from previous run.
                            2. c of recurrent unit,
                                output of g_encoder layer from previous run.
                        ]
        """
        """
        cc_i + cc_f + cc_g <--> input_tensor + cur_state
        cc_f <--> c_cur <--> cc_i <--> cc_g <--> c_next
        """
        # h_cur, c_cur = cur_state
        # print(f"input {input_tensor.shape}, h_cur {h_cur.shape}, c_cur {c_cur.shape}")

        combined = torch.cat(cur_state, dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)

        if c_list is None:
            c_list = [self.out_channel] * 3
        cc_i, cc_f, cc_g = torch.split(combined_conv, c_list, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        # o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * input_tensor + i * g
        return c_next
