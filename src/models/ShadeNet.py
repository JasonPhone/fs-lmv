import torch.nn as nn
import torch.nn.functional as F
import torch
from .ShadeNetModules import (
    ResBlock,
    DoubleConv,
    SingleConv,
    RecurrentEncoder,
    ConvLSTMCell,
    midOutChannels,
)
from .loader import RegisterModel


def warp(previous_frame, motion_vector):
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


@RegisterModel("ShadeNet")
class ShadeNet(nn.Module):
    """
    Encoder for down-sampling + decoder for up-sampling.
    Skip and concat here.
    """

    def __init__(self, device="cuda") -> None:
        super().__init__()
        """
        #   act_func: prelu
        #   class: ShadeNetEncoder
        #   encoders_out_channel: [16, 24, 32, 48, 48]
        #   input_buffer: ["scene_light_no_st", "sky_color", "st_color", "st_alpha"]
        #   skip-layer: True
        #   struct:
        #     encoder: [[24], [32], [48], [48]]
        #     input: [16]
        """
        self.his_encoder_input = nn.ModuleList(
            [DoubleConv(in_c=3, out_c=16, mid_c=16) for _ in range(3)]
        )
        self.his_encoder_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DoubleConv(in_c=16, out_c=24, mid_c=24, strides=[2, 1]),
                        DoubleConv(in_c=24, out_c=32, mid_c=32, strides=[2, 1]),
                        DoubleConv(in_c=32, out_c=48, mid_c=48, strides=[2, 1]),
                        DoubleConv(in_c=48, out_c=48, mid_c=48, strides=[2, 1]),
                    ]
                )
                for _ in range(3)
            ]
        )
        """
        #   act_func: prelu
        #   class: ShadeNetEncoder
        #   encoders_out_channel: [16, 24, 32, 48, 48]
        #   input_buffer: ["brdf_color"]  # Composed g-buffers
        #   output_prefix: se_
        #   skip-layer: True
        #   struct:
        #     encoder: [[24], [32], [48], [48]]
        #     input: [16]
        """
        self.g_encoder_input = DoubleConv(in_c=3, out_c=16, mid_c=16)
        self.g_encoder_layers = nn.ModuleList(
            [
                DoubleConv(in_c=16, out_c=24, mid_c=24, strides=[2, 1]),
                DoubleConv(in_c=24, out_c=32, mid_c=32, strides=[2, 1]),
                DoubleConv(in_c=32, out_c=48, mid_c=48, strides=[2, 1]),
                DoubleConv(in_c=48, out_c=48, mid_c=48, strides=[2, 1]),
            ]
        )
        self.decode_compress = nn.ModuleList(
            [
                RecurrentEncoder(in_c=48, out_c=24),
                RecurrentEncoder(in_c=64, out_c=32),
                RecurrentEncoder(in_c=96, out_c=48),
            ]
        )
        """
        ConvLSTMCell in code/src/models/shade_net/conv_lstm_v5.py
        """
        self.recurrent_units = nn.ModuleList(
            [
                ConvLSTMCell(in_c=24, hidden_c=48, out_c=24, ks=1, bias=True),
                ConvLSTMCell(in_c=32, hidden_c=64, out_c=32, ks=1, bias=True),
                ConvLSTMCell(in_c=48, hidden_c=96, out_c=48, ks=1, bias=True),
            ]
        )
        """
        #   act_func: prelu
        #   class: ShadeNetDecoder
        #   encoders_out_channel: [64, 96, 128, 192, 192]
        #   num_he: 3
        #   output_buffer:
        #     [
        #       { "name": "residual_output", "channel": 3 },
        #       { "name": "pred_smv0_raw", "channel": 2 },
        #       { "name": "pred_smv_res_raw", "channel": 2 },
        #     ]
        #   skip-add: False
        #   skip-cat: True
        #   skip_conn_offset: 1
        #   skip_conn_start: 1
        #   struct:
        #     decoder: [[96, 96], [128, 64], [96, 48], [64, 32]]
        #     output: [32, 7]
        """
        self.decoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    SingleConv(in_c=144, out_c=64),
                    ResBlock(in_c=64, mid_c=64, side_c=32),
                    nn.ConvTranspose2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                ),
                nn.Sequential(
                    SingleConv(in_c=192, out_c=96),
                    ResBlock(in_c=96, mid_c=96, side_c=48),
                    nn.ConvTranspose2d(
                        in_channels=96,
                        out_channels=48,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                ),
                nn.Sequential(
                    SingleConv(in_c=288, out_c=128),
                    ResBlock(in_c=128, mid_c=128, side_c=64),
                    nn.ConvTranspose2d(
                        in_channels=128,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                ),
                nn.Sequential(
                    SingleConv(in_c=192, out_c=96),
                    ResBlock(in_c=96, mid_c=96, side_c=96),
                    nn.ConvTranspose2d(
                        in_channels=96,
                        out_channels=96,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                ),
            ]
        )
        self.decoder_output = nn.Sequential(
            SingleConv(in_c=32, out_c=32, ks=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=7, kernel_size=1),
        )

    def beginState(self, device, w, h):
        g_state = [
            [
                torch.zeros((1, 24, h // 2, w // 2)).clone().detach().to(device),
                torch.zeros((1, 32, h // 4, w // 4)).clone().detach().to(device),
                torch.zeros((1, 48, h // 8, w // 8)).clone().detach().to(device),
            ]
            for _ in range(3)
        ]
        decoder_state = [
            [
                torch.zeros((1, 24, h // 2, w // 2)).clone().detach().to(device),
                torch.zeros((1, 32, h // 4, w // 4)).clone().detach().to(device),
                torch.zeros((1, 48, h // 8, w // 8)).clone().detach().to(device),
            ]
            for _ in range(3)
        ]
        return g_state, decoder_state

    def forward(
        self,
        brdf_color,
        dmdl_frame0,
        dmdl_frame1,
        dmdl_frame2,
        rmv0,
        rmv1,
        rmv2,
        g_state,
        decoder_state,
    ):
        """
        Gbuffer encoding.
        Input: brdf_color of f_i. Created using rough, nov, albedo, metallic, spec.
        """
        g_input = self.g_encoder_input(brdf_color)
        g_skip_layers = []
        g_skip_layers.append(self.g_encoder_layers[0](g_input))
        for his_idx in range(1, len(self.g_encoder_layers)):
            code = self.g_encoder_layers[his_idx](g_skip_layers[his_idx - 1])
            g_skip_layers.append(code)
        # print("g_skip_layers", [t.shape for t in g_skip_layers])

        """
        History encoding.
        Input: Other buffers and masks of f_{i..i-2}.
        """
        h_skip_layers = []
        h_input_frames = [dmdl_frame0, dmdl_frame1, dmdl_frame2]
        for idx, input_layer in enumerate(self.his_encoder_input):
            h_input = input_layer(h_input_frames[idx])
            h_scp = []
            h_scp.append(self.his_encoder_layers[idx][0](h_input))
            for his_idx in range(1, len(self.his_encoder_layers[idx])):
                code = self.his_encoder_layers[idx][his_idx](h_scp[his_idx - 1])
                h_scp.append(code)
            h_skip_layers.append(h_scp)
        # print("h_skip_layers", [t.shape for t in h_skip_layers[0]])

        """
        Recurrent pass, for each frame i..i-2.
        Input:
            current h_encoder skip layers,
            previous decoder skip layers,
            previous g_encoder skip layers.
        Output: Overwrite(update) h_encoder skip layers.
        NOTE: Shared LSTM weights but individual his recurrent.
        TODO: Try one recurrent state stream.
        """
        h_skip_layers_recur = []
        for his_idx, skip_layers in enumerate(h_skip_layers):
            h_scp_recur = []
            for idx, h_scp in enumerate(skip_layers):
                if idx < 3:
                    prev_g_scp = g_state[his_idx][idx]
                    prev_decoder_scp = decoder_state[his_idx][idx]
                    # print(f"recur at layer {layer_idx}")
                    h_scp = self.recurrent_units[idx](
                        h_scp, [prev_decoder_scp, prev_g_scp]
                    )
                h_scp_recur.append(h_scp)
            h_skip_layers_recur.append(h_scp_recur)

        """
        Decoding.
        Input:
            RMV or LMV {i..i-2}->i+1,
            (updated) h_encoder skip layers of f_{i..i-2},
            g_encoder skip layers,
            last layer output of decoder.
        Output:
            residual L of f_{i+1},
            LMV i->i+1,
            compressed encoder skip layers.
        """

        def resize(tensor, scale):
            return F.interpolate(
                tensor, scale_factor=scale, mode="bilinear", align_corners=False
            )

        smv_list = []
        smv_residual_list = []
        mv_list = [[rmv0, rmv1, rmv2]]
        decoder_layer_output = []

        def getLMV(tensor, i, idx, update=True):
            if i == 0 and update:
                smv_list.append([])
                smv_residual_list.append([])
            smv_raw = tensor[:, i * 4 : i * 4 + 2]
            smv = torch.tanh(smv_raw)
            smv_residual_raw = tensor[:, i * 4 + 2 : i * 4 + 4]
            smv_residual = torch.tanh(smv_residual_raw)
            if idx > 0:
                last_smv = smv_list[idx - 1]
                last_smv_residual = smv_residual_list[idx - 1]
                smv = smv + resize(last_smv[i], smv.shape[2] / last_smv[i].shape[2])
                smv_residual = smv_residual + resize(
                    last_smv_residual[i],
                    smv_residual.shape[2] / last_smv_residual[i].shape[2],
                )
            scaled_rmv = resize(mv_list[0][i], smv.shape[2] / mv_list[0][i].shape[2])
            rmv_warp = warp(scaled_rmv, smv)
            lmv = rmv_warp + smv_residual
            if update:
                smv_list[idx].append(smv)
                smv_residual_list[idx].append(smv_residual)
            #     print(f"append smv and residual [{idx}]")
            # print(
            #     f"getMV from tensor {tensor.shape}, h{i} layer {4+layer_idx}, got {lmv.shape}"
            # )
            return lmv

        for idx in range(0, 4):
            layer_idx = -idx - 1
            # print(f"-----------decoder layer {4 + layer_idx}")
            g_scp_layer = g_skip_layers[layer_idx]
            cat_list = []
            for his_idx in range(3):
                h_scp_layer = h_skip_layers_recur[his_idx][layer_idx]
                # print(f"his {his_idx}")
                for frame_idx in range(his_idx, -1, -1):
                    # print(f"warp {frame_idx}")
                    mv = mv_list[idx][frame_idx]
                    scaled_mv = resize(
                        mv,
                        h_scp_layer.shape[2] / mv.shape[2],
                    )
                    h_scp_layer = warp(h_scp_layer, scaled_mv)
                cat_list.append(h_scp_layer)
                # print(
                #     f"cat warped h_scp_layer[{his_idx}][{layer_idx}] {h_scp_layer.shape}"
                # )

            cat_list.append(g_scp_layer)
            # print(f"cat g_scp_layer[{layer_idx}] {g_scp_layer.shape}")
            if idx > 0:
                # Prepare next mv.
                cat_list.extend(smv_list[idx - 1])
                # print(f"cat smv [{idx-1}] {smv_list[idx-1][0].shape}")
                cat_list.extend(smv_residual_list[idx - 1])
                # print(f"cat smv residual [{idx-1}] {smv_residual_list[idx-1][0].shape}")
                cat_list.append(decoder_layer_output[idx - 1][:, 12:])
                # print(f"cat decoder layer output [{idx-1}]")

            # print("cat list", [t.shape for t in cat_list])
            layer_input = torch.cat(cat_list, dim=1)
            layer_output = self.decoder_layers[layer_idx](layer_input)
            # print(
            #     f"decoder layer input {layer_input.shape} output {layer_output.shape}"
            # )
            decoder_layer_output.append(layer_output)
            # lmv_list = []
            # for hi in range(3):
            #     lmv_list.append(getLMV(decoder_layer_output[idx], hi, idx))
            # mv_list.append(lmv_list)
            mv_list.append(
                [getLMV(decoder_layer_output[idx], i, idx) for i in range(3)]
            )
            # print()
        # print("decoder layer output", [t.shape for t in decoder_layer_output])

        # def iterative_warp(tensor, mvs):
        #     res = tensor
        #     for mv in mvs:
        #         res = warp(res, mv)
        #     return res

        # rmv_list = [rmv0, rmv1, rmv2]
        # """ decoder 3 """
        # cat_list = []
        # for i in range(3):
        #     h_scp_layer = h_skip_layers_recur[i][3]
        #     resized_rmv = resize(
        #         rmv_list[i], h_scp_layer.shape[2] / rmv_list[i].shape[2]
        #     )
        #     cat_list.append(warp(h_scp_layer, resized_rmv))
        # cat_list.append(g_skip_layers[3])
        # decoder_3_input = torch.cat(cat_list, dim=1)
        # decoder_3_output = self.decoder_layers[3](decoder_3_input)

        # """ decoder 2 """
        # cat_list = []
        # for i in range(3):
        #     h_scp_layer = h_skip_layers_recur[i][2]
        #     resized_rmv = resize(
        #         rmv_list[i], h_scp_layer.shape[2] / rmv_list[i].shape[2]
        #     )
        #     cat_list.append(warp(h_scp_layer, resized_rmv))
        # cat_list.append(g_skip_layers[2])
        # decoder_2_smv0 = decoder_3_output[:, 0:2]
        # decoder_2_smv0_resi = decoder_3_output[:, 2:4]

        # decoder_2_smv1 = decoder_3_output[:, 4:6]
        # decoder_2_smv1_resi = decoder_3_output[:, 6:8]

        # decoder_2_smv2 = decoder_3_output[:, 8:10]
        # decoder_2_smv2_resi = decoder_3_output[:, 10:12]

        # """ decoder 1 """
        # """ decoder 0 """
        # """ decoder output """

        # decoder_layer_output = []

        """
        Merge output.
        """
        final_lmv = getLMV(decoder_layer_output[-1], 0, idx=0, update=False)
        brdf_residual = self.decoder_output(decoder_layer_output[-1])[:, 0:3]
        # print(f"final lmv {final_lmv.shape}, brdf_residual {brdf_residual.shape}")
        brdf_warp = warp(brdf_color, final_lmv)
        brdf_result = brdf_residual + brdf_warp
        """
        Gather states.
        next_g_states = g_skip_layers.
        next_decoder_states = compress(decoder_output).
        """
        next_g_states = [t.clone().detach() for t in g_skip_layers]
        # print("decoder layer output", [t.shape for t in decoder_layer_output])
        next_decoder_states = [
            self.decode_compress[i](t).clone().detach()
            for i, t in enumerate(decoder_layer_output[2::-1])
        ]

        return (
            brdf_result,
            next_g_states,
            next_decoder_states,
            smv_list,
            smv_residual_list,
        )
