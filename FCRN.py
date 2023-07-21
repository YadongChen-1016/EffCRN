# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np

from torchinfo import summary

from ConvLSTM import ConvLSTM


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding="same", activate=True):
        super(ConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if activate:
            self.convblock = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0),
                nn.LeakyReLU(0.2))
        else:
            self.convblock = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0),
                nn.Identity())

    def forward(self, x):
        global pad_x
        if self.padding == "same":
            pad = self.kernel_size[0] - 1
            pad_x = torch.nn.functional.pad(x, (0, 0, 0, pad), "constant", 0)
        elif self.padding == "valid":
            pad_x = x
        output = self.convblock(pad_x)
        return output


class FCRN(nn.Module):
    def __init__(self, in_channels, out_channels, F, N):
        super(FCRN, self).__init__()
        self.en_convblock_1 = ConvBlock(in_channels, F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.en_convblock_2 = ConvBlock(F, F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.down_sampling_1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.en_convblock_3 = ConvBlock(F, 2 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.en_convblock_4 = ConvBlock(2 * F, 2 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.down_sampling_2 = nn.MaxPool2d(kernel_size=(2, 1))

        self.convlstm = ConvLSTM(65, 2 * F, F, kernel_size=N, cnn_dropout=0.2, rnn_dropout=0.2, batch_first=True, bias=False)

        self.up_sampling_1 = nn.Upsample(scale_factor=(2, 1))
        self.de_convblock_1 = ConvBlock(F, 2 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.de_convblock_2 = ConvBlock(2 * F, 2 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.up_sampling_2 = nn.Upsample(scale_factor=(2, 1))
        self.de_convblock_3 = ConvBlock(2 * F, F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.de_convblock_4 = ConvBlock(F, F, kernel_size=(N, 1), stride=(1, 1), padding="same")

        self.out_convblock = ConvBlock(F, out_channels, kernel_size=(N, 1), stride=(1, 1), padding="same", activate=False)

    def forward(self, x, rnn_state=None):
        input = x
        pad = torch.nn.functional.pad(input, (0, 0, 0, 3), "constant", 0)
        en_conv_1 = self.en_convblock_1(pad)
        en_conv_2 = self.en_convblock_2(en_conv_1)
        down_1 = self.down_sampling_1(en_conv_2)
        en_conv_3 = self.en_convblock_3(down_1)
        en_conv_4 = self.en_convblock_4(en_conv_3)
        down_2 = self.down_sampling_2(en_conv_4)

        conv_lstm, state_out = self.convlstm(down_2.squeeze(-1).unsqueeze(1), rnn_state)
        conv_lstm = conv_lstm.squeeze(1).unsqueeze(-1)
        up_1 = self.up_sampling_1(conv_lstm)
        de_conv_1 = self.de_convblock_1(up_1)
        de_conv_2 = self.de_convblock_2(de_conv_1)
        up_2 = self.up_sampling_2(de_conv_2 + en_conv_4)
        de_conv_3 = self.de_convblock_3(up_2)
        de_conv_4 = self.de_convblock_4(de_conv_3)
        out = self.out_convblock(de_conv_4 + en_conv_2)[:, :, :-3, :]
        return out, state_out


class TorchSignalToFrames(object):
    def __init__(self, frame_size=128, frame_shift=64):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        nframes = math.ceil((sig_len - self.frame_size) / self.frame_shift + 1)
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class TorchOLA(object):
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def __call__(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device,
                          requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


if __name__ == '__main__':
    n_fft = 512
    hop_length = 256
    x = torch.randn(1, 16000)

    model = FCRN(in_channels=6, out_channels=2, F=88, N=24)
    TO_SAMPLES = TorchOLA(frame_shift=hop_length)
    TO_FRAME = TorchSignalToFrames(frame_size=n_fft, frame_shift=hop_length)
    h_zeros = torch.zeros(1, 88, 65, dtype=x.dtype, device=x.device)
    c_zeros = torch.zeros(1, 88, 65, dtype=x.dtype, device=x.device)
    state_list = [h_zeros, c_zeros]
    input = TO_FRAME(x)
    window = torch.sqrt(torch.from_numpy(np.hanning(n_fft).astype(np.float32)))
    spec = torch.view_as_real(torch.fft.rfft(input)).permute(0, 3, 2, 1).contiguous()
    _, _, _, t = spec.size()
    for i in range(t):
        if i < t - 2:
            input_frame = torch.cat([spec[:, :, :, i], spec[:, :, :, i+1], spec[:, :, :, i+2]], dim=1).unsqueeze(-1)
        elif i == t - 2:
            input_frame = torch.cat([spec[:, :, :, i-1], spec[:, :, :, i], spec[:, :, :, i+1]], dim=1).unsqueeze(-1)
        else:
            input_frame = torch.cat([spec[:, :, :, i-2], spec[:, :, :, i-1], spec[:, :, :, i]], dim=1).unsqueeze(-1)

        out, state_list = model(input_frame, state_list)
    summary(model, input_size=(1, 6, 257, 1))
