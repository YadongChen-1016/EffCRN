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


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeconvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.deconvblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=((kernel_size[0] - 2) // 2, 0)),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        output = self.deconvblock(x)
        return output


class GRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, bidirectional=False):
        super(GRUBlock, self).__init__()
        self.gru = nn.GRU(in_channels, hidden_channels, bidirectional=bidirectional, bias=True)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, rnn_state):
        gru_out, rnn_state = self.gru(x, rnn_state)
        output = self.linear(gru_out)
        return output, rnn_state


class EffCRN23(nn.Module):
    def __init__(self, in_channels, out_channels, F, N):
        super(EffCRN23, self).__init__()
        self.N = N
        self.en_convblock_1 = ConvBlock(in_channels, F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.en_downblock_1 = ConvBlock(F, F, kernel_size=(N, 1), stride=(2, 1), padding="valid")

        self.en_convblock_2 = ConvBlock(F, 2 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.en_downblock_2 = ConvBlock(2 * F, 2 * F, kernel_size=(N, 1), stride=(2, 1), padding="valid")

        self.en_convblock_3 = ConvBlock(2 * F, 3 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.en_downblock_3 = ConvBlock(3 * F, 3 * F, kernel_size=(N, 1), stride=(2, 1), padding="valid")

        self.en_convblock_4 = ConvBlock(3 * F, 4 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.en_downblock_4 = ConvBlock(4 * F, 4 * F, kernel_size=(N, 1), stride=(2, 1), padding="valid")

        self.en_convblock_5 = ConvBlock(4 * F, 5 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")
        self.en_downblock_5 = ConvBlock(5 * F, 5 * F, kernel_size=(N, 1), stride=(2, 1), padding="valid")

        self.convlstm = ConvLSTM(9, 5 * F, F, kernel_size=N, cnn_dropout=0.2, rnn_dropout=0.2, batch_first=True, bias=False)
        self.gru = GRUBlock(F, 9 * F, F, bidirectional=False)

        self.de_upblock_1 = DeconvBlock(F, 5 * F, kernel_size=(N, 1), stride=(2, 1))
        self.de_skip_conv_1 = ConvBlock(5 * F, 5 * F, kernel_size=(1, 1), stride=(1, 1), padding="valid")
        self.de_convblock_1 = ConvBlock(5 * F, 5 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")

        self.de_upblock_2 = DeconvBlock(5 * F, 4 * F, kernel_size=(N, 1), stride=(2, 1))
        self.de_skip_conv_2 = ConvBlock(4 * F, 4 * F, kernel_size=(1, 1), stride=(1, 1), padding="valid")
        self.de_convblock_2 = ConvBlock(4 * F, 4 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")

        self.de_upblock_3 = DeconvBlock(4 * F, 3 * F, kernel_size=(N, 1), stride=(2, 1))
        self.de_skip_conv_3 = ConvBlock(3 * F, 3 * F, kernel_size=(1, 1), stride=(1, 1), padding="valid")
        self.de_convblock_3 = ConvBlock(3 * F, 3 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")

        self.de_upblock_4 = DeconvBlock(3 * F, 2 * F, kernel_size=(N, 1), stride=(2, 1))
        self.de_skip_conv_4 = ConvBlock(2 * F, 2 * F, kernel_size=(1, 1), stride=(1, 1), padding="valid")
        self.de_convblock_4 = ConvBlock(2 * F, 2 * F, kernel_size=(N, 1), stride=(1, 1), padding="same")

        self.de_upblock_5 = DeconvBlock(2 * F, F, kernel_size=(N, 1), stride=(2, 1))
        self.de_skip_conv_5 = ConvBlock(F, F, kernel_size=(1, 1), stride=(1, 1), padding="valid")
        self.de_convblock_5 = ConvBlock(F, F, kernel_size=(N, 1), stride=(1, 1), padding="same")

        self.out_convblock = ConvBlock(F, out_channels, kernel_size=(N, 1), stride=(1, 1), padding="same", activate=False)

    def mask_func(self, noisy, estimate):
        r, i = noisy[:, 0, :, :], noisy[:, 1, :, :]
        r_hat, i_hat = estimate[:, 0, :, :], noisy[:, 1, :, :]
        mag_mask = torch.sqrt(r_hat ** 2 + i_hat ** 2)
        phase_rotate = torch.atan2(i_hat, r_hat)
        mag_mask = torch.tanh(mag_mask)
        mag = mag_mask * torch.sqrt(r ** 2 + i ** 2)
        phase = phase_rotate + torch.atan2(i, r)
        # return real, imag
        return mag * torch.cos(phase), mag * torch.sin(phase)

    def forward(self, x, rnn_state_1=None, rnn_state_2=None):
        input = x
        pad = torch.nn.functional.pad(input, (0, 0, 0, 3), "constant", 0)
        en_conv_1 = self.en_convblock_1(pad)
        pad_en1 = torch.nn.functional.pad(en_conv_1, (0, 0, 0, self.N - 2), "constant", 0)
        en_down_1 = self.en_downblock_1(pad_en1)

        en_conv_2 = self.en_convblock_2(en_down_1)
        pad_en2_1 = torch.nn.functional.pad(en_conv_2, (0, 0, 0, self.N - 2), "constant", 0)
        en_down_2 = self.en_downblock_2(pad_en2_1)
        pad_en2_2 = torch.nn.functional.pad(en_down_2, (0, 0, 0, 1), "constant", 0)

        en_conv_3 = self.en_convblock_3(pad_en2_2)
        pad_en3_1 = torch.nn.functional.pad(en_conv_3, (0, 0, 0, self.N - 2), "constant", 0)
        en_down_3 = self.en_downblock_3(pad_en3_1)
        pad_en3_2 = torch.nn.functional.pad(en_down_3, (0, 0, 0, 1), "constant", 0)

        en_conv_4 = self.en_convblock_4(pad_en3_2)
        pad_en4_1 = torch.nn.functional.pad(en_conv_4, (0, 0, 0, self.N - 2), "constant", 0)
        en_down_4 = self.en_downblock_4(pad_en4_1)
        pad_en4_2 = torch.nn.functional.pad(en_down_4, (0, 0, 0, 1), "constant", 0)

        en_conv_5 = self.en_convblock_5(pad_en4_2)
        pad_en5_1 = torch.nn.functional.pad(en_conv_5, (0, 0, 0, self.N - 2), "constant", 0)
        en_down_5 = self.en_downblock_5(pad_en5_1)

        conv_lstm, state_out_1 = self.convlstm(en_down_5.squeeze(-1).unsqueeze(1), rnn_state_1)
        conv_lstm = conv_lstm.squeeze(1).transpose(1, 2)
        gru_out, state_out_2 = self.gru(conv_lstm, rnn_state_2)
        gru_out = gru_out.transpose(1, 2).unsqueeze(-1)

        de_up_1 = self.de_upblock_1(gru_out)
        de_skip_1 = self.de_skip_conv_1(en_conv_5)
        de_conv_1 = self.de_convblock_1(de_up_1 + de_skip_1)

        de_up_2 = self.de_upblock_2(de_conv_1[:, :, :-1, :])
        de_skip_2 = self.de_skip_conv_2(en_conv_4)
        de_conv_2 = self.de_convblock_2(de_up_2 + de_skip_2)

        de_up_3 = self.de_upblock_3(de_conv_2[:, :, :-1, :])
        de_skip_3 = self.de_skip_conv_3(en_conv_3)
        de_conv_3 = self.de_convblock_3(de_up_3 + de_skip_3)

        de_up_4 = self.de_upblock_4(de_conv_3[:, :, :-1, :])
        de_skip_4 = self.de_skip_conv_4(en_conv_2)
        de_conv_4 = self.de_convblock_4(de_up_4 + de_skip_4)

        de_up_5 = self.de_upblock_5(de_conv_4)
        de_skip_5 = self.de_skip_conv_5(en_conv_1)
        de_conv_5 = self.de_convblock_5(de_up_5 + de_skip_5)

        out = self.out_convblock(de_conv_5)[:, :, :-3, :]
        real, imag = self.mask_func(x, out)
        est = torch.cat([real.unsqueeze(1), imag.unsqueeze(1)], dim=1)
        return est, state_out_1, state_out_2


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
    model = EffCRN23(in_channels=2, out_channels=2, F=17, N=4)
    TO_SAMPLES = TorchOLA(frame_shift=hop_length)
    TO_FRAME = TorchSignalToFrames(frame_size=n_fft, frame_shift=hop_length)
    h_zeros_1 = torch.zeros(1, 17, 9, dtype=x.dtype, device=x.device)
    c_zeros_1 = torch.zeros(1, 17, 9, dtype=x.dtype, device=x.device)
    h_zeros_2 = torch.zeros(1, 9, 17 * 9, dtype=x.dtype, device=x.device)
    state_list_1 = [h_zeros_1, c_zeros_1]
    state_list_2 = h_zeros_2
    input = TO_FRAME(x)
    window = torch.sqrt(torch.from_numpy(np.hanning(n_fft).astype(np.float32)))
    spec = torch.view_as_real(torch.fft.rfft(input)).permute(0, 3, 2, 1).contiguous()
    _, _, _, t = spec.size()
    est_out = torch.zeros(spec.size(), dtype=spec.dtype, device=spec.device)
    for i in range(t):
        input_frame = spec[:, :, :, i].unsqueeze(-1)
        est_out_t, state_list_1, state_list_2 = model(input_frame, state_list_1, state_list_2)
        est_out[:, :, :, i] = est_out_t.squeeze(-1)
    summary(model, input_size=(1, 2, 257, 1))
