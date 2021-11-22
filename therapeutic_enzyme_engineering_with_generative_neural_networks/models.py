"""Code defining VAE models"""
from typing import List

import numpy as np
import torch
from torch import nn


class VAE(nn.Module):
    """This is an generic class for sequence VAEs."""

    def __init__(self, sequence_length: int, layer_sizes: List[int], z_size, batch_size=128, alphabet_size=28):
        """TODO

        :param sequence_length: TODO
        :param layer_sizes: TODO
        :param z_size: TODO
        :param batch_size: TODO, defaults to 128
        :param alphabet_size: TODO, defaults to 28
        """
        super().__init__()

        self.sequence_length = sequence_length

        self.layer_sizes = layer_sizes
        self.z_size = z_size
        self.batch_size = batch_size
        self.alphabet_size = alphabet_size
        self.data_input_size = (1, self.sequence_length, self.alphabet_size)

        self.padding = []
        n_ = self.sequence_length
        for _ in range(len(layer_sizes)):
            self.padding.append(n_ % 2)
            n_ = np.ceil(n_ / 2.0)

        self.encoder = Encoder(self.layer_sizes, self.padding, self.z_size, self.data_input_size)

        self.encoded_full_size = self.encoder.encoded_full_size
        self.encoded_flat_size = self.encoder.encoded_flat_size

        self.decoder = Decoder(
            self.layer_sizes, self.padding, self.z_size, self.encoded_flat_size, self.encoded_full_size
        )
        self.step_number = 0

    def sample_gaussian(self, mu, logvar):
        """TODO

        :param mu: TODO
        :param logvar: TODO
        :return: TODO
        """
        std = torch.exp(logvar * 0.5)
        eps = torch.normal(mean=torch.zeros_like(std), std=torch.ones_like(std)).to(std.device)
        return (eps * std) + mu

    def forward(self, x):
        """TODO

        :param x: TODO
        :return: TODO
        """
        mu, logvar = self.encoder(x)

        z_sample = self.sample_gaussian(mu, logvar)

        recon = self.decoder(z_sample)
        return recon, mu, logvar, z_sample


######################################################
# Encoding / Decoding Layers #########################
######################################################


class EncodingConvLayer(nn.Module):
    """This is an class for module that contains conv, batch norm, elu and pooling."""

    def __init__(self, input_size, layer_size, kernel_size=(3, 3), asymmetric_padding=False):
        """TODO

        :param input_size: TODO
        :param layer_size: TODO
        :param kernel_size: TODO, defaults to (3, 3)
        :param asymmetric_padding: TODO, defaults to False
        """
        super(EncodingConvLayer, self).__init__()
        self.asymmetric_padding = asymmetric_padding

        self.padder = nn.ConstantPad2d(padding=(0, 0, 1, 0), value=0)

        self.layers = nn.Sequential(
            nn.Conv2d(input_size, layer_size, kernel_size, padding=1),
            nn.BatchNorm2d(layer_size),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
        )

    def forward(self, x):
        """TODO

        :param x: TODO
        :return: TODO
        """
        if self.asymmetric_padding:
            x = self.padder(x)
        return self.layers(x)


class DecodingConvLayer(nn.Module):
    """This is an class for module that contains conv, batch norm, elu and upsampling."""

    def __init__(self, input_size, layer_size, kernel_size=(3, 3), asymmetric_cropping=False):
        """TODO

        :param input_size: TODO
        :param layer_size: TODO
        :param kernel_size: TODO, defaults to (3, 3)
        :param asymmetric_cropping: TODO, defaults to False
        """
        super(DecodingConvLayer, self).__init__()
        self.asymmetric_cropping = asymmetric_cropping

        self.layers = nn.Sequential(
            nn.Conv2d(input_size, layer_size, kernel_size, padding=1),
            nn.BatchNorm2d(layer_size),
            nn.ELU(),
            nn.Upsample(scale_factor=(2, 1)),
        )

        self.cropper = nn.ConstantPad2d(padding=(0, 0, -1, 0), value=0)

    def forward(self, x):
        """TODO

        :param x: TODO
        :return: TODO
        """
        x = self.layers(x)
        if self.asymmetric_cropping:
            x = self.cropper(x)
        return x


######################################################
# Encoder / Decoder ##################################
######################################################


class Encoder(nn.Module):
    """This is an class defining a convolution encoder."""

    def __init__(self, layer_sizes, padding, z_size, data_input_size):
        """TODO

        :param layer_sizes: TODO
        :param padding: TODO
        :param z_size: TODO
        :param data_input_size: TODO
        """
        super().__init__()
        self.layers = []

        # input size is 1 channel (ie: 1 x len(seq) x 28, for one-hot encoded AAs)
        input_sizes = [1] + layer_sizes[:-1]
        for (input_size, layer_size, extra_pad) in zip(input_sizes, layer_sizes, padding):
            self.layers.append(EncodingConvLayer(input_size, layer_size, asymmetric_padding=extra_pad))

        self.layers = nn.ModuleList(self.layers)

        # must be after self.layers is set!
        self.encoded_full_size, self.encoded_flat_size = self.calculate_encoded_sizes(data_input_size)

        self.fc_mu = nn.Linear(self.encoded_flat_size, z_size)
        self.fc_log_var = nn.Linear(self.encoded_flat_size, z_size)

    def calculate_encoded_sizes(self, data_input_size):
        """TODO

        :param data_input_size: TODO
        :return: TODO
        """
        x = torch.ones(1, *data_input_size, requires_grad=False)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x.size()[1:], int(np.prod(x.size()[1:]))

    def forward(self, x):
        """TODO

        :param x: TODO
        :return: TODO
        """
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, self.encoded_flat_size)
        return self.fc_mu(x), self.fc_log_var(x)


class Decoder(nn.Module):
    """This is an class defining a convolution decoder."""

    def __init__(self, layer_sizes, padding, z_size, encoded_flat_size, encoded_full_size):
        """TODO

        :param layer_sizes: TODO
        :param padding: TODO
        :param z_size: TODO
        :param encoded_flat_size: TODO
        :param encoded_full_size: TODO
        """
        super().__init__()
        self.layers = []
        self.encoded_flat_size = encoded_flat_size
        self.encoded_full_size = encoded_full_size

        # first layer is a dense from a sample z0 to encoded_flat
        self.z_out = nn.Linear(z_size, encoded_flat_size)
        self.z_out_elu = nn.ELU()

        # conv out
        input_sizes = [self.encoded_full_size[0]] + layer_sizes[::-1]
        for input_size, layer_size, crop in zip(input_sizes, layer_sizes[::-1], padding[::-1]):
            self.layers.append(DecodingConvLayer(input_size, layer_size, asymmetric_cropping=crop))

        self.layers.append(nn.Conv2d(in_channels=layer_sizes[0], out_channels=1, kernel_size=(3, 3), padding=(1, 1)))

        self.layers.append(nn.Softmax(dim=3))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        """TODO

        :param x: TODO
        :return: TODO
        """
        x = self.z_out(x)
        x = self.z_out_elu(x)
        x = x.view(-1, *self.encoded_full_size)
        for layer in self.layers:
            x = layer(x)
        return x
