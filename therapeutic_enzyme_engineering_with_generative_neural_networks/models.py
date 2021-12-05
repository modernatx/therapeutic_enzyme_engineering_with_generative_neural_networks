"""Code defining VAE models"""
from typing import List

import numpy as np
import torch
from torch import nn


class VAE(nn.Module):
    """This is an generic class for sequence VAEs."""

    def __init__(self, sequence_length: int, layer_sizes: List[int], z_size, alphabet_size=28):
        """This class initializes internal encoder and decoder objects that downsample/
        upsample 2x at every layer.

        :param sequence_length: the length of the sequence alignment
        :param layer_sizes: a list of number of kernels in the encoder and decoder.
        :param z_size: size of the latent space
        :param alphabet_size: number of AAs in the alphabet, defaults to 28 which is the full SeqLike AA alphabet
        """
        super().__init__()

        self.sequence_length = sequence_length

        self.layer_sizes = layer_sizes
        self.z_size = z_size
        self.alphabet_size = alphabet_size
        self.data_input_size = (1, self.sequence_length, self.alphabet_size)

        # build padding amounts
        # We do this because 2x down and up sampling only works for sequences that are
        # a power of 2 in length otherwise.  We add or remove single "columns" so that
        # the length (3rd dim) is always even (batch x 1 x length x alphabet size)
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
        """Return a multivariate sample given means and log variances.  This implements the "reparametrization trick"
        and is passed into the decoder.

        :param mu: a set of means
        :param logvar: a set of logvariances
        :return: a sample
        """
        std = torch.exp(logvar * 0.5)
        eps = torch.normal(mean=torch.zeros_like(std), std=torch.ones_like(std)).to(std.device)
        return (eps * std) + mu

    def forward(self, x):
        """Forward pass of the whole VAE

        :param x: a batch of one-hot encoded sequences (batch x 1 x length x alphabet size)
        :return: a tuple of reconstructed samples, mu and logvar from the encoder, and the specific sample that
                 was sent to the decord.
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
        """This implements a "compound layer" which pads if needed, then computes a convolution
        followed by batch norm, ELU and then 2x pooling in the length dimension.

        Typically not used directly, but only within an encoder

        :param input_size: size of input
        :param layer_size: number of conv kernels
        :param kernel_size: kernel size, defaults to (3, 3)
        :param asymmetric_padding: weather to pad or not, defaults to False
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
        """Forward pass of compound layer.

        :param x: input (batch x 1 x length x alphabet size)
        :return: output (batch x 1 x length/2 x alphabet size)
        """
        if self.asymmetric_padding:
            x = self.padder(x)
        return self.layers(x)


class DecodingConvLayer(nn.Module):
    """This is an class for module that contains conv, batch norm, elu and upsampling."""

    def __init__(self, input_size, layer_size, kernel_size=(3, 3), asymmetric_cropping=False):
        """This implements a "compound layer" which crops if needed, then computes a convolution
        followed by batch norm, ELU and then 2x upsampling in the length dimension.

        Typically not used directly, but only within an encoder

        :param input_size: size of input
        :param layer_size: number of conv kernels
        :param kernel_size: kernel size, defaults to (3, 3)
        :param asymmetric_cropping: weather to crop or not, defaults to False
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
        """Forward pass of compound layer.

        :param x: input (batch x 1 x length x alphabet size)
        :return: output (batch x 1 x length*2 x alphabet size)
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
        """Class for the encoder, which uses multiple EncodingConvLayers.

        :param layer_sizes: the number of kernels in each layer
        :param padding: a list to specify if we need to pad or not before pooling
        :param z_size: size of the latent space
        :param data_input_size: initial input size, used to define fully connected layer
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
        """This method takes an expected data input size and calculates what the size of the input to the fully connect layer will be.

        :param data_input_size: input datasize
        :return: tuple of the full encoded size and the flattened encoded size
        """
        x = torch.ones(1, *data_input_size, requires_grad=False)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x.size()[1:], int(np.prod(x.size()[1:]))

    def forward(self, x):
        """forward pass of the encoder

        :param x: input data (batch x 1 x sequence length x alphabet size)
        :return: mus and logvars of the latent space
        """
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, self.encoded_flat_size)
        return self.fc_mu(x), self.fc_log_var(x)


class Decoder(nn.Module):
    """This is an class defining a convolution decoder."""

    def __init__(self, layer_sizes, padding, z_size, encoded_flat_size, encoded_full_size):
        """Class for the decoder, which uses multiple DecodingConvLayers

        :param layer_sizes: the number of kernels in each layer
        :param padding: a list to specify if we need to crop or not before upsampling
        :param z_size: size of the latent space
        :param encoded_flat_size: size of the flattened encoded sequence
        :param encoded_full_size: size of the full encoded sequence
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
        """forward pass of the encoder

        :param x: sample from the laten space
        :return: reconstructed data (batch x 1 x sequence length x alphabet size)
        """
        x = self.z_out(x)
        x = self.z_out_elu(x)
        x = x.view(-1, *self.encoded_full_size)
        for layer in self.layers:
            x = layer(x)
        return x
