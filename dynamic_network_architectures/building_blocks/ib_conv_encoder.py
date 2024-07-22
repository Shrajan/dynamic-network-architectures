import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
import math
from sympy import *

def DoG_IB_2D(x, y, center, gamma, radius):
    """compute weight at location (x, y) in the OOCS kernel with given parameters
        Parameters:
            x , y : position of the current weight
            center : position of the kernel center
            gamma : center to surround ratio
            radius : center radius

        Returns:
            excite and inhibit : calculated from Equation2 in the paper, without the coefficients A-c and A-s

    """
    # compute sigma from radius of the center and gamma(center to surround ratio)
    sigma = (radius / (2 * gamma)) * (math.sqrt((1 - gamma ** 2) / (-math.log(gamma))))
    excite = (1 / (gamma ** 2)) * math.exp(-1 * ((x - center) ** 2 + (y - center) ** 2) / (2 * ((gamma * sigma) ** 2)))
    inhibit = math.exp(-1 * ((x - center) ** 2 + (y - center) ** 2) / (2 * (sigma ** 2)))

    return excite , inhibit


def IB_filters_2D(radius, gamma, in_channels, out_channels, off=False):
    """compute the kernel filters with given shape and parameters
        Parameters:
            gamma : center to surround ratio
            radius : center radius
            in_channels and out_channels: filter dimensions
            off(boolean) : if false, calculates on center kernel, and if true, off center

        Returns:
            kernel : On or Off center conv filters with requested shape

    """

    # size of the kernel
    kernel_size = int((radius / gamma) * 2 - 1)
    # center node index
    centerX = int((kernel_size + 1) / 2)

    posExcite = 0
    posInhibit = 0
    negExcite = 0
    negInhibit = 0

    for i in range(kernel_size):
        for j in range(kernel_size):
            excite, inhibit = DoG_IB_2D(i + 1, j + 1, centerX, gamma, radius)
            if excite > inhibit:
                posExcite += excite
                posInhibit += inhibit
            else:
                negExcite += excite
                negInhibit += inhibit

    # Calculating A-c and A-s, with requiring the positive values sum up to 1 and negative values to -1
    x, y = symbols('x y')
    sum = 3.
    if kernel_size == 3:
        sum = 1.
    elif kernel_size == 5:
        sum = 3.
    solution = solve((x * posExcite + y * posInhibit - sum, negExcite * x + negInhibit * y + sum), x, y)
    A_c, A_s = float(solution[x].evalf()), float(solution[y].evalf())

    # making the On-center and Off-center conv filters
    kernel = torch.zeros([out_channels, in_channels, kernel_size, kernel_size], requires_grad=False)
    kernel_2D = torch.zeros([kernel_size, kernel_size])

    for i in range(kernel_size):
        for j in range(kernel_size):
            excite, inhibit = DoG_IB_2D(i + 1, j + 1, centerX, gamma, radius)
            weight = excite * A_c + inhibit * A_s
            if off:
                weight *= -1.
            kernel_2D[i][j] = weight

    # Creating all the necessary kernels based on the input and output channels.
    for i in range(out_channels):
        for j in range(in_channels):
            kernel[i][j] = kernel_2D

    return kernel.float()

def DoG_IB_3D(x, y, z, center, gamma, radius):
    """compute weight at location (x, y, z) in the OOCS kernel with given parameters
        Parameters:
            x , y, z : position of the current weight
            center : position of the kernel center
            gamma : center to surround ratio
            radius : center radius

        Returns:
            excite and inhibit : calculated from Equation2 in the paper, without the coefficients A-c and A-s

    """
    # compute sigma from radius of the center and gamma(center to surround ratio)
    sigma = (radius / gamma) * (math.sqrt((1 - gamma ** 2) / (-6 * math.log(gamma))))
    excite = (1 / (gamma ** 3)) * math.exp(
        -1 * ((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) / (2 * ((gamma * sigma) ** 2)))
    inhibit = math.exp(-1 * ((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) / (2 * (sigma ** 2)))

    return excite, inhibit


def IB_filters_3D(radius, gamma, in_channels, out_channels, off=False):
    """compute the kernel filters with given shape and parameters
        Parameters:
            gamma : center to surround ratio
            radius : center radius
            in_channels and out_channels: filter dimensions
            off(boolean) : if false, calculates on center kernel, and if true, off center

        Returns:
            kernel : On or Off center conv filters with requested shape

    """

    # size of the kernel
    kernel_size = int((radius / gamma) * 2 - 1)
    # center node index
    centerX = int((kernel_size + 1) / 2)

    posExcite = 0
    posInhibit = 0
    negExcite = 0
    negInhibit = 0

    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                excite, inhibit = DoG_IB_3D(i + 1, j + 1, k + 1, centerX, gamma, radius)
                if excite > inhibit:
                    posExcite += excite
                    posInhibit += inhibit
                else:
                    negExcite += excite
                    negInhibit += inhibit

    # Calculating A-c and A-s, with requiring the positive vlaues sum up to 1 and negative vlaues to -1
    x, y = symbols('x y')
    sum = 3.
    if kernel_size == 3:
        sum = 1.
    elif kernel_size == 5:
        sum = 3.
    elif kernel_size == 7:
        sum = 5.
    elif kernel_size == 9:
        sum = 7.
    solution = solve((x * posExcite + y * posInhibit - sum, negExcite * x + negInhibit * y + sum), x, y)
    A_c, A_s = float(solution[x].evalf()), float(solution[y].evalf())

    # making the On-center and Off-center conv filters
    kernel = torch.zeros([out_channels, in_channels, kernel_size, kernel_size, kernel_size], requires_grad=False)
    kernel_3D = torch.zeros([kernel_size, kernel_size, kernel_size])

    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                excite, inhibit = DoG_IB_3D(i + 1, j + 1, k + 1, centerX, gamma, radius)
                weight = excite * A_c + inhibit * A_s
                if off:
                    weight *= -1.
                kernel_3D[i][j][k] = weight

    # Creating all the necessary kernels based on the input and output channels.
    for i in range(out_channels):
        for j in range(in_channels):
            kernel[i][j] = kernel_3D

    return kernel.float()


class IBConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output

class IBConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'max',
                 ib_stages: Union[int, List[int], str] = 2, 
                 # "ib_stages: cannot be 1 and cannot be more than n_stages. But could be a 
                 # single int value (2 to n_stages), str value 'all' (will apply IB to all 
                 # encoder layers except 1), or a list of stages (for example: [2,3,5], [2,4], etc.)

                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        self.n_stages = n_stages
        self.ib_stages = self.get_ib_stages(ib_stages)
        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            
            # We are counting encoders from 1 to n_stages (instead of 0 to n_stages-1).
            if s+1 in self.ib_stages:
                stage_modules.append(IBConvBlocks(
                    n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                ))

            else:
                stage_modules.append(StackedConvBlocks(
                    n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                ))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output
    
    def get_ib_stages(self, ib_stages):
        possible_ib_stages = [x for x in range(2, self.n_stages+1)]
        if len(ib_stages) == 1:
            if ib_stages[0] == 'all':
                return possible_ib_stages
            elif type(ib_stages[0]) is int and ib_stages[0] in possible_ib_stages:
                return ib_stages[0]
            else:
                raise Exception(f"The provided IB stages are not compatible with the 
                                possible values: single value or combination of these 
                                {possible_ib_stages} or 'all'. You have given
                                {ib_stages}.")
        else:
            for stage in ib_stages:
                if type(stage) is int and stage in possible_ib_stages:
                    pass
                else:
                    raise Exception(f"The provided IB stages are not compatible with the 
                                    possible values: single value or combination of these 
                                    {possible_ib_stages} or 'all'. You have given
                                    {ib_stages}.")

            return ib_stages




