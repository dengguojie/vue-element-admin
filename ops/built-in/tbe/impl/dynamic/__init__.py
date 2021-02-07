# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
dynamic operator
"""
# pylint: disable=W0622
from __future__ import absolute_import as _abs

from .broadcast_to import broadcast_to
from .softsign import softsign
from .softplus_grad import softplus_grad
from .softplus import softplus
from .unsorted_segment_sum import unsorted_segment_sum
from .div_no_nan import div_no_nan
from .gather_nd import gather_nd
from .gather import gather
from .gather_v2 import gather_v2
from .kl_div import kl_div
from .bias import bias
from .bninference_d import bninference_d
from .scatter_nd import scatter_nd
from .scatter_add import scatter_add
from .scatter_update import scatter_update
from .scatter_sub import scatter_sub
from .scatter_max import scatter_max
from .scatter_min import scatter_min
from .scatter_mul import scatter_mul
from .scatter_div import scatter_div
from .equal import equal
from .relu import relu
from .adam_apply_one_assign import adam_apply_one_assign
from .add import add
from .inv import inv
from .gelu import gelu
from .axpy import axpy
from .floor_mod import floor_mod
from .mul import mul
from .invert import invert
from .muls import muls
from .reduce_sum import reduce_sum
from .reduce_sum_d import reduce_sum_d
from .reduce_max_d import reduce_max_d
from .reduce_mean import reduce_mean
from .reduce_min import reduce_min
from .reduce_mean_d import reduce_mean_d
from .resize_nearest_neighbor_v2 import resize_nearest_neighbor_v2
from .conv2d import conv2d
from .conv3d import conv3d
from .dynamic_atomic_addr_clean import dynamic_atomic_addr_clean
from .sparse_apply_ftrl_d import sparse_apply_ftrl_d
from .div import div
from .sqrt import sqrt
from .square import square
from .sparse_apply_proximal_adagrad_d import sparse_apply_proximal_adagrad_d
from .maximum import maximum
from .minimum import minimum
from .accumulate_nv2 import accumulate_nv2
from .adam_apply_one import adam_apply_one
from .add_n import add_n
from .greater_equal import greater_equal
from .less import less
from .less_equal import less_equal
from .floor_div import floor_div
from .gelu_grad import gelu_grad
from .fused_mul_apply_momentum import fused_mul_apply_momentum
from .tile_d import tile_d
from .tile import tile
from .apply_adam_d import apply_adam_d
from .logical_or import logical_or
from .real_div import real_div
from .reciprocal import reciprocal
from .concat_offset import concat_offset
from .neg import neg
from .concat_d import concat_d
from .concat_v2_d import concat_v2_d
from .pack import pack
from .strided_slice import strided_slice
from .slice import slice
from .cast import cast
from .exp import exp
from .leaky_relu_grad import leaky_relu_grad
from .log1p import log1p
from .sigmoid_grad import sigmoid_grad
from .sqrt_grad import sqrt_grad
from .zeros_like import zeros_like
from .conv2d_backprop_input import conv2d_backprop_input
from .conv2d_backprop_filter import conv2d_backprop_filter
from .conv2d_transpose import conv2d_transpose
from .deconvolution import deconvolution
from .mat_mul import mat_mul
from .batch_matmul import batch_matmul
from .batch_matmul_v2 import batch_matmul_v2
from .sub import sub
from .transpose_d import transpose_d
from .transpose import transpose
from .trans_data import trans_data
from .trans_data_rnn import trans_data_rnn
from .unpack import unpack
from .top_k_d import top_k_d
from .top_k_v2_d import top_k_v2_d
from .pad_d import pad_d
from .split_d import split_d
from .split_v import split_v
from .strided_slice_grad import strided_slice_grad
from .fill import fill
from .drop_out_do_mask import drop_out_do_mask
from .tanh import tanh
from .inv_grad import inv_grad
from .sigmoid_cross_entropy_with_logits import sigmoid_cross_entropy_with_logits
from .abs import abs
from .apply_centered_rms_prop_d import apply_centered_rms_prop_d
from .apply_ftrl_d import apply_ftrl_d
from .apply_ftrl_v2_d import apply_ftrl_v2_d
from .apply_momentum_d import apply_momentum_d
from .assign import assign
from .assign_add import assign_add
from .bias_add import bias_add
from .bias_add_grad import bias_add_grad
from .greater import greater
from .l2_loss import l2_loss
from .log import log
from .logical_and import logical_and
from .logical_not import logical_not
from .reduce_all import reduce_all
from .reduce_any import reduce_any
from .reduce_prod import reduce_prod
from .relu6 import relu6
from .relu6_grad import relu6_grad
from .relu_grad import relu_grad
from .select import select
from .max_pool import max_pool
from .sign import sign
from .acosh import acosh
from .adds import adds
from .asin import asin
from .asin_grad import asin_grad
from .ones_like import ones_like
from .ceil import ceil
from .cos import cos
from .cosh import cosh
from .sigmoid import sigmoid
from .sin import sin
from .sinh import sinh
from .tan import tan
from .asinh import asinh
from .asinh_grad import asinh_grad
from .expm1 import expm1
from .floor import floor
from .leaky_relu import leaky_relu
from .mod import mod
from .power import power
from .truncate_div import truncate_div
from .truncate_mod import truncate_mod
from .xdivy import xdivy
from .xlogy import xlogy
from .atan import atan
from .atanh import atanh
from .assign_sub import assign_sub
from .atan_grad import atan_grad
from .tanh_grad import tanh_grad
from .abs_grad import abs_grad
from .acos_grad import acos_grad
from .acosh_grad import acosh_grad
from .nms_with_mask import nms_with_mask
from .mul_no_nan import mul_no_nan
from .reduce_max import reduce_max
from .maximum_grad import maximum_grad
from .minimum_grad import minimum_grad
from .atan2 import atan2
from .reciprocal_grad import reciprocal_grad
from .not_equal import not_equal
from .erf import erf
from .erfc import erfc
from .data_format_dim_map import data_format_dim_map
from .elu import elu
from .bn_infer import bn_infer
from .bn_infer_grad import bn_infer_grad
