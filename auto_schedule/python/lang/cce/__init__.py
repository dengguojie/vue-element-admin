# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
comput schedule init
"""
from .te_compute.broadcast_compute import broadcast
from .te_compute.cast_compute import ceil, floor, round, trunc
from .te_compute.common import round_to, cast_to, cast_to_round, calculate_one_or_zero
from .te_compute.concat_compute import concat
from .te_compute.conv_compute import conv
from .te_compute.conv_compute import ConvParam
from .te_compute.conv2d_backprop_input_compute import DeconvParam
from .te_compute.conv2d_backprop_input_general_compute import DeConvPattern
from .te_compute.conv2d_backprop_input_opti_compute \
    import DeConvKernelSize1Pattern
from .te_compute.depthwise_conv2d_compute import depthwise_conv2d_backprop_filter_d_compute
from .te_compute.depthwise_conv2d_compute import depthwise_conv2d_backprop_input_d_compute
from .te_compute.depthwise_conv2d_compute import depthwise_conv2d_compute
from .te_compute.depthwise_conv2d_compute import DepthwiseConv2dParam
from .te_compute.conv_compute import check_conv_shape
from .te_compute.conv_compute import conv_compress
from .te_compute.conv_compute import is_support_v200
from .te_compute.max_pool2d_3_2_fusion_compute import MaxPoolParam
from .te_compute.max_pool2d_3_2_fusion_compute import max_pool_compute
from .te_compute.dim_conv import compute_four2five, compute_five2four
from .te_compute.elewise_compute import vmuls, vadds, vlog, vexp, vabs, vrec, \
    vrelu, vnot, vsqrt, vrsqrt, vdiv, vmul, vadd, vsub, vmin, vmax, vor, vand, vaxpy, \
    vmla, vmadd, vmaddrelu, vmaxs, vmins, vcmp, vlogic, vsel, vcmpsel, vmod, \
    vlrelu, vaddrelu, vsubrelu
from .te_compute.reduction_compute import sum, reduce_min, reduce_max, \
    reduce_prod, tuple_sum
from .te_compute.segment_compute import unsorted_segment_max, \
    unsorted_segment_min, unsorted_segment_sum, unsorted_segment_mean, \
    unsorted_segment_prod
from .te_compute import util
from .te_schedule import cce_build_code
from .te_compute.mmad_compute import matmul
from .te_compute.mmad_compute import matmul_cv_split
from .te_compute.mmad_compute import get_matmul_performance_format
from .te_compute.gemm_compute import gemm
from .te_compute.pooling2d_compute import pooling2d
from .te_compute.pooling2d_compute import get_caffe_out_size_and_pad
from .te_compute.pooling3d_compute import pooling3d
from .te_compute.pooling3d_max_grad_grad_compute import pooling3d_max_grad_grad
from .te_compute.conv2d_backprop_filter_compute import conv2d_backprop_filter_compute
from .te_compute.conv2d_backprop_input_compute import conv2d_backprop_input_compute
from .te_compute.conv2d_backprop_input_compute import DynamicConv2dBpInputParams
from .te_compute.split_compute import split_compute_com
from .te_schedule.split_schedule import split_schedule_com
from .te_compute.inplace_compute import inplace_add, inplace_sub, inplace_update
from .te_compute.conv3d_backprop_filter_compute import \
    conv3d_backprop_filter_compute
from .te_compute.util import dsl_check_support

from .te_schedule.conv_schedule import CceConvOp, AutoScheduleOp
from .te_schedule.cce_schedule import reget_tensor_list, get_op_info
from .te_schedule.conv2d_backprop_input_schedule import CceConv2dBackpropInputOp

from .te_schedule.cce_schedule import schedule_cce
from te.utils.cce import auto_schedule, build
