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
cce compute API:
In order to simplify the  procedure of writing schedule,TE provides a set of
TensorEngine APIs.
Using those API to develop operators, you can use the "Auto_schedule" create
schedule.
"""
import warnings

# pylint: disable=redefined-builtin
from .broadcast_compute import broadcast
from .cast_compute import ceil, floor, round, trunc, round_half_up, round_d
from .common import round_to, cast_to, cast_to_round
from .concat_compute import concat
from .conv_compute import conv, ConvParam, check_conv_shape, conv_compress
from .conv_compute import is_support_v200
from .dim_conv import compute_four2five, compute_five2four
from .elewise_compute import vmuls, vadds, vlog, vexp, vabs, vrec, vrelu, vnot, \
    vsqrt, vrsqrt, vdiv, vmul, vadd, vsub, vmin, vmax, vor, vand, vaxpy, vmla, \
    vmadd, \
    vmaddrelu, vmaxs, vmins, vcmp, vlogic, vsel, vcmpsel, vmod, \
    vlrelu, vaddrelu, vsubrelu
from .reduction_compute import sum, reduce_min, reduce_max, reduce_prod
from .segment_compute import unsorted_segment_max, unsorted_segment_min, \
    unsorted_segment_sum, \
    unsorted_segment_mean, unsorted_segment_prod
from .depthwise_conv2d_compute import depthwise_conv2d_backprop_filter_d_compute
from .depthwise_conv2d_compute import depthwise_conv2d_backprop_input_d_compute
from .depthwise_conv2d_compute import depthwise_conv2d_compute, \
    DepthwiseConv2dParam
from .inplace_compute import inplace_add, inplace_sub, inplace_update

warnings.filterwarnings(action='default',
                        message="round_d|cast_to_round|compute_four2five|compute_five2four|"
                                "get_caffe_out_size_and_pad|pooling3d_max_grad_grad|tuple_sum|unsorted_segment_sum|"
                                "unsorted_segment_mean|unsorted_segment_prod|unsorted_segment_min|"
                                "unsorted_segment_max|split_compute_com",
                        category=DeprecationWarning)
