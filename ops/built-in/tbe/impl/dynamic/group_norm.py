# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
group_norm
"""

from impl.util.platform_adapter import tik
import tbe.common.platform as tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.platform_adapter import para_check


MAX_INT32 = 2 ** 31 - 1
TILING_NUM = 64
MASK = 64

TILING_MODE0 = 0
TILING_MODE1 = 1
TILING_MODE2 = 2


def support_5hd(shape_x, num_groups):
    """
    check if support 5HD or not
    """
    if len(shape_x) != 4:
        return False

    c = shape_x[1]
    c0 = 16
    if c % c0 != 0:
        return False
    else:
        c1 = c // c0
        if c1 and c1 % num_groups == 0:
            return True
        else:
            return False


# 'pylint: disable=unused-argument,too-many-locals
def op_select_format(x, scale, offset, y, mean, variance, num_groups, data_format="NCHW", epsilon=1e-5,
                     is_training=False, kernel_name="group_norm"):
    """
    op_select format func for dynamic format
    """

    shape_x = x.get("ori_shape")

    soc_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    dtype_list = []
    format_list0 = []
    format_list1 = []

    if soc_version in ["Ascend310B", tbe_platform.ASCEND_310P, "Ascend910B"]:
        dtype_list = ["float16", "float32", "float16", "float32"]
        format_list0 = ["NC1HWC0", "NC1HWC0", "ND", "ND"]
        format_list1 = ["ND", "ND", "ND", "ND"]

        if -1 in shape_x:
            dtype_list = ["float16", "float32"]
            format_list0 = ["ND", "ND"]
            format_list1 = ["ND", "ND"]
        else:
            if not support_5hd(shape_x, num_groups):
                dtype_list = ["float16", "float32"]
                format_list0 = ["ND", "ND"]
                format_list1 = ["ND", "ND"]

    if soc_version == tbe_platform.ASCEND_910:
        dtype_list = ["float16", "float32", "float32"]
        format_list0 = ["NC1HWC0", "NC1HWC0", "ND"]
        format_list1 = ["ND", "ND", "ND"]

        if is_training:
            dtype_list = ["float32", "float32"]
            format_list0 = ["NC1HWC0", "ND"]
            format_list1 = ["ND", "ND"]

        if -1 in shape_x:
            dtype_list = ["float32"]
            format_list0 = ["ND"]
            format_list1 = ["ND"]
        else:
            if not support_5hd(shape_x, num_groups):
                dtype_list = ["float32"]
                format_list0 = ["ND"]
                format_list1 = ["ND"]

    input0 = gen_param(classify="input0", name="x", datatype=",".join(dtype_list), format=",".join(format_list0))
    input1 = gen_param(classify="input1", name="gamma", datatype=",".join(dtype_list), format=",".join(format_list1))
    input2 = gen_param(classify="input2", name="beta", datatype=",".join(dtype_list), format=",".join(format_list1))
    output0 = gen_param(classify="output0", name="y", datatype=",".join(dtype_list), format=",".join(format_list0))
    output1 = gen_param(classify="output1", name="mean", datatype=",".join(dtype_list), format=",".join(format_list1))
    output2 = gen_param(classify="output2", name="variance", datatype=",".join(dtype_list),
                        format=",".join(format_list1))

    param_dynamic_in_json = get_dynamic_param_in_json([input0, input1, input2, output0, output1, output2])
    return param_dynamic_in_json


# 'pylint: disable=unused-argument,unused-variable,too-many-arguments,too-many-locals
def check_supported(x, scale, offset, y, mean, variance, num_groups, data_format="NCHW", epsilon=1e-5,
                    is_training=False, kernel_name="group_norm"):
    """
    check_supported
    """
    format_x = x.get("format")
    shape_x = x.get("shape")
    gamma_shape = scale.get("shape")
    beta_shape = offset.get("shape")
    c0 = 16

    if num_groups <= 0:
        return False, "num_groups must bigger than zero"

    if epsilon < 0:
        return False, "not support eps"

    if shape_x[1] != -1 and gamma_shape[0] != -1 and beta_shape[0] != -1:
        if format_x == "NC1HWC0":
            channel = shape_x[1] * c0
        else:
            channel = shape_x[1]

        if gamma_shape[0] != channel:
            return False, "gamma shape is not equal to channel"

        if beta_shape[0] != channel:
            return False, "beta shape is not equal to channel"

        if channel % num_groups != 0:
            return False, "channel must can be divided by num_groups"

    return True, ""


# 'pylint: disable=unused-argument,too-many-locals
class GroupNorm5HD(object):
    """
    object of GroupNorm5HD
    """
    def __init__(self, x, scale, offset, y, mean, variance, num_groups, data_format="NCHW", epsilon=1e-5,
                 is_training=False, kernel_name="group_norm"):
        self.tik_instance = tik.Tik()
        self.dtype = x.get("dtype")
        self.is_fp16 = self.dtype == "float16"
        self.fp32 = "float32"
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.kernel_name = kernel_name
        self.is_training = is_training

        self.c0 = 16
        self.block_byte_size = 32
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_byte_size = self.get_dtype_size(self.fp32)
        self.data_each_block = self.block_byte_size // self.dtype_byte_size
        self.c_burst = self.c0 // self.data_each_block
        self.max_mask = 64

        self.ub_n = 512
        self.iter_num = 9
        self.scale_n = 512
        self.offset_n = 512
        self.atomic_num = 2 if self.is_fp16 else 1

        self.input_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="input_gm")
        self.scale_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="scale_gm")
        self.offset_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="offset_gm")
        self.output_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="output_gm")
        self.mean_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="mean_gm",
                                                is_atomic_add=True)
        self.var_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="var_gm",
                                               is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor("int32", [TILING_NUM], scope=tik.scope_gm, name="tiling_gm")
        self.tmp_ub = None
        self.tiling_mode = None
        self.elem_num = None
        self.elem_num_fp = None
        self.hw_num = None
        self.group_c = None
        self.loop_m = None
        self.last_m = None
        self.loop_w = None
        self.last_w = None
        self.avg_ng = None
        self.block_num = None
        self.last_ng = None
        self.shape_c = None
        self.group_hw = None
        self.hw = None
        self.back_m = None
        self.back_w = None

    def get_tiling_params(self):
        """
        get runtime params from tiling
        :return: None
        """
        self.tiling_mode = self.tik_instance.Scalar("int32")
        self.elem_num = self.tik_instance.Scalar("int32")
        self.hw_num = self.tik_instance.Scalar("int32")
        self.group_c = self.tik_instance.Scalar("int32")
        self.loop_m = self.tik_instance.Scalar("int32")
        self.last_m = self.tik_instance.Scalar("int32")
        self.loop_w = self.tik_instance.Scalar("int32")
        self.last_w = self.tik_instance.Scalar("int32")
        self.avg_ng = self.tik_instance.Scalar("int32")
        self.block_num = self.tik_instance.Scalar("int32")
        self.last_ng = self.tik_instance.Scalar("int32")
        self.shape_c = self.tik_instance.Scalar("int32")
        self.group_hw = self.tik_instance.Scalar("int32")
        self.hw = self.tik_instance.Scalar("int32")
        self.elem_num_fp = self.tik_instance.Scalar("float32")

        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int32", shape=(TILING_NUM,), scope=tik.scope_ubuf, name="tiling_ub")
            self.data_move(tiling_ub, self.tiling_gm, num=TILING_NUM)

            self.tiling_mode.set_as(tiling_ub[0])
            self.elem_num.set_as(tiling_ub[1])
            self.hw_num.set_as(tiling_ub[2])
            self.group_c.set_as(tiling_ub[3])
            self.loop_m.set_as(tiling_ub[4])
            self.last_m.set_as(tiling_ub[5])
            self.loop_w.set_as(tiling_ub[6])
            self.last_w.set_as(tiling_ub[7])
            self.avg_ng.set_as(tiling_ub[8])
            self.block_num.set_as(tiling_ub[9])
            self.last_ng.set_as(tiling_ub[10])
            self.shape_c.set_as(tiling_ub[11])
            self.group_hw.set_as(tiling_ub[12])
            self.hw.set_as(tiling_ub[13])
            self.elem_num_fp.set_as(self.elem_num)

    def get_dtype_size(self, dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2, "int64": 8}
        return dtype_dict.get(dtype)

    def compute(self):
        """
        main compute func
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_idx:
            self.get_tiling_params()
            ng_num = self.tik_instance.Scalar("int32")
            self.tmp_ub = self.tik_instance.Tensor("float16", [self.ub_n, self.c0], scope=tik.scope_ubuf,
                                                   name="conv_ub")
            with self.tik_instance.if_scope(block_idx < self.block_num):
                with self.tik_instance.if_scope(block_idx < self.block_num - 1):
                    ng_num.set_as(self.avg_ng)
                with self.tik_instance.if_scope(block_idx == self.block_num - 1):
                    ng_num.set_as(self.last_ng)

                self.compute_per_core(block_idx, ng_num)

        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num, "num_groups": self.num_groups})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        outputs = [self.output_gm, self.mean_gm, self.var_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm, self.scale_gm, self.offset_gm],
                                   outputs=outputs,
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    def compute_per_core(self, block_idx, ng_num):
        """
        compute per ai_core
        """
        loop_ub = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="loop_ub")
        self.normalize_input(block_idx, ng_num, loop_ub)

    def normalize_input(self, block_idx, ng_num, loop_ub):
        """
        normalization
        """
        ng_idx = self.tik_instance.Scalar("int32")
        g_idx = self.tik_instance.Scalar("int32")
        move_offset = self.tik_instance.Scalar("int32")
        offset = self.tik_instance.Scalar("int32")
        mean_scalar = self.tik_instance.Scalar(self.fp32)
        var_scalar = self.tik_instance.Scalar(self.fp32)
        mean_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="mean_ub")
        var_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="var_ub")
        sum_ub0 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub0")
        sum_ub1 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub1")
        scale_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                            name="scale_ub")
        offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                             name="offset_ub")

        with self.tik_instance.if_scope(self.tiling_mode == 0):
            self.data_move(scale_ub, self.scale_gm, num=self.shape_c, need_conv=True,
                           conv_shape=[self.scale_n, self.c0])
            self.data_move(offset_ub, self.offset_gm, num=self.shape_c, need_conv=True,
                           conv_shape=[self.offset_n, self.c0])

            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                g_idx.set_as(ng_idx % self.num_groups)

                self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                  move_offset, ng_idx, False)
                with self.tik_instance.for_range(0, self.group_c) as group_idx:
                    offset.set_as((g_idx * self.group_c + group_idx) * self.c0)
                    self.calc_out(loop_ub, scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                                  mean_scalar, var_scalar)

        with self.tik_instance.elif_scope(self.tiling_mode == 1):
            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                g_idx.set_as(ng_idx % self.num_groups)
                offset.set_as(g_idx * self.group_c * self.c0)
                self.data_move(scale_ub, self.scale_gm[offset], num=self.group_c * self.c0, need_conv=True,
                               conv_shape=[self.scale_n, self.c0])
                self.data_move(offset_ub, self.offset_gm[offset], num=self.group_c * self.c0, need_conv=True,
                               conv_shape=[self.offset_n, self.c0])

                self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                  move_offset, ng_idx, False)
                with self.tik_instance.for_range(0, self.group_c) as group_idx:
                    offset.set_as(group_idx * self.c0)
                    self.calc_out(loop_ub, scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                                  mean_scalar, var_scalar)

        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                g_idx.set_as(ng_idx % self.num_groups)

                self.get_mean_var(loop_ub,  sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                  move_offset, ng_idx, False)
                with self.tik_instance.for_range(0, self.group_c) as group_idx:
                    offset.set_as((g_idx * self.group_c + group_idx) * self.c0)
                    self.data_move(scale_ub, self.scale_gm[offset], num=self.c0, need_conv=True,
                                   conv_shape=[self.scale_n, self.c0])
                    self.data_move(offset_ub, self.offset_gm[offset], num=self.c0, need_conv=True,
                                   conv_shape=[self.offset_n, self.c0])
                    offset.set_as(0)
                    self.calc_out(loop_ub, scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx,
                                  mean_scalar, var_scalar)

    def calc_out(self, loop_ub, scale_ub, offset_ub, offset, move_offset, ng_idx, group_idx, mean_scalar, var_scalar):
        """
        sub mean and divide variance, then mul with scale and add offset
        """
        with self.tik_instance.for_range(0, self.loop_w) as w_idx:
            move_offset.set_as(ng_idx * self.elem_num + group_idx * self.hw_num +
                               w_idx * self.ub_n * self.c0)
            with self.tik_instance.if_scope(w_idx != self.loop_w - 1):
                self.data_move(loop_ub, self.input_gm[move_offset], num=self.ub_n * self.c0, need_conv=True,
                               conv_shape=[self.ub_n, self.c0])
                self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.ub_n * self.c0)
                self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.ub_n * self.c0)
                self.mul_add(loop_ub, scale_ub[offset], offset_ub[offset], num=self.ub_n * self.c0)
                self.data_move(self.output_gm[move_offset], loop_ub, num=self.ub_n * self.c0,
                               need_conv=True, conv_shape=[self.ub_n, self.c0], out=True)

            with self.tik_instance.else_scope():
                self.data_move(loop_ub, self.input_gm[move_offset], num=self.last_w * self.c0, need_conv=True,
                               conv_shape=[self.ub_n, self.c0])
                self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.last_w * self.c0)
                self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.last_w * self.c0)
                self.mul_add(loop_ub, scale_ub[offset], offset_ub[offset], num=self.last_w * self.c0)
                self.data_move(self.output_gm[move_offset], loop_ub, num=self.last_w * self.c0,
                               need_conv=True, conv_shape=[self.ub_n, self.c0], out=True)

    def get_mean_var(self, loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                     move_offset, ng_idx, is_nd):
        self.dup_value(mean_sum, self.c0, 0)
        self.dup_value(var_sum, self.c0, 0)
        self.dup_value(sum_ub0, self.c0, 0)
        self.dup_value(sum_ub1, self.c0, 0)

        work_tensor = self.tik_instance.Tensor(self.fp32, [256], scope=tik.scope_ubuf, name="work_ub")
        tmp_var = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="tmp_ub")
        loop_ub2 = self.tik_instance.Tensor(self.fp32, [self.ub_n, self.c0], scope=tik.scope_ubuf, name="loop_ub2")
        self.dup_value(loop_ub2, self.ub_n * self.c0, 0)

        with self.tik_instance.for_range(0, self.loop_m) as m_idx:
            move_offset.set_as(ng_idx * self.elem_num + m_idx * self.ub_n * self.c0)

            with self.tik_instance.if_scope(m_idx != self.loop_m - 1):
                self.data_move(loop_ub, self.input_gm[move_offset], num=self.ub_n * self.c0, need_conv=True,
                               conv_shape=[self.ub_n, self.c0])
                self.data_mul(loop_ub2, loop_ub, loop_ub, [0, 0, 0], num=self.ub_n * self.c0)
            with self.tik_instance.else_scope():
                self.dup_value(loop_ub, self.ub_n * self.c0, 0)
                self.dup_value(loop_ub2, self.ub_n * self.c0, 0)
                self.data_move(loop_ub, self.input_gm[move_offset], num=self.last_m * self.c0, need_conv=True,
                               conv_shape=[self.ub_n, self.c0])
                if is_nd:
                    self.back_zero(loop_ub, self.back_m, self.last_m * self.c0)
                self.data_mul(loop_ub2, loop_ub, loop_ub, [0, 0, 0], num=self.last_m * self.c0)

            self.data_sum(loop_ub, self.ub_n * self.c0, self.iter_num)
            self.tik_instance.vcadd(16, sum_ub0, loop_ub, 1, 1, 1, 1)
            self.tik_instance.vec_add(16, mean_sum, mean_sum, sum_ub0, 1, 1, 1, 1)

            self.data_sum(loop_ub2, self.ub_n * self.c0, self.iter_num)
            self.tik_instance.vcadd(16, sum_ub1, loop_ub2, 1, 1, 1, 1)
            self.tik_instance.vec_add(16, var_sum, var_sum, sum_ub1, 1, 1, 1, 1)

        self.tik_instance.vec_muls(16, mean_sum, mean_sum, 1 / self.elem_num_fp, 1, 1, 1)
        if self.is_training:
            self.tik_instance.set_atomic_add(self.atomic_num)
            self.data_move(self.mean_gm[ng_idx], mean_sum, self.c0, need_conv=True, conv_shape=[self.c0], out=True)
            self.tik_instance.set_atomic_add(0)
        mean_scalar.set_as(mean_sum[0])
        mean_scalar.set_as(-1 * mean_scalar)

        self.tik_instance.vec_muls(16, var_sum, var_sum, 1 / self.elem_num_fp, 1, 1, 1)
        self.tik_instance.vec_mul(16, mean_sum, mean_sum, mean_sum, 1, 1, 1, 1)
        self.tik_instance.vec_muls(16, mean_sum, mean_sum, -1, 1, 1, 1)
        self.data_add(var_sum, var_sum, mean_sum, [0, 0, 0], num=self.c0)
        if self.is_training:
            self.tik_instance.set_atomic_add(self.atomic_num)
            self.data_move(self.var_gm[ng_idx], var_sum, self.c0, need_conv=True, conv_shape=[self.c0], out=True)
            self.tik_instance.set_atomic_add(0)
        self.tik_instance.vec_adds(16, var_sum, var_sum, self.epsilon, 1, 1, 1)
        self.tik_instance.vec_rsqrt_high_preci(16, tmp_var, var_sum, work_tensor, 1, 1, 1)
        var_scalar.set_as(tmp_var[0])

    def data_move(self, dst, src, num, src_stride=0, dst_stride=0, need_conv=False, conv_shape=None, out=False):
        """
        move data
        """
        sid = 0
        nburst = 1
        if self.is_fp16 and need_conv:
            dtype_byte_size = self.get_dtype_size("float16")
            data_each_block = self.block_byte_size // dtype_byte_size
            burst_len = (num + data_each_block - 1) // data_each_block
            if not out:
                self.tik_instance.data_move(self.tmp_ub, src, sid, nburst, burst_len, src_stride=src_stride,
                                            dst_stride=dst_stride)
                self.data_conv(dst, self.tmp_ub, [0, 0], mode="", num=num, dst_stride=8, src_stride=4)
            else:
                self.data_conv(self.tmp_ub, src, [0, 0], mode="", num=num, dst_stride=4, src_stride=8)
                self.tik_instance.data_move(dst, self.tmp_ub, sid, nburst, burst_len, src_stride=src_stride,
                                            dst_stride=dst_stride)
        else:
            dtype_byte_size = self.get_dtype_size(dst.dtype)
            data_each_block = self.block_byte_size // dtype_byte_size
            burst_len = (num + data_each_block - 1) // data_each_block
            self.tik_instance.data_move(dst, src, sid, nburst, burst_len, src_stride=src_stride, dst_stride=dst_stride)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        offset = self.tik_instance.Scalar("int32", init_value=offset)
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        stride = mask // self.data_each_block

        loop = num // (mask * 255)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
            offset.set_as(offset + loop * mask * 255)

        repeat_time = (num % (mask * 255)) // mask
        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
            offset.set_as(offset + repeat_time * mask)

        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)

    def data_sum(self, src, num, iter_num):
        """
        sum data
        """
        for _ in range(iter_num):
            num = num // 2
            if num // self.max_mask > 0:
                mask = self.max_mask
                repeat_time = num // self.max_mask
            else:
                mask = num
                repeat_time = 1

            src_stride = mask // self.data_each_block
            self.tik_instance.vec_add(mask, src, src[num], src, repeat_time, 0, src_stride, 0)

    def mul_add(self, loop_ub, scale_ub, offset_ub, num):
        """
        mul and add
        """
        mask = 16
        loop = num // (16 * 255)
        stride = mask // self.data_each_block

        offset = self.tik_instance.Scalar("int32", init_value=0)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = index * mask * 255
                self.tik_instance.vec_mul(mask, loop_ub[tmp_offset], loop_ub[tmp_offset], scale_ub, 255,
                                          stride, stride, 0)
                self.tik_instance.vec_add(mask, loop_ub[tmp_offset], loop_ub[tmp_offset], offset_ub, 255,
                                          stride, stride, 0)

            offset.set_as(loop * mask * 255)

        repeat_time = (num % (mask * 255)) // mask

        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_mul(mask, loop_ub[offset], loop_ub[offset], scale_ub, repeat_time,
                                      stride, stride, 0)
            self.tik_instance.vec_add(mask, loop_ub[offset], loop_ub[offset], offset_ub, repeat_time,
                                      stride, stride, 0)
            offset.set_as(offset + repeat_time * mask)

        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_mul(last_num, loop_ub[offset], loop_ub[offset], scale_ub, 1,
                                      stride, stride, 0)
            self.tik_instance.vec_add(last_num, loop_ub[offset], loop_ub[offset], offset_ub, 1,
                                      stride, stride, 0)

    def back_zero(self, loop_ub, back_num, ub_num):
        """
        when format is ND, need set some zero
        """
        with self.tik_instance.if_scope(back_num > 0):
            with self.tik_instance.for_range(0, back_num) as idx:
                loop_ub[ub_num - 1 - idx].set_as(0)

    def single_operator_template(self, op_obj, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        dst_offset = self.tik_instance.Scalar("int32", init_value=offsets[0])
        src_offset = self.tik_instance.Scalar("int32", init_value=offsets[1])
        vector_mask_max = 256 // self.dtype_byte_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], scalar, 255,
                       dst_stride, src_stride)

            dst_offset.set_as(dst_offset + loop * vector_mask_max * 255)
            src_offset.set_as(src_offset + loop * vector_mask_max * 255)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_stride, src_stride)
            dst_offset.set_as(dst_offset + repeat_time * vector_mask_max)
            src_offset.set_as(src_offset + repeat_time * vector_mask_max)

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset], src[src_offset], scalar, 1, dst_stride, src_stride)

    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8,
                                 src1_stride=8):
        """
        tik api template
        """
        dst_offset = self.tik_instance.Scalar("int32", init_value=offsets[0])
        src0_offset = self.tik_instance.Scalar("int32", init_value=offsets[1])
        src1_offset = self.tik_instance.Scalar("int32", init_value=offsets[2])
        vector_mask_max = 256 // self.dtype_byte_size

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src0_offset = src0_offset + index * vector_mask_max * 255
                tmp_src1_offset = src1_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src0[tmp_src0_offset], src1[tmp_src1_offset], 255,
                       dst_stride, src0_stride, src1_stride)

            dst_offset.set_as(dst_offset + loop * vector_mask_max * 255)
            src0_offset.set_as(src0_offset + loop * vector_mask_max * 255)
            src1_offset.set_as(src1_offset + loop * vector_mask_max * 255)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            op_obj(vector_mask_max, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time, dst_stride,
                   src0_stride, src1_stride)
            dst_offset.set_as(dst_offset + repeat_time * vector_mask_max)
            src0_offset.set_as(src0_offset + repeat_time * vector_mask_max)
            src1_offset.set_as(src1_offset + repeat_time * vector_mask_max)

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            op_obj(last_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1, dst_stride, src0_stride,
                   src1_stride)

    def data_conv(self, dst, src, offsets, mode="ceil", num=0, dst_stride=8, src_stride=8):
        """
        tik conv
        """
        dst_offset = self.tik_instance.Scalar("int32", init_value=offsets[0])
        src_offset = self.tik_instance.Scalar("int32", init_value=offsets[1])
        vector_mask_max = 64

        tensor_size = num
        loop = tensor_size // (vector_mask_max * 255)

        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                self.tik_instance.vec_conv(vector_mask_max, mode, dst[tmp_dst_offset], src[tmp_src_offset], 255,
                                           dst_stride, src_stride)

            dst_offset.set_as(dst_offset + loop * vector_mask_max * 255)
            src_offset.set_as(src_offset + loop * vector_mask_max * 255)

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_conv(vector_mask_max, mode, dst[dst_offset], src[src_offset], repeat_time,
                                       dst_stride, src_stride)
            dst_offset.set_as(dst_offset + repeat_time * vector_mask_max)
            src_offset.set_as(src_offset + repeat_time * vector_mask_max)

        last_num = tensor_size % vector_mask_max
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_conv(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)

    def data_adds(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik adds
        """
        self.single_operator_template(self.tik_instance.vec_adds, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)

    def data_add(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik add
        """
        self.double_operator_template(self.tik_instance.vec_add, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_mul(self, dst, src0, src1, offsets, num=0, dst_stride=8, src0_stride=8, src1_stride=8):
        """
        tik mul
        """
        self.double_operator_template(self.tik_instance.vec_mul, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_muls(self, dst, src, scalar, offsets, num=0, dst_stride=8, src_stride=8):
        """
        tik muls
        """
        self.single_operator_template(self.tik_instance.vec_muls, dst, src, scalar, offsets, num, dst_stride,
                                      src_stride)


class GroupNormND(GroupNorm5HD):
    """
    object of GroupNorm when format is ND
    """
    def __init__(self, x, scale, offset, y, mean, variance, num_groups, data_format="NCHW", epsilon=1e-5,
                 is_training=False, kernel_name="group_norm"):
        super(GroupNormND, self).__init__(x, scale, offset, y, mean, variance, num_groups, data_format, epsilon,
                                          is_training, kernel_name)
        self.input_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="input_gm_nd")
        self.scale_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="scale_gm_nd")
        self.offset_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="offset_gm_nd")
        self.output_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="output_gm_nd",
                                                  is_atomic_add=True)
        self.mean_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="mean_gm_nd",
                                                is_atomic_add=True)
        self.var_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], scope=tik.scope_gm, name="var_gm_nd",
                                               is_atomic_add=True)
        self.tiling_gm = self.tik_instance.Tensor("int32", [TILING_NUM], scope=tik.scope_gm, name="tiling_gm_nd")

    def compute(self):
        """
        main compute func
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_idx:
            self.get_tiling_params()
            with self.tik_instance.if_scope(block_idx < self.block_num):
                self.compute_back()
                ng_num = self.tik_instance.Scalar("int32")
                self.tmp_ub = self.tik_instance.Tensor("float16", [self.ub_n, self.c0], scope=tik.scope_ubuf,
                                                       name="conv_ub")
                with self.tik_instance.if_scope(block_idx < self.block_num - 1):
                    ng_num.set_as(self.avg_ng)
                with self.tik_instance.if_scope(block_idx == self.block_num - 1):
                    ng_num.set_as(self.last_ng)

                self.compute_per_core(block_idx, ng_num)

        outputs = [self.output_gm, self.mean_gm, self.var_gm]
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num, "num_groups": self.num_groups})
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_gm, self.scale_gm, self.offset_gm],
                                   outputs=outputs,
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        return self.tik_instance

    def normalize_input(self, block_idx, ng_num, loop_ub):
        """
        normalize x
        """
        ng_idx = self.tik_instance.Scalar("int32")
        g_idx = self.tik_instance.Scalar("int32")
        move_offset = self.tik_instance.Scalar("int32")
        offset = self.tik_instance.Scalar("int32")
        scale_scalar = self.tik_instance.Scalar(self.fp32)
        offset_scalar = self.tik_instance.Scalar(self.fp32)
        mean_scalar = self.tik_instance.Scalar(self.fp32)
        var_scalar = self.tik_instance.Scalar(self.fp32)
        mean_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="mean_ub")
        var_sum = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="var_ub")
        sum_ub0 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub0")
        sum_ub1 = self.tik_instance.Tensor(self.fp32, [self.c0], scope=tik.scope_ubuf, name="sum_ub1")
        scale_ub = self.tik_instance.Tensor(self.fp32, [self.scale_n, self.c0], scope=tik.scope_ubuf,
                                            name="scale_ub")
        offset_ub = self.tik_instance.Tensor(self.fp32, [self.offset_n, self.c0], scope=tik.scope_ubuf,
                                             name="offset_ub")

        with self.tik_instance.if_scope(self.tiling_mode == 0):
            self.data_move(scale_ub, self.scale_gm, num=self.shape_c, need_conv=True,
                           conv_shape=[self.scale_n, self.c0])
            self.data_move(offset_ub, self.offset_gm, num=self.shape_c, need_conv=True,
                           conv_shape=[self.offset_n, self.c0])

            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                g_idx.set_as(ng_idx % self.num_groups)

                self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                  move_offset, ng_idx, True)
                with self.tik_instance.for_range(0, self.group_c) as group_idx:
                    offset.set_as(g_idx * self.group_c + group_idx)
                    scale_scalar.set_as(scale_ub[offset])
                    offset_scalar.set_as(offset_ub[offset])
                    self.calc_out_nd(loop_ub, scale_scalar, offset_scalar, move_offset, ng_idx, group_idx, mean_scalar,
                                     var_scalar)

        with self.tik_instance.elif_scope(self.tiling_mode == 1):
            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                g_idx.set_as(ng_idx % self.num_groups)
                offset.set_as(g_idx * self.group_c)
                self.data_move(scale_ub, self.scale_gm[offset], num=self.group_c, need_conv=True,
                               conv_shape=[self.scale_n, self.c0])
                self.data_move(offset_ub, self.offset_gm[offset], num=self.group_c, need_conv=True,
                               conv_shape=[self.offset_n, self.c0])

                self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                  move_offset, ng_idx, True)
                with self.tik_instance.for_range(0, self.group_c) as group_idx:
                    scale_scalar.set_as(scale_ub[group_idx])
                    offset_scalar.set_as(offset_ub[group_idx])
                    self.calc_out_nd(loop_ub, scale_scalar, offset_scalar, move_offset, ng_idx, group_idx, mean_scalar,
                                     var_scalar)

        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, ng_num) as n_idx:
                ng_idx.set_as(block_idx * self.avg_ng + n_idx)
                g_idx.set_as(ng_idx % self.num_groups)

                self.get_mean_var(loop_ub, sum_ub0, sum_ub1, mean_sum, var_sum, var_scalar, mean_scalar,
                                  move_offset, ng_idx, True)
                with self.tik_instance.for_range(0, self.group_c) as group_idx:
                    offset.set_as(g_idx * self.group_c + group_idx)
                    self.data_move(scale_ub, self.scale_gm[offset], num=self.c0, need_conv=True,
                                   conv_shape=[self.scale_n, self.c0])
                    self.data_move(offset_ub, self.offset_gm[offset], num=self.c0, need_conv=True,
                                   conv_shape=[self.offset_n, self.c0])
                    scale_scalar.set_as(scale_ub[0])
                    offset_scalar.set_as(offset_ub[0])
                    self.calc_out_nd(loop_ub, scale_scalar, offset_scalar, move_offset, ng_idx, group_idx, mean_scalar,
                                     var_scalar)

    def compute_back(self):
        """
        compute number of back
        """
        self.back_m = self.tik_instance.Scalar("int32")
        self.back_w = self.tik_instance.Scalar("int32")
        self.back_m.set_as(self.group_hw * self.c0 - self.elem_num)
        self.back_w.set_as(self.hw * self.c0 - self.hw_num)

    def calc_out_nd(self, loop_ub, scale_scalar, offset_scalar, move_offset, ng_idx, group_idx,
                    mean_scalar, var_scalar):
        """
        calculate output when format is ND
        """
        self.tik_instance.set_atomic_add(self.atomic_num)
        with self.tik_instance.for_range(0, self.loop_w) as w_idx:
            move_offset.set_as(ng_idx * self.elem_num + group_idx * self.hw_num +
                               w_idx * self.ub_n * self.c0)
            with self.tik_instance.if_scope(w_idx != self.loop_w - 1):
                self.data_move(loop_ub, self.input_gm[move_offset], num=self.ub_n * self.c0, need_conv=True,
                               conv_shape=[self.ub_n, self.c0])
                self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.ub_n * self.c0)
                self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.ub_n * self.c0)
                self.mul_add_nd(loop_ub, scale_scalar, offset_scalar, num=self.ub_n * self.c0)
                self.data_move(self.output_gm[move_offset], loop_ub, num=self.ub_n * self.c0, need_conv=True,
                               conv_shape=[self.ub_n, self.c0], out=True)
            with self.tik_instance.else_scope():
                self.dup_value(loop_ub, self.ub_n * self.c0, 0)
                self.data_move(loop_ub, self.input_gm[move_offset], num=self.last_w * self.c0, need_conv=True,
                               conv_shape=[self.last_w, self.c0])
                self.data_adds(loop_ub, loop_ub, mean_scalar, [0, 0], num=self.last_w * self.c0)
                self.data_muls(loop_ub, loop_ub, var_scalar, [0, 0], num=self.last_w * self.c0)
                self.mul_add_nd(loop_ub, scale_scalar, offset_scalar, num=self.last_w * self.c0)
                self.back_zero(loop_ub, self.back_w, self.last_w * self.c0)
                self.data_move(self.output_gm[move_offset], loop_ub, num=self.last_w * self.c0, need_conv=True,
                               conv_shape=[self.last_w, self.c0], out=True)
        self.tik_instance.set_atomic_add(0)

    def mul_add_nd(self, loop_ub, scale_scalar, offset_scalar, num):
        """
        mul and add
        """
        self.data_muls(loop_ub, loop_ub, scale_scalar, [0, 0], num)
        self.data_adds(loop_ub, loop_ub, offset_scalar, [0, 0], num=num)


def check_params(x, scale, offset):
    """
    check params of GroupNorm
    """
    dtype_x = x.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_offset = offset.get("dtype")

    if dtype_x != dtype_scale or dtype_x != dtype_offset:
        raise RuntimeError("dtype of x, scale, offset must be same")

    if dtype_x not in ("float16", "float32"):
        raise RuntimeError("only support float16 and float32")


# 'pylint: disable=unused-argument,too-many-locals
@register_operator("GroupNorm")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def group_norm(x, scale, offset, y, mean, variance, num_groups, data_format="NCHW", epsilon=1e-5, is_training=False,
               kernel_name="group_norm"):
    """
    :param x: input_data, support ND and 5HD of float16 or float32
    :param scale: scale_factor
    :param offset: offset_factor
    :param y: The result of GroupNorm
    :param mean: mean of x
    :param variance: variance of x
    :param num_groups: number of groups
    :param data_format: data_format, default to NCHW
    :param epsilon: epsilon avoid divided by zero, default to 1e-5
    :param is_training: is_training
    :param kernel_name: kernel_name, default to group_norm
    :return: instance
    """
    check_params(x, scale, offset)
    format_x = x.get("format")
    if format_x == "NC1HWC0":
        instance = GroupNorm5HD(x, scale, offset, y, mean, variance, num_groups, data_format, epsilon, is_training,
                                kernel_name)
    else:
        instance = GroupNormND(x, scale, offset, y, mean, variance, num_groups, data_format, epsilon, is_training,
                               kernel_name)
    return instance.compute()
