# Copyright 2022 Huawei Technologies Co., Ltd
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
balance_rois
"""

import math
from te import tik
from te.utils import para_check
from impl.util.platform_adapter import tbe_platform
from impl.sort import sort_compute, tune
from impl.sort import BLOCK, NUM_BLOCK
from impl.sort import PROPOSAL_NUM
from impl.sort import functools_reduce


class BalanceRoiByArea(object):
    """
    sort rois to be balanced
    """
    def __init__(self, rois, sort_rois, sort_idx, kernel_name):
        self.tik_instance = tik.Tik()
        self.rois_dtype = rois.get("dtype")
        self.rois_shape = rois.get("shape")
        self.idx_dtype = sort_idx.get("dtype")
        self.batch_n = self.rois_shape[0]
        self.proposal_num = self.rois_shape[1]
        self.idx_shape = [self.batch_n]
        self.block_byte_size = 32
        self.block_num = 1
        self.kernel_name = kernel_name
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        self.loop_num = 2048
        self.loop = math.ceil(self.batch_n / self.loop_num)
        self.tail_num = self.batch_n % self.loop_num

        self.rois_gm = self.tik_instance.Tensor(self.rois_dtype, self.rois_shape, scope=tik.scope_gm, name="rois_gm")
        self.sort_gm = self.tik_instance.Tensor(self.rois_dtype, self.rois_shape, scope=tik.scope_gm, name="sort_gm")
        self.idx_gm = self.tik_instance.Tensor(self.idx_dtype, self.idx_shape, scope=tik.scope_gm, name="idx_gm")
        self.area_gm = self.tik_instance.Tensor(self.rois_dtype, self.idx_shape, scope=tik.scope_gm, name="area_gm",
                                                is_workspace=True)

    def extract_roi(self, roi_ub, x0_ub, y0_ub, x1_ub, y1_ub, loop_idx, num):
        """
        extract roi
        """
        with self.tik_instance.for_range(0, num) as idx:
            x0_ub[loop_idx * self.loop_num + idx].set_as(roi_ub[idx, 1])
            y0_ub[loop_idx * self.loop_num + idx].set_as(roi_ub[idx, 2])
            x1_ub[loop_idx * self.loop_num + idx].set_as(roi_ub[idx, 3])
            y1_ub[loop_idx * self.loop_num + idx].set_as(roi_ub[idx, 4])

    def compute(self):
        """
        compute func
        """
        with self.tik_instance.new_stmt_scope():
            x0_ub = self.tik_instance.Tensor(self.rois_dtype, [self.batch_n], scope=tik.scope_ubuf, name="x0_ub")
            y0_ub = self.tik_instance.Tensor(self.rois_dtype, [self.batch_n], scope=tik.scope_ubuf, name="y0_ub")
            x1_ub = self.tik_instance.Tensor(self.rois_dtype, [self.batch_n], scope=tik.scope_ubuf, name="x1_ub")
            y1_ub = self.tik_instance.Tensor(self.rois_dtype, [self.batch_n], scope=tik.scope_ubuf, name="y1_ub")

            with self.tik_instance.new_stmt_scope():
                roi_ub = self.tik_instance.Tensor(self.rois_dtype, [self.loop_num, self.proposal_num],
                                                  scope=tik.scope_ubuf, name="roi_ub")
                with self.tik_instance.for_range(0, self.loop) as loop_idx:
                    with self.tik_instance.if_scope(loop_idx != self.loop - 1):
                        self.data_move(roi_ub[0, 0], self.rois_gm[loop_idx * self.loop_num, 0],
                                       num=self.loop_num * self.proposal_num)
                        self.extract_roi(roi_ub, x0_ub, y0_ub, x1_ub, y1_ub, loop_idx, self.loop_num)
                    with self.tik_instance.else_scope():
                        self.data_move(roi_ub[0, 0], self.rois_gm[loop_idx * self.loop_num, 0],
                                       num=self.tail_num * self.proposal_num)
                        self.extract_roi(roi_ub, x0_ub, y0_ub, x1_ub, y1_ub, loop_idx, self.tail_num)

            self.data_sub(x1_ub, x1_ub, x0_ub, [0, 0, 0], num=self.batch_n)
            self.data_sub(y1_ub, y1_ub, y0_ub, [0, 0, 0], num=self.batch_n)
            self.data_maxs(x1_ub, x1_ub, 0, [0, 0], self.batch_n)
            self.data_maxs(y1_ub, y1_ub, 0, [0, 0], self.batch_n)
            self.data_sqrt(x1_ub, x1_ub, [0, 0], self.batch_n)
            self.data_sqrt(y1_ub, y1_ub, [0, 0], self.batch_n)
            self.data_mul(x1_ub, x1_ub, y1_ub, [0, 0, 0], num=self.batch_n)

            self.data_move(self.area_gm, x1_ub, num=self.batch_n)

        data_indices = self.sort_area()
        indice_ub = self.tik_instance.Tensor("int32", [self.batch_n], scope=tik.scope_ubuf, name="indice_ub")
        self.data_move(indice_ub, data_indices, num=self.batch_n)
        self.re_rois(indice_ub)

        self.tik_instance.BuildCCE(inputs=[self.rois_gm], outputs=[self.sort_gm, self.idx_gm],
                                   kernel_name=self.kernel_name)

        return self.tik_instance

    def sort_area(self):
        """
        sort rois area
        """
        descending = False
        shape, dtype, num = [self.batch_n], "float16", self.batch_n
        allnum = functools_reduce(lambda x, y: x * y, shape)
        rounds = allnum // num

        num_16 = (num + BLOCK - 1) // BLOCK * BLOCK
        num_2048 = (num + NUM_BLOCK - 1) // NUM_BLOCK * NUM_BLOCK
        num_gm = num_2048 // NUM_BLOCK

        if self.rois_dtype == "float32":
            input_gm = self.tik_instance.Tensor("float16", self.idx_shape, scope=tik.scope_gm, name="input_gm",
                                                is_workspace=True)
            fp32_ub = self.tik_instance.Tensor("float32", self.idx_shape, scope=tik.scope_ubuf, name="fp32_ub")
            fp16_ub = self.tik_instance.Tensor("float16", self.idx_shape, scope=tik.scope_ubuf, name="fp16_ub")

            self.data_move(fp32_ub, self.area_gm, num=self.batch_n)
            self.data_conv(fp16_ub, fp32_ub, [0, 0], mode="", num=self.batch_n, dst_stride=4, src_stride=8)
            self.data_move(input_gm, fp16_ub, num=self.batch_n)
        else:
            input_gm = self.area_gm

        if num <= NUM_BLOCK:
            data_out = self.tik_instance.Tensor(dtype, [rounds * num_16], name="data_out", scope=tik.scope_gm,
                                                is_workspace=True)
            data_indices = self.tik_instance.Tensor("int32", [rounds * num_16], name="data_indices", scope=tik.scope_gm,
                                                    is_workspace=True)
            data_out_ = self.tik_instance.Tensor(dtype, shape, name="data_out_", scope=tik.scope_gm, is_workspace=True)
            data_indices_ = self.tik_instance.Tensor("int32", shape, name="data_indices_", scope=tik.scope_gm,
                                                     is_workspace=True)

        else:
            data_out = self.tik_instance.Tensor(dtype, [rounds * num_2048], name="data_out", scope=tik.scope_gm,
                                                is_workspace=True)
            data_indices = self.tik_instance.Tensor("int32", [rounds * num_2048], name="data_indices",
                                                    scope=tik.scope_gm, is_workspace=True)

            data_out_ = self.tik_instance.Tensor(dtype, shape, name="data_out_", scope=tik.scope_gm, is_workspace=True)
            data_indices_ = self.tik_instance.Tensor("int32", shape, name="data_indices_", scope=tik.scope_gm,
                                                     is_workspace=True)

        cce_product = tbe_platform.get_soc_spec(tbe_platform.SOC_VERSION)
        available_aicore_num = tik.Dprofile().get_aicore_num()
        used_aicore_num = available_aicore_num if rounds > available_aicore_num else rounds

        temp = self.tik_instance.Tensor(dtype, [used_aicore_num * num_2048 * PROPOSAL_NUM], name="temp",
                                        scope=tik.scope_gm, is_workspace=True)

        data_out, data_indices = sort_compute(self.tik_instance, dtype, num, num_16, num_2048, 0,
                                              used_aicore_num, data_out, data_indices, input_gm,
                                              temp, num_gm, descending, cce_product)

        data_out_, data_indices_ = tune(self.tik_instance, num, num_16, num_2048, rounds, num_gm, data_out, data_out_,
                                        data_indices, data_indices_)
        return data_indices_

    def re_rois(self, indice_ub):
        """
        balance rois
        """
        loop = self.batch_n // self.core_num
        tmp_scalar = self.tik_instance.Scalar("int32")
        tmp_roi = self.tik_instance.Tensor(self.rois_dtype, [self.proposal_num], scope=tik.scope_ubuf, name="tmp_roi")
        sort_ub = self.tik_instance.Tensor("int32", [self.batch_n], scope=tik.scope_ubuf, name="sort_ub")

        with self.tik_instance.for_range(0, loop) as idx:
            with self.tik_instance.if_scope(idx % 2 == 0):
                with self.tik_instance.for_range(0, self.core_num) as i:
                    sort_ub[i * loop + idx].set_as(indice_ub[idx * self.core_num + i])

            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, self.core_num) as i:
                    sort_ub[i * loop + idx].set_as(indice_ub[idx * self.core_num + self.core_num - 1 - i])

        with self.tik_instance.for_range(0, self.batch_n) as n:
            tmp_scalar.set_as(sort_ub[n])
            self.data_move(tmp_roi, self.rois_gm[tmp_scalar * self.proposal_num], num=self.proposal_num)
            self.data_move(self.sort_gm[n * self.proposal_num], tmp_roi, num=self.proposal_num)
        self.data_move(self.idx_gm, sort_ub, num=self.batch_n)

    def get_dtype_size(self, dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2}
        return dtype_dict.get(dtype)

    def data_move(self, dst, src, num, src_stride=0, dst_stride=0):
        """
        move data
        """
        sid = 0
        nburst = 1
        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        burst_len = (num + data_each_block - 1) // data_each_block
        self.tik_instance.data_move(dst, src, sid, nburst, burst_len, src_stride=src_stride,
                                    dst_stride=dst_stride)

    def single_operator_template(self, op_obj, dst, src, offsets, scalar=None, num=0, dst_stride=8, src_stride=8):
        """
        tik api template
        """
        vector_mask_max = 64
        dst_offset, src_offset = offsets

        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        if not dst_stride:
            dst_stride = vector_mask_max // data_each_block
            src_stride = vector_mask_max // data_each_block

        tensor_size = num if num else src.size
        loop = tensor_size // (vector_mask_max * 255)
        dst_blk_stride = 1
        src_blk_stride = 1

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                if scalar is not None:
                    op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], scalar, 255, dst_blk_stride,
                           src_blk_stride, dst_stride, src_stride)
                else:
                    op_obj(vector_mask_max, dst[tmp_dst_offset], src[tmp_src_offset], 255, dst_blk_stride,
                           src_blk_stride, dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            if scalar is not None:
                op_obj(vector_mask_max, dst[dst_offset], src[src_offset], scalar, repeat_time, dst_blk_stride,
                       src_blk_stride, dst_stride, src_stride)
            else:
                op_obj(vector_mask_max, dst[dst_offset], src[src_offset], repeat_time, dst_blk_stride,
                       src_blk_stride, dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            if scalar is not None:
                op_obj(last_num, dst[dst_offset], src[src_offset], scalar, 1, dst_blk_stride, src_blk_stride,
                       dst_stride, src_stride)
            else:
                op_obj(last_num, dst[dst_offset], src[src_offset], 1, dst_blk_stride, src_blk_stride,
                       dst_stride, src_stride)

    def double_operator_template(self, op_obj, dst, src0, src1, offsets, num=0, dst_stride=None, src0_stride=None,
                                 src1_stride=None):
        """
        tik api template
        """
        vector_mask_max = 64
        dst_offset, src0_offset, src1_offset = offsets

        dtype_byte_size = self.get_dtype_size(dst.dtype)
        data_each_block = self.block_byte_size // dtype_byte_size
        if not dst_stride:
            dst_stride = vector_mask_max // data_each_block
            src0_stride = vector_mask_max // data_each_block
            src1_stride = vector_mask_max // data_each_block

        tensor_size = num if num else src1.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src0_offset = src0_offset + index * vector_mask_max * 255
                tmp_src1_offset = src1_offset + index * vector_mask_max * 255
                op_obj(vector_mask_max, dst[tmp_dst_offset], src0[tmp_src0_offset], src1[tmp_src1_offset], 255,
                       dst_stride, src0_stride, src1_stride)

            dst_offset += loop * vector_mask_max * 255
            src0_offset += loop * vector_mask_max * 255
            src1_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            op_obj(vector_mask_max, dst[dst_offset], src0[src0_offset], src1[src1_offset], repeat_time, dst_stride,
                   src0_stride, src1_stride)
            dst_offset += repeat_time * vector_mask_max
            src0_offset += repeat_time * vector_mask_max
            src1_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            op_obj(last_num, dst[dst_offset], src0[src0_offset], src1[src1_offset], 1, dst_stride, src0_stride,
                   src1_stride)

    def data_sub(self, dst, src0, src1, offsets, num=0, dst_stride=None, src0_stride=None, src1_stride=None):
        """
        tik sub
        """
        self.double_operator_template(self.tik_instance.vec_sub, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_mul(self, dst, src0, src1, offsets, num=0, dst_stride=None, src0_stride=None, src1_stride=None):
        """
        tik mul
        """
        self.double_operator_template(self.tik_instance.vec_mul, dst, src0, src1, offsets, num, dst_stride, src0_stride,
                                      src1_stride)

    def data_conv(self, dst, src, offsets, mode="ceil", num=0, dst_stride=8, src_stride=8):
        """
        tik conv
        """
        vector_mask_max = 64
        dst_offset, src_offset = offsets

        tensor_size = num if num else src.size
        loop = tensor_size // (vector_mask_max * 255)

        if loop > 0:
            with self.tik_instance.for_range(0, loop) as index:
                tmp_dst_offset = dst_offset + index * vector_mask_max * 255
                tmp_src_offset = src_offset + index * vector_mask_max * 255
                self.tik_instance.vec_conv(vector_mask_max, mode, dst[tmp_dst_offset], src[tmp_src_offset], 255,
                                           dst_stride, src_stride)

            dst_offset += loop * vector_mask_max * 255
            src_offset += loop * vector_mask_max * 255

        repeat_time = (tensor_size % (vector_mask_max * 255)) // vector_mask_max

        if repeat_time > 0:
            self.tik_instance.vec_conv(vector_mask_max, mode, dst[dst_offset], src[src_offset], repeat_time,
                                       dst_stride, src_stride)
            dst_offset += repeat_time * vector_mask_max
            src_offset += repeat_time * vector_mask_max

        last_num = tensor_size % vector_mask_max
        if last_num > 0:
            self.tik_instance.vec_conv(last_num, mode, dst[dst_offset], src[src_offset], 1, dst_stride, src_stride)

    def data_sqrt(self, dst, src, offsets, num=0, dst_stride=None, src_stride=None):
        """
        tik sqrt
        """
        self.single_operator_template(self.tik_instance.vsqrt, dst, src, offsets, None, num, dst_stride, src_stride)

    def data_maxs(self, dst, src, scalar, offsets, num=0, dst_stride=None, src_stride=None):
        """
        tik maxs
        """
        self.single_operator_template(self.tik_instance.vmaxs, dst, src, offsets, scalar, num, dst_stride, src_stride)


# 'pylint: disable=unused-argument,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def balance_rois(rois, sort_rois, sort_idx, kernel_name):
    """
    op func
    """
    obj = BalanceRoiByArea(rois, sort_rois, sort_idx, kernel_name)
    return obj.compute()
