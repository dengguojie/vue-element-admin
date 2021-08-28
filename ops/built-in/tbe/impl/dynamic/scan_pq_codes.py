# Copyright 2021 Huawei Technologies Co., Ltd
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
scan_pq_codes.py
"""
import numpy as np
from impl.util import util_tik_comm_func
from impl.util import util_common
from impl.util.util_tik_comm_func import OpBase
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import set_current_compile_soc_info

# max int64
MAX_INT64 = 2 ** 63 - 1
MIN_FP16 = -65504
MAX_FP16 = 65504
# tiling param num
TILING_ARG_NUM = 16
# reserved ub size
RESERVED_UB_SIZE = 4 * 1024
IVF_SEGMENT = 512
INDEX_LEN = 16
MAX_BUCKET_LEN = 64
MASK_FLOAT16 = 128
MASK_INT32 = 64
BLOCK_INT32 = 8
BLOCK_FLOAT16 = 16
BLOCK_INT8 = 32
DIV_RATE = 2
IVF_UNIT_LEN = 16


def _ceil_div(dividend, divisor):
    result = (dividend + divisor - 1) // divisor
    return result


def _ceil_fill(dividend, divisor):
    result = ((dividend + divisor - 1) // divisor) * divisor
    return result


# pylint: disable=too-many-instance-attributes,too-many-arguments,unused-argument
# pylint: disable=too-many-locals,too-many-statements,unused-argument,invalid-name
class ScanPQCodes(object):
    """
    Function: use to store ScanPQCodes base parameters
    """
    def __init__(self, attrs, dtypes, bucket_shape):
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik()
        self.opt_config = {"out_of_bound_sync_check": True,
                           "enable_const_fold": True}
        self.tiling_gm = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.bucket_loop = self.tik_instance.Scalar("int32", name="bucket_loop")
        self.slice_size = 1024
        # attrs
        (group_size, total_limit, extreme_mode, split_count, split_index) = attrs
        self.group_size = group_size
        self.total_limit = total_limit
        self.extreme_mode = extreme_mode
        self.is_second = False
        self.bucket_shape = bucket_shape
        self.split_count = split_count
        self.split_index = split_index
        # dtype
        (ivf_dtype, bucket_list_dtype, bucket_base_distance_dtype,
         bucket_limits_dtype, bucket_offsets_dtype,
         adc_tables_dtype) = dtypes
        self.ivf_dtype = ivf_dtype
        self.bucket_list_dtype = bucket_list_dtype
        self.bucket_base_distance_dtype = bucket_base_distance_dtype
        self.bucket_limits_dtype = bucket_limits_dtype
        self.bucket_offsets_dtype = bucket_offsets_dtype
        self.adc_tables_dtype = adc_tables_dtype
        # input gm
        self.ivf_gm = None
        self.bucket_list_gm = None
        self.bucket_base_distance_gm = None
        self.bucket_limits_gm = None
        self.bucket_offsets_gm = None
        self.adc_tables_gm = None
        # output gm
        self.actual_count_gm = None
        self.pq_distance_gm = None
        self.grouped_extrim_distance_gm = None
        self.pq_ivf_gm = None
        self.pq_index_gm = None
        # ub
        self.adc_tables_ub = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                      (256, 16, 16),
                                                      name="adc_tables_ub",
                                                      scope=tik.scope_ubuf)

        self.bucket_list_ub_int32 = self.tik_instance.Tensor("int32",
                                                             (MAX_BUCKET_LEN,),
                                                             name="bucket_list_ub_int32",
                                                             scope=tik.scope_ubuf)
        self.bucket_counts_ub_int32 = self.tik_instance.Tensor("int32",
                                                               (MAX_BUCKET_LEN,),
                                                               name="bucket_counts_ub_int32",
                                                               scope=tik.scope_ubuf)
        self.bucket_offsets_ub_int32 = self.tik_instance.Tensor("int32",
                                                                (MAX_BUCKET_LEN,),
                                                                name="bucket_offsets_ub_int32",
                                                                scope=tik.scope_ubuf)
        self.bucket_base_distance_ub_fp16 = self.tik_instance.Tensor(
            self.bucket_base_distance_dtype, (MAX_BUCKET_LEN,),
            name="bucket_base_distance_ub_fp16",
            scope=tik.scope_ubuf)
        self.actual_count_ub_int32 = self.tik_instance.Tensor("int32",
                                                              (MAX_BUCKET_LEN,),
                                                              name="actual_count_ub_int32",
                                                              scope=tik.scope_ubuf)
        self.buffer_ub_fp32 = self.tik_instance.Tensor("float32",
                                                       (MAX_BUCKET_LEN,),
                                                       name="buffer_ub_fp32",
                                                       scope=tik.scope_ubuf)
        self.adc_tables_trans2_ub = self.tik_instance.Tensor(
            self.adc_tables_dtype, (16, 128, 16),
            name="adc_tables_trans2_ub", scope=tik.scope_ubuf)
        self.adc_tables_input_ub = self.tik_instance.Tensor(
            self.adc_tables_dtype, (16, 256),
            name="adc_tables_input_ub", scope=tik.scope_ubuf)
        self.adc_tables_trans1_ub = self.tik_instance.Tensor(
            self.adc_tables_dtype, (256, 16),
            name="adc_tables_trans1_ub", scope=tik.scope_ubuf)

    def _init_gm_tensor(self):
        # input gm
        self.ivf_gm = self.tik_instance.Tensor(self.ivf_dtype, (MAX_INT64,),
                                               name="ivf", scope=tik.scope_gm)
        self.bucket_list_gm = self.tik_instance.Tensor(self.bucket_list_dtype,
                                                       (MAX_INT64,),
                                                       name="bucket_list",
                                                       scope=tik.scope_gm)
        self.bucket_base_distance_gm = self.tik_instance.Tensor(
            self.bucket_base_distance_dtype, (MAX_INT64,),
            name="bucket_base_distance", scope=tik.scope_gm)
        self.bucket_limits_gm = self.tik_instance.Tensor(
            self.bucket_limits_dtype, (MAX_INT64,),
            name="bucket_limits", scope=tik.scope_gm)
        self.bucket_offsets_gm = self.tik_instance.Tensor(
            self.bucket_offsets_dtype, (MAX_INT64,),
            name="bucket_offsets", scope=tik.scope_gm)
        self.adc_tables_gm = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                      (MAX_INT64,),
                                                      name="adc_tables",
                                                      scope=tik.scope_gm)
        # output gm
        self.actual_count_gm = self.tik_instance.Tensor(self.bucket_list_dtype,
                                                        (BLOCK_INT32,),
                                                        name="actual_count",
                                                        scope=tik.scope_gm)
        self.pq_distance_gm = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                       (self.total_limit,),
                                                       name="pq_distance",
                                                       scope=tik.scope_gm)
        self.grouped_extrim_distance_gm = self.tik_instance.Tensor(
            self.adc_tables_dtype,
            (_ceil_div(self.total_limit, self.group_size),),
            name="grouped_extrim_distance", scope=tik.scope_gm)
        self.pq_ivf_gm = self.tik_instance.Tensor(self.bucket_list_dtype,
                                                  (self.total_limit,),
                                                  name="pq_ivf",
                                                  scope=tik.scope_gm)
        self.pq_index_gm = self.tik_instance.Tensor(self.bucket_list_dtype,
                                                    (self.total_limit,),
                                                    name="pq_index",
                                                    scope=tik.scope_gm)

    def _vector_div(self, vector_ub, div_value):
        self.tik_instance.vconv(MASK_INT32, "", self.buffer_ub_fp32, vector_ub,
                                1, 1, 1, 8, 8)
        div_value_fp32 = self.tik_instance.Scalar("float32",
                                                  name="div_value_fp32")
        div_value_fp32.set_as(1 / div_value)
        self.tik_instance.vmuls(MASK_INT32, self.buffer_ub_fp32,
                                self.buffer_ub_fp32, div_value_fp32, 1, 1, 1, 8,
                                8)
        self.tik_instance.vconv(MASK_INT32, "floor", vector_ub,
                                self.buffer_ub_fp32, 1, 1, 1, 8, 8)

    def _get_input_data_paras(self):
        bucket_block_int32_num = self.tik_instance.Scalar("int32",
                                                          name="bucket_block_int32_num")
        bucket_block_int32_left = self.tik_instance.Scalar("int32",
                                                           name="bucket_block_int32_left")
        bucket_block_fp16_num = self.tik_instance.Scalar("int32",
                                                         name="bucket_block_fp16_num")
        bucket_block_fp16_left = self.tik_instance.Scalar("int32",
                                                          name="bucket_block_fp16_left")
        bucket_block_int32_num.set_as(self.bucket_shape // BLOCK_INT32)
        bucket_block_int32_left.set_as(self.bucket_shape % BLOCK_INT32)
        bucket_block_fp16_num.set_as(self.bucket_shape // BLOCK_FLOAT16)
        bucket_block_fp16_left.set_as(self.bucket_shape % BLOCK_FLOAT16)
        self.tik_instance.vector_dup(MAX_BUCKET_LEN,
                                     self.bucket_counts_ub_int32, 0, 1, 1, 8)
        with self.tik_instance.if_scope(bucket_block_int32_num > 0):
            self.tik_instance.data_move(self.bucket_list_ub_int32,
                                        self.bucket_list_gm, 0, 1,
                                        bucket_block_int32_num, 0,
                                        0)
            self.tik_instance.data_move(self.bucket_counts_ub_int32,
                                        self.bucket_limits_gm, 0, 1,
                                        bucket_block_int32_num, 0, 0)
            self.tik_instance.data_move(self.bucket_offsets_ub_int32,
                                        self.bucket_offsets_gm, 0, 1,
                                        bucket_block_int32_num, 0, 0)
        with self.tik_instance.if_scope(bucket_block_int32_left > 0):
            with self.tik_instance.for_range(0,
                                             bucket_block_int32_left) as bucket_int32_idx:
                self.bucket_list_ub_int32[bucket_int32_idx].set_as(
                    self.bucket_list_gm[
                        bucket_block_int32_num * BLOCK_INT32 + bucket_int32_idx])
                self.bucket_counts_ub_int32[bucket_int32_idx].set_as(
                    self.bucket_limits_gm[
                        bucket_block_int32_num * BLOCK_INT32 + bucket_int32_idx])
                self.bucket_offsets_ub_int32[bucket_int32_idx].set_as(
                    self.bucket_offsets_gm[
                        bucket_block_int32_num * BLOCK_INT32 + bucket_int32_idx])
        with self.tik_instance.if_scope(bucket_block_fp16_num > 0):
            self.tik_instance.data_move(self.bucket_base_distance_ub_fp16,
                                        self.bucket_base_distance_gm, 0, 1,
                                        bucket_block_fp16_num, 0, 0)
        with self.tik_instance.if_scope(bucket_block_fp16_left > 0):
            with self.tik_instance.for_range(0,
                                             bucket_block_fp16_left) as bucket_fp16_idx:
                self.bucket_base_distance_ub_fp16[bucket_fp16_idx].set_as(
                    self.bucket_base_distance_gm[
                        bucket_block_fp16_num * BLOCK_FLOAT16 + bucket_fp16_idx])
        self._vector_div(self.bucket_counts_ub_int32, self.split_count)
        unit_counts_ub_int32 = self.tik_instance.Tensor("int32",
                                                        (MAX_BUCKET_LEN,),
                                                        name="unit_counts_ub_int32",
                                                        scope=tik.scope_ubuf)
        self.tik_instance.vmuls(MASK_INT32, unit_counts_ub_int32,
                                self.bucket_counts_ub_int32, self.split_index,
                                1, 1,
                                1, 8, 8)
        self.tik_instance.vadd(MAX_BUCKET_LEN, self.bucket_offsets_ub_int32,
                               self.bucket_offsets_ub_int32,
                               unit_counts_ub_int32, 1, 1, 1, 1, 8, 8, 8)

    def _tiling_args(self):
        """
        tiling_args
        """
        tiling_ub = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,),
                                             name="tiling_ub",
                                             scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
        tiling_para_index = 0
        self.bucket_loop.set_as(tiling_ub[tiling_para_index])
        # self.bucket_loop.set_as(2)

    def _create_adc_table(self, bucket_idx):
        # conver adc_tables shape from (256,16) to (256,16,16) to prevent bank conflict
        dst_list = [self.adc_tables_trans1_ub[16 * i] for i in range(16)]
        src_list = [self.adc_tables_input_ub[256 * i] for i in range(16)]
        # move in adc_tables
        self.tik_instance.data_move(self.adc_tables_input_ub,
                                    self.adc_tables_gm[bucket_idx * 256 * 16],
                                    0, 1, 256, 0,
                                    0)
        self.tik_instance.vnchwconv(False, False, dst_list, src_list, 16, 16, 1)
        with self.tik_instance.for_range(0, 2) as i:
            self.tik_instance.vector_dup(128, self.adc_tables_trans2_ub[i * 8 * 128 * 16], 0.0, 128, 1, 8)
        with self.tik_instance.for_range(0, 2) as tran_idx:
            self.tik_instance.data_move(self.adc_tables_trans2_ub,
                                        self.adc_tables_trans1_ub[tran_idx * 128 * 16], 0, 1,
                                        128, 0, 0)
            dst_list = [self.adc_tables_ub[128 * 16 * 16 * tran_idx + 16 * j]
                        for j in range(16)]
            src_list = [self.adc_tables_trans2_ub[128 * 16 * j] for j in range(16)]
            self.tik_instance.vnchwconv(False, False, dst_list, src_list, 128, 16, 1)

    def _run_one_core(self, ivf_input, ivf_output, ivf_max_base, index_base,
                      do_num_per_core, bucket_base_distance,
                      bucket_idx, is_dynamic):
        thread_loop = self.tik_instance.Scalar("int32", name="thread_loop")
        thread_tail_num = self.tik_instance.Scalar("int32",
                                                   name="thread_tail_num")
        handle_offset = self.tik_instance.Scalar("int32", name="handle_offset")
        handle_buffer_size = self.tik_instance.Scalar("int32",
                                                      name="handle_buffer_size")
        bucket_id = self.tik_instance.Scalar("int32", name="bucket_id")
        thread_loop.set_as(do_num_per_core // self.slice_size)
        thread_tail_num.set_as(do_num_per_core % self.slice_size)
        bucket_id.set_as(self.bucket_list_ub_int32[bucket_idx])
        assist_add_ub = self.tik_instance.Tensor(self.bucket_offsets_dtype,
                                                 [MAX_BUCKET_LEN, ],
                                                 name="assist_add_ub",
                                                 scope=tik.scope_ubuf)
        assist_add_init_ub = self.bucket_list_ub_int32
        with self.tik_instance.for_range(0, 64) as i:
            assist_add_init_ub[i].set_as((i % 16) * 16)
        self.tik_instance.data_move(assist_add_ub, assist_add_init_ub, 0, 1, 8,
                                    0, 0)
        if is_dynamic == 1:
            grouped_extrim_distance_ub = self.tik_instance.Tensor(
                self.adc_tables_dtype, (
                    _ceil_div(do_num_per_core,
                              self.slice_size) * BLOCK_FLOAT16,),
                name="grouped_extrim_distance_ub",
                scope=tik.scope_ubuf)
            block_extrim_ub = self.tik_instance.Tensor(self.adc_tables_dtype, (
                _ceil_div(do_num_per_core,
                          self.slice_size) * BLOCK_FLOAT16 * 2,),
                                                       name="block_extrim_ub",
                                                       scope=tik.scope_ubuf)
        else:
            grouped_extrim_distance_ub = self.tik_instance.Tensor(
                self.adc_tables_dtype, (2 * BLOCK_FLOAT16,),
                name="grouped_extrim_distance_ub",
                scope=tik.scope_ubuf)
            block_extrim_ub = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                       (2 * BLOCK_FLOAT16 * 2,),
                                                       name="block_extrim_ub",
                                                       scope=tik.scope_ubuf)

        def _ivf_inner_func(args):
            (handle_buffer_size, handle_offset, block_extrim_ub, ivf_input,
             ivf_output, index_base, thread_idx) = args
            ub_size = self.tik_instance.Scalar("int32", name="ub_size")
            ub_size.set_as(_ceil_fill(handle_buffer_size, self.group_size))
            if is_dynamic == 1:
                ivf_cur_process_ub_fp16 = self.tik_instance.Tensor(
                    self.adc_tables_dtype, (ub_size, 16),
                    name="ivf_cur_process_ub_fp16", scope=tik.scope_ubuf)
            else:
                ivf_cur_process_ub_fp16 = self.tik_instance.Tensor(
                    self.adc_tables_dtype, (self.slice_size // 2, 16),
                    name="ivf_cur_process_ub_fp16", scope=tik.scope_ubuf)
            ivf_cur_process_ub_uint8 = self.adc_tables_trans2_ub.reinterpret_cast_to(
                self.ivf_dtype)
            assist_pq_index_ub = self.adc_tables_trans1_ub.reinterpret_cast_to(
                "int32")
            if is_dynamic == 1:
                pq_distance_ub = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                          (ub_size,),
                                                          name="pq_distance_ub",
                                                          scope=tik.scope_ubuf)
            else:
                pq_distance_ub = self.tik_instance.Tensor(self.adc_tables_dtype,
                                                          (self.slice_size // 2,),
                                                          name="pq_distance_ub",
                                                          scope=tik.scope_ubuf)
            # input data
            with self.tik_instance.if_scope(handle_buffer_size // 2 > 0):
                self.tik_instance.data_move(
                    ivf_cur_process_ub_uint8[thread_idx * ub_size * 16],
                    self.ivf_gm[
                        ivf_input * IVF_UNIT_LEN + handle_offset * IVF_UNIT_LEN],
                    0, 1,
                    handle_buffer_size // 2, 0, 0)
            with self.tik_instance.if_scope(handle_buffer_size % 2 > 0):
                with self.tik_instance.for_range(0, BLOCK_FLOAT16) as idx:
                    ivf_cur_process_ub_uint8[thread_idx * ub_size * 16 + (
                            handle_buffer_size // 2) * 16 + idx].set_as(
                        self.ivf_gm[
                            ivf_input * IVF_UNIT_LEN + handle_offset * IVF_UNIT_LEN + (
                                    handle_buffer_size // 2) * 16 + idx]
                    )
            # ivf index reprocess
            self.tik_instance.vconv(MASK_FLOAT16, "", ivf_cur_process_ub_fp16,
                                    ivf_cur_process_ub_uint8[
                                        thread_idx * ub_size * 16],
                                    ub_size * 16 // MASK_FLOAT16,
                                    1, 1, 8, 4)
            ivf_cur_process_ub_int32 = self.adc_tables_trans2_ub.reinterpret_cast_to(
                "int32")
            self.tik_instance.vconv(MASK_INT32, "floor",
                                    ivf_cur_process_ub_int32,
                                    ivf_cur_process_ub_fp16,
                                    ub_size * 16 // MASK_INT32, 1, 1, 8, 4)

            self.tik_instance.vmuls(MASK_INT32, ivf_cur_process_ub_int32,
                                    ivf_cur_process_ub_int32, 256,
                                    ub_size * 16 // 64, 1, 1, 8, 8)
            self.tik_instance.vadd(MASK_INT32, ivf_cur_process_ub_int32,
                                   ivf_cur_process_ub_int32,
                                   assist_add_ub, ub_size * 16 // MASK_INT32, 1,
                                   1, 1, 8, 8, 0)
            # *2 because offset Bytes
            self.tik_instance.vmuls(MASK_INT32, ivf_cur_process_ub_int32,
                                    ivf_cur_process_ub_int32, 2,
                                    ub_size * 16 // MASK_INT32, 1, 1, 8, 8)
            # distance calc
            vgather_out_ub = ivf_cur_process_ub_fp16
            pq_distance_vcgadd_ub = self.adc_tables_input_ub
            self.tik_instance.vgather(handle_buffer_size * 16, vgather_out_ub,
                                      self.adc_tables_ub,
                                      ivf_cur_process_ub_int32, 1, 8, 0, 0,
                                      "counter")
            self.tik_instance.vcgadd(MASK_FLOAT16, pq_distance_vcgadd_ub[
                thread_idx * ub_size], vgather_out_ub,
                                     ub_size * 16 // MASK_FLOAT16, 1, 1, 8)
            # tail for pq_distance_ub
            with self.tik_instance.if_scope(
                    handle_buffer_size % self.group_size > 0):
                if self.extreme_mode == 1:
                    self.tik_instance.vector_dup(self.group_size,
                                                 pq_distance_ub[
                                                     (
                                                             handle_buffer_size // self.group_size) * self.group_size],
                                                 MIN_FP16, 1, 1, 4)
                else:
                    self.tik_instance.vector_dup(self.group_size,
                                                 pq_distance_ub[
                                                     (handle_buffer_size // self.group_size) * self.group_size],
                                                 MAX_FP16, 1, 1, 4)
            with self.tik_instance.if_scope(
                    handle_buffer_size // BLOCK_FLOAT16 > 0):
                self.tik_instance.data_move(pq_distance_ub,
                                            pq_distance_vcgadd_ub[
                                                thread_idx * ub_size], 0, 1,
                                            handle_buffer_size // BLOCK_FLOAT16,
                                            0, 0)
            with self.tik_instance.if_scope(handle_buffer_size % BLOCK_FLOAT16 > 0):
                with self.tik_instance.for_range(0,
                                                 handle_buffer_size % BLOCK_FLOAT16) as idx:
                    vcgadd_offset = self.tik_instance.Scalar("int32",
                                                             name="vcgadd_offset")
                    vcgadd_offset.set_as(
                        (handle_buffer_size // BLOCK_FLOAT16) * BLOCK_FLOAT16)
                    pq_distance_ub[vcgadd_offset + idx].set_as(
                        pq_distance_vcgadd_ub[
                            thread_idx * ub_size + vcgadd_offset + idx])
                    self.tik_instance.vadds(MASK_FLOAT16, pq_distance_ub,
                                            pq_distance_ub, bucket_base_distance,
                                            ub_size // MASK_FLOAT16, 1, 1, 8, 8)
            # index and bucketid
            assist_pq_index_init_ub = self.adc_tables_input_ub.reinterpret_cast_to(
                "int32")
            with self.tik_instance.for_range(0, self.slice_size // 2) as i:
                assist_pq_index_init_ub[i].set_as(i)
            with self.tik_instance.if_scope(
                    ub_size // (self.slice_size // 2) > 0):
                self.tik_instance.data_move(
                    assist_pq_index_ub[thread_idx * ub_size],
                    assist_pq_index_init_ub, 0, 1,
                    (self.slice_size // 2) // 8, 0, 0)
            with self.tik_instance.if_scope(
                    ub_size % (self.slice_size // 2) > 0):
                self.tik_instance.data_move(assist_pq_index_ub[
                                                thread_idx * ub_size + (
                                                        ub_size // (
                                                        self.slice_size // 2)) * (
                                                        self.slice_size // 2)],
                                            assist_pq_index_init_ub, 0, 1,
                                            (ub_size % (
                                                    self.slice_size // 2)) // 8,
                                            0, 0)
            self.tik_instance.vadds(MASK_INT32, assist_pq_index_ub[
                thread_idx * ub_size + (ub_size // (self.slice_size // 2)) * (
                        self.slice_size // 2)],
                                    assist_pq_index_ub[
                                        (ub_size // (self.slice_size // 2)) * (
                                                self.slice_size // 2)],
                                    (ub_size // (self.slice_size // 2)) * (
                                            self.slice_size // 2),
                                    (ub_size % (self.slice_size // 2)) // 64, 1,
                                    1, 8, 8)
            self.tik_instance.vadds(MASK_INT32,
                                    assist_pq_index_ub[thread_idx * ub_size],
                                    assist_pq_index_ub[thread_idx * ub_size],
                                    index_base + handle_offset, ub_size // 64,
                                    1, 1, 8, 8)
            # extreme
            if self.extreme_mode == 1:
                self.tik_instance.vcmax(self.group_size, block_extrim_ub[
                    (handle_offset // self.group_size) * 2],
                                        pq_distance_ub,
                                        ub_size // self.group_size, 1, 1,
                                        self.group_size // BLOCK_FLOAT16)
            else:
                self.tik_instance.vcmin(self.group_size, block_extrim_ub[
                    (handle_offset // self.group_size) * 2],
                                        pq_distance_ub,
                                        ub_size // self.group_size, 1, 1,
                                        self.group_size // BLOCK_FLOAT16)

            # move out
            self.tik_instance.data_move(
                self.pq_distance_gm[ivf_output + handle_offset], pq_distance_ub, 0, 1,
                ub_size // BLOCK_FLOAT16, 0, 0)
            self.tik_instance.data_move(
                self.pq_index_gm[ivf_output + handle_offset],
                assist_pq_index_ub[thread_idx * ub_size], 0, 1,
                ub_size // BLOCK_INT32, 0, 0)
            pq_ivf_ub = assist_pq_index_ub
            self.tik_instance.vector_dup(MASK_INT32,
                                         pq_ivf_ub[thread_idx * ub_size],
                                         bucket_id, ub_size // MASK_INT32,
                                         1, 8)
            self.tik_instance.data_move(
                self.pq_ivf_gm[ivf_output + handle_offset],
                pq_ivf_ub[thread_idx * ub_size], 0,
                1, ub_size // BLOCK_INT32, 0, 0)

        with self.tik_instance.for_range(0, thread_loop * 2,
                                         thread_num=2) as thread_idx:
            handle_buffer_size.set_as(self.slice_size // 2)
            handle_offset.set_as(thread_idx * handle_buffer_size)
            args = (
                handle_buffer_size, handle_offset, block_extrim_ub, ivf_input,
                ivf_output, index_base, thread_idx)
            _ivf_inner_func(args)
        with self.tik_instance.if_scope(thread_loop > 0):
            self.tik_instance.vreduce(thread_loop * BLOCK_FLOAT16 * 2,
                              grouped_extrim_distance_ub, block_extrim_ub, 1, 1,
                              1, 2,
                              0, 0, None, "counter")
            self.tik_instance.data_move(
                self.grouped_extrim_distance_gm[ivf_max_base],
                grouped_extrim_distance_ub, 0, 1,
                thread_loop, 0, 0)
        # tail
        with self.tik_instance.if_scope(thread_tail_num > 0):
            tail_block_extrim_ub = self.tik_instance.Tensor(
                self.adc_tables_dtype, (BLOCK_FLOAT16 * 2,),
                name="tail_block_extrim_ub", scope=tik.scope_ubuf)
            tail_ivf_base_input = self.tik_instance.Scalar("int32",
                                                           name="tail_ivf_base_input")
            tail_ivf_base_output = self.tik_instance.Scalar("int32",
                                                            name="tail_ivf_base_output")
            tail_index_base = self.tik_instance.Scalar("int32",
                                                       name="tail_index_base")
            tail_ivf_base_input.set_as(
                ivf_input + thread_loop * self.slice_size)
            tail_ivf_base_output.set_as(
                ivf_output + thread_loop * self.slice_size)
            tail_index_base.set_as(
                index_base + thread_loop * self.slice_size)
            args = (
                thread_tail_num, 0, tail_block_extrim_ub, tail_ivf_base_input,
                tail_ivf_base_output, tail_index_base, 0)
            _ivf_inner_func(args)
            self.tik_instance.data_move(
                block_extrim_ub[thread_loop * BLOCK_FLOAT16 * 2],
                tail_block_extrim_ub, 0, 1, 2,
                0, 0)
            self.tik_instance.vreduce(BLOCK_FLOAT16 * 2,
                                      grouped_extrim_distance_ub[
                                          thread_loop * BLOCK_FLOAT16],
                                      block_extrim_ub[
                                          thread_loop * BLOCK_FLOAT16 * 2], 1,
                                      1, 1, 2, 0, 0, None,
                                      "counter")
            self.tik_instance.data_move(self.grouped_extrim_distance_gm[
                                            ivf_max_base + thread_loop * BLOCK_FLOAT16],
                                        grouped_extrim_distance_ub[
                                            thread_loop * BLOCK_FLOAT16], 0, 1,
                                        1, 0, 0)

    def _run_multi_core(self, is_dynamic):
        with self.tik_instance.for_range(0, self.core_nums,
                                         block_num=self.core_nums) as core_idx:
            ivf_base_output = self.tik_instance.Scalar("int32",
                                                       name="ivf_base_output")
            actual_count = self.tik_instance.Scalar("int32",
                                                    name="actual_count")
            bucket_counts = self.tik_instance.Scalar("int32",
                                                     name="bucket_counts")
            ivf_base_output.set_as(0)
            actual_count.set_as(0)
            with self.tik_instance.for_range(0, self.bucket_loop) as idx:
                bucket_counts.set_as(self.bucket_counts_ub_int32[idx])
                actual_count.set_as(
                    actual_count + _ceil_fill(bucket_counts, self.group_size))
            with self.tik_instance.for_range(0, self.bucket_loop) as bucket_idx:
                core_used_num = self.tik_instance.Scalar("int32",
                                                         name="core_used_num")
                do_num_per_core = self.tik_instance.Scalar("int32",
                                                           name="do_num_per_core")
                do_num_tail_core = self.tik_instance.Scalar("int32",
                                                            name="do_num_tail_core")
                ivf_base = self.tik_instance.Scalar("int32", name="ivf_base")
                ivf_input = self.tik_instance.Scalar("int32", name="ivf_input")
                ivf_output = self.tik_instance.Scalar("int32",
                                                      name="ivf_output")
                ivf_max_base = self.tik_instance.Scalar("int32",
                                                        name="ivf_max_base")
                index_base = self.tik_instance.Scalar("int32",
                                                      name="index_base")
                bucket_base_distance = self.tik_instance.Scalar("float16",
                                                                name="bucket_base_distance")
                bucket_block_num = self.tik_instance.Scalar("int32",
                                                            name="bucket_block_num")
                core_block_offset = self.tik_instance.Scalar("int32",
                                                             name="core_block_offset")
                bucket_counts.set_as(self.bucket_counts_ub_int32[bucket_idx])
                bucket_block_num.set_as(
                    _ceil_div(bucket_counts, self.slice_size))
                do_num_per_core.set_as(_ceil_div(bucket_block_num,
                                                 self.core_nums) * self.slice_size)
                core_used_num.set_as(_ceil_div(bucket_counts, do_num_per_core))
                do_num_tail_core.set_as(bucket_counts % do_num_per_core)
                bucket_base_distance.set_as(
                    self.bucket_base_distance_ub_fp16[bucket_idx])
                ivf_base.set_as(self.bucket_offsets_ub_int32[bucket_idx])
                core_block_offset.set_as(core_idx * do_num_per_core)
                ivf_input.set_as(ivf_base + core_block_offset)
                ivf_output.set_as(ivf_base_output + core_block_offset)
                ivf_max_base.set_as(
                    ivf_base_output // self.group_size + core_block_offset // self.group_size)
                index_base.set_as(core_block_offset)
                with self.tik_instance.if_scope(core_idx == 0):
                    self.actual_count_ub_int32[0].set_as(actual_count)
                    self.tik_instance.data_move(self.actual_count_gm,
                                                self.actual_count_ub_int32, 0, 1, 1,
                                                0, 0)
                    with self.tik_instance.if_scope(do_num_tail_core > 0):
                        self._create_adc_table(bucket_idx)
                with self.tik_instance.if_scope(core_idx < core_used_num - 1):
                    self._run_one_core(ivf_input, ivf_output, ivf_max_base,
                                       index_base, do_num_per_core,
                                       bucket_base_distance, bucket_idx, is_dynamic)
                with self.tik_instance.if_scope(core_idx == core_used_num - 1):
                    self._run_one_core(ivf_input, ivf_output, ivf_max_base,
                                       index_base, do_num_tail_core,
                                       bucket_base_distance, bucket_idx, is_dynamic)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(core_idx < core_used_num):
                        self._create_adc_table(bucket_idx)
                        self._run_one_core(ivf_input, ivf_output, ivf_max_base,
                                           index_base, do_num_per_core,
                                           bucket_base_distance, bucket_idx,
                                           is_dynamic)
                        ivf_base_output.set_as(ivf_base_output + _ceil_fill(bucket_counts,
                                                                     self.group_size))

    def scan_pq_codes_operator(self, is_dynamic, kernel_name):
        """
        scan_pq_codes_operator
        """
        self._tiling_args()
        self._init_gm_tensor()
        self._get_input_data_paras()
        self._run_multi_core(0)
        ub_size_bytes = tbe_platform.get_soc_spec(
            tbe_platform.UB_SIZE) - RESERVED_UB_SIZE
        ub_once_size = 122 * 1024
        # Build CCE
        # this "global_variable_link" flag suggest ccec.py do link without "-r" option
        # which will result in global variable in cce file with wrong address
        tbe_context.get_context().add_compile_info("vars", {
            "ub_total_size": ub_size_bytes,
            "ub_once_size": ub_once_size})
        input_list = [self.ivf_gm, self.bucket_list_gm,
                      self.bucket_base_distance_gm,
                      self.bucket_limits_gm, self.bucket_offsets_gm,
                      self.adc_tables_gm]
        output_list = [self.actual_count_gm, self.pq_distance_gm,
                       self.grouped_extrim_distance_gm,
                       self.pq_ivf_gm, self.pq_index_gm]
        self.tik_instance.BuildCCE(kernel_name=kernel_name,
                                   inputs=input_list,
                                   outputs=output_list,
                                   flowtable=(self.tiling_gm,),
                                   config=self.opt_config)

        return self.tik_instance


def _para_dtype_check(args_list):
    (ivf, bucket_list, bucket_base_distance, bucket_limits, bucket_offsets,
     adc_tables, actual_count, pq_distance,
     grouped_extrim_distance, pq_ivf, pq_index) = args_list
    # input
    ivf_dtype = ivf.get("dtype").lower()
    bucket_list_dtype = bucket_list.get("dtype").lower()
    bucket_base_distance_dtype = bucket_base_distance.get("dtype").lower()
    bucket_limits_dtype = bucket_limits.get("dtype").lower()
    bucket_offsets_dtype = bucket_offsets.get("dtype").lower()
    adc_tables_dtype = adc_tables.get("dtype").lower()
    para_check.check_dtype(ivf_dtype, ("uint8"), param_name="ivf")
    para_check.check_dtype(bucket_list_dtype, ("int32"),
                           param_name="bucket_list")
    para_check.check_dtype(bucket_base_distance_dtype, ("float16"),
                           param_name="bucket_base_distance")
    para_check.check_dtype(bucket_limits_dtype, ("int32"),
                           param_name="bucket_limits")
    para_check.check_dtype(bucket_offsets_dtype, ("int32"),
                           param_name="bucket_offsets")
    para_check.check_dtype(adc_tables_dtype, ("float16"),
                           param_name="adc_tables")
    # output
    actual_count_dtype = actual_count.get("dtype").lower()
    pq_distance_dtype = pq_distance.get("dtype").lower()
    grouped_extrim_distance_dtype = grouped_extrim_distance.get("dtype").lower()
    pq_ivf_dtype = pq_ivf.get("dtype").lower()
    pq_index_dtype = pq_index.get("dtype").lower()
    para_check.check_dtype(actual_count_dtype, ("int32"),
                           param_name="actual_count")
    para_check.check_dtype(pq_distance_dtype, ("float16"),
                           param_name="pq_distance")
    para_check.check_dtype(grouped_extrim_distance_dtype, ("float16"),
                           param_name="grouped_extrim_distance")
    para_check.check_dtype(pq_ivf_dtype, ("int32"), param_name="pq_ivf")
    para_check.check_dtype(pq_index_dtype, ("int32"), param_name="pq_index")


@register_operator("ScanPQCodes")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def scan_pq_codes(ivf, bucket_list, bucket_base_distance, bucket_limits,
                  bucket_offsets, adc_tables,
                  actual_count, pq_distance, grouped_extrim_distance, pq_ivf,
                  pq_index,
                  group_size, total_limit, extreme_mode, split_count,
                  split_index, kernel_name="scan_pq_codes"):
    args_list = (
        ivf, bucket_list, bucket_base_distance, bucket_limits, bucket_offsets,
        adc_tables,
        actual_count, pq_distance, grouped_extrim_distance, pq_ivf, pq_index)

    _para_dtype_check(args_list)
    ivf_dtype = ivf.get("dtype").lower()
    bucket_list_dtype = bucket_list.get("dtype").lower()
    bucket_base_distance_dtype = bucket_base_distance.get("dtype").lower()
    bucket_limits_dtype = bucket_limits.get("dtype").lower()
    bucket_offsets_dtype = bucket_offsets.get("dtype").lower()
    adc_tables_dtype = adc_tables.get("dtype").lower()
    bucket_shape = bucket_list.get("shape")[0]
    dtypes = (ivf_dtype, bucket_list_dtype, bucket_base_distance_dtype,
              bucket_limits_dtype, bucket_offsets_dtype,
              adc_tables_dtype)
    attrs = (group_size, total_limit, extreme_mode, split_count, split_index)
    obj = ScanPQCodes(attrs, dtypes, bucket_shape)
    if bucket_list.get("shape") == -1:
        is_dynamic = 1
    else:
        is_dynamic = 0
    is_dynamic = 0
    return obj.scan_pq_codes_operator(is_dynamic, kernel_name)
