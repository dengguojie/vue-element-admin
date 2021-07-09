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
lru
"""
# pylint: disable=unused-argument,too-many-arguments,invalid-name,no-self-use,too-many-branches
# pylint: disable=too-many-instance-attributes,unnecessary-comprehension,inconsistent-return-statements
import math
import functools
from te import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context

SHAPE_SIZE_ONE = 1
SHAPE_SIZE_FOUR = 4
FP16_VECTOR_MASK_MAX = 128
FP32_VECTOR_MASK_MAX = 64
INT8_VECTOR_MASK_MAX = 256
# vnchwconv instr compute min blocks
TRANS_MIN_BLKS = 16
# do transpose need split ub to trisection
TRISECTION_UB = 3
BLOCK_BYTES = 32
CONFIG_TWO = 2
REPEAT_TIMES_MAX = 255
SMMU_ID = 0
DATA_MOV_STRIDE = 0
DATA_MOV_NBURST = 1
REMAIN_REPEAT = 1
DEQ_SCALE = 1.0
SCALAR_HALF = 0.5
FP32_MAX = 3.4028 * 10**38


# pylint: disable=no-member,attribute-defined-outside-init,dangerous-default-value,consider-using-enumerate
class LRU:
    """
       Function: use to store  base parameters
       Modify : 2021-07-09
    """

    # pylint: disable=too-many-statements
    def __init__(self, index_list, data, cache, tag, is_last_call, out_data, out_cache, out_tag, index_offset_list,
                 not_in_cache_index_list, not_in_cache_number, pre_route_count, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.index_list_shape = index_list.get("shape")
        self.index_list_dtype = index_list.get("dtype").lower()
        self.data_shape = data.get("shape")
        self.cache_shape = cache.get("shape")
        self.cache_size = self.cache_shape[-1]
        self.cache_dtype = cache.get("dtype").lower()
        self.data_dtype = data.get("dtype").lower()
        self.tag_shape = tag.get("shape")
        self.tag_dtype = tag.get("dtype").lower()
        self.way_number = 64
        self.embedding_size = self.data_shape[-1]
        self.set_number = self.cache_size // self.embedding_size // self.way_number
        self.pre_route_number = pre_route_count
        self.kernel_name = kernel_name
        # input param check
        self.check_param()
        self.total_ub = tik.Dprofile().get_unified_buffer_size()
        self.aicore_num = 8  # only use 8 cores
        self.sets_list = []
        self.set_number_each_core = 0
        self.use_core_nums = 0
        # cal every core process sets
        self.every_core_process_sets()
        self.vector_mask_max_dict = {
            "float32": FP32_VECTOR_MASK_MAX,
            "float16": FP16_VECTOR_MASK_MAX,
            "int32": FP32_VECTOR_MASK_MAX,
            "int16": FP16_VECTOR_MASK_MAX,
            "uint8": INT8_VECTOR_MASK_MAX,
            "uint16": FP16_VECTOR_MASK_MAX,
        }
        self.index_list_ele_per_block = self._get_ele_block(self.index_list_dtype)
        self.tag_ele_per_block = self._get_ele_block(self.tag_dtype)
        self.index_list_len = int(functools.reduce(lambda x, y: x * y, self.index_list_shape))
        self.sorted_index_shape = self.get_least_multiple(self.way_number, 32)
        self.sorted_index_blocks = int(self.sorted_index_shape / 8)
        self.cache_exchange_shape = self.get_least_multiple(self.embedding_size, self._get_ele_block(self.cache_dtype))

        # Define input and output gm
        self.index_list_gm = self.tik_instance.Tensor(self.index_list_dtype,
                                                      self.index_list_shape,
                                                      name="index_list_gm",
                                                      scope=tik.scope_gm)
        self.tag_gm = self.tik_instance.Tensor(self.tag_dtype, self.tag_shape, name="tag_gm", scope=tik.scope_gm)
        self.tag_out_gm = self.tik_instance.Tensor(self.tag_dtype,
                                                   self.tag_shape,
                                                   name="tag_out_gm",
                                                   scope=tik.scope_gm)
        self.cache_gm = self.tik_instance.Tensor(self.cache_dtype, (self.cache_size,),
                                                 name="cache_gm",
                                                 scope=tik.scope_gm)
        self.cache_out_gm = self.tik_instance.Tensor(self.cache_dtype, (self.cache_size,),
                                                     name="cache_out_gm",
                                                     scope=tik.scope_gm)
        self.data_gm = self.tik_instance.Tensor(self.data_dtype, self.data_shape, name="data_gm", scope=tik.scope_gm)
        self.data_out_gm = self.tik_instance.Tensor(self.data_dtype,
                                                    self.data_shape,
                                                    name="data_out_gm",
                                                    scope=tik.scope_gm)
        self.index_offset_gm = self.tik_instance.Tensor(self.index_list_dtype,
                                                        self.index_list_shape,
                                                        name="index_offset_gm",
                                                        scope=tik.scope_gm)
        self.not_in_cache_index_list_gm = self.tik_instance.Tensor(self.index_list_dtype,
                                                                   self.index_list_shape,
                                                                   name="not_in_cache_index_list_gm",
                                                                   scope=tik.scope_gm)
        self.not_in_cache_count_gm = self.tik_instance.Tensor("int32", (8,),
                                                              name="not_in_cache_count_gm",
                                                              scope=tik.scope_gm,
                                                              is_atomic_add=True)
        self.miss_index_wsp = self.tik_instance.Tensor(self.tag_dtype, [self.set_number, 2 + self.index_list_len],
                                                       name="miss_index_wsp",
                                                       is_workspace=True,
                                                       scope=tik.scope_gm)
        sorted_index_init_list = [i for i in range(0, self.sorted_index_shape)]
        self.sorted_index_gm = self.tik_instance.Tensor("uint32", [
            self.sorted_index_shape,
        ],
                                                        name="sorted_index_gm",
                                                        scope=tik.scope_gm,
                                                        init_value=sorted_index_init_list)
        self.time_stamp_wsp = self.tik_instance.Tensor(self.tag_dtype, [self.set_number, self.way_number],
                                                       name="time_stamp_wsp",
                                                       is_workspace=True,
                                                       is_atomic_add=True,
                                                       scope=tik.scope_gm)
        self.sync_wsp = self.tik_instance.Tensor("int64", [
            self.use_core_nums * 4,
        ],
                                                 name="sync_wsp",
                                                 is_workspace=True,
                                                 is_atomic_add=True,
                                                 scope=tik.scope_gm)
        # The number of times that each core data cycle is moved in
        self.index_list_len_every_loop = 1024
        self.index_list_loop_times = 0
        self.index_list_len_last_loop = 0
        self.tag_loop_times = 0
        # The blocks that need to be calculated in each cycle
        self.one_set_size = self.way_number * tbe_platform.get_bit_len(self.tag_dtype) // 8
        # The blocks to be calculated in the last loop
        self.index_list_size = self.index_list_len_every_loop * tbe_platform.get_bit_len(self.index_list_dtype) // 8
        self.remain_ub_size = self.total_ub - self.index_list_size
        self.tag_set_nums_per_loop = 0
        self.way_number_32b_align = self.get_least_multiple(self.way_number, self._get_ele_block(self.tag_dtype))
        self.way_number_vector_compute_align = self.get_least_multiple(self.way_number,
                                                                       self.vector_mask_max_dict.get(self.tag_dtype))
        self.vcmp_out_uint8_blocks = math.ceil(self.way_number_vector_compute_align / 8 / 32)
        self.vcmax_result_shape = math.ceil(self.vcmp_out_uint8_blocks * 2 / 8 * 2 / 16) * 16
        # The number of elements to be calculated in each loop
        self.vector_mask_max = 0
        self.blk_stride = 0
        self.dst_rep_stride = 0
        self.src_rep_stride = 0
        self.index_vand_ub = None
        self.vcmax_scalar_max = None
        self.vcmax_scalar_cnt = None
        self.vcmax_scalar_index = None
        self.not_in_cache_count_ub = None
        self.index_offset_ub = None

    def _get_ele_block(self, dtype):
        """
        get this dtype block num
        """
        return 32 // (tbe_platform.get_bit_len(dtype) // 8)

    def ub_scalar_init(self):
        """
        ub_init
        """
        # index_list_len_32B_align = 1024
        self.index_list_ub = self.tik_instance.Tensor(self.index_list_dtype, [
            self.index_list_len_every_loop,
        ],
                                                      name="index_list_ub",
                                                      scope=tik.scope_ubuf)
        self.index_set_ub = self.tik_instance.Tensor(self.index_list_dtype, [
            self.index_list_len_every_loop,
        ],
                                                     name="index_set_ub",
                                                     scope=tik.scope_ubuf)
        self.index_vand_ub = self.tik_instance.Tensor(self.index_list_dtype, [
            self.index_list_len_every_loop,
        ],
                                                      name="index_vand_ub",
                                                      scope=tik.scope_ubuf)
        # ? 32B align [self.way_number]
        self.tag_ub = self.tik_instance.Tensor(self.tag_dtype, [
            self.way_number_vector_compute_align,
        ],
                                               name="tag_ub",
                                               scope=tik.scope_ubuf)
        self.timestamp_ub = self.tik_instance.Tensor(self.tag_dtype, [
            self.sorted_index_shape,
        ],
                                                     name="timestamp_ub",
                                                     scope=tik.scope_ubuf)
        self.timestamp_ub_fp32 = self.tik_instance.Tensor("float32", [self.sorted_index_shape],
                                                          name="timestamp_ub_fp32",
                                                          scope=tik.scope_ubuf)
        self.sorted_index_ub = self.tik_instance.Tensor("uint32", [
            self.sorted_index_shape,
        ],
                                                        name="sorted_index_ub",
                                                        scope=tik.scope_ubuf)
        self.vsort_result_ub_fp32 = self.tik_instance.Tensor("float32", [self.sorted_index_shape * 2],
                                                             name="vsort_result_ub_fp32",
                                                             scope=tik.scope_ubuf)
        self.cache_exchange_ub = self.tik_instance.Tensor(self.cache_dtype, [self.cache_exchange_shape],
                                                          name="cache_exchange_ub",
                                                          scope=tik.scope_ubuf)
        self.not_in_cache_count_ub = self.tik_instance.Tensor("int32", [
            8,
        ],
                                                              name="not_in_cache_count_ub",
                                                              scope=tik.scope_ubuf)
        self.index_offset_ub = self.tik_instance.Tensor("int32", [
            8,
        ], name="index_offset_ub", scope=tik.scope_ubuf)
        self.count_ub_for_wsp = self.tik_instance.Tensor("int32", [
            8,
        ], name="count_ub_for_wsp", scope=tik.scope_ubuf)
        self.vcmp_out_uint8 = self.tik_instance.Tensor("uint8", [self.vcmp_out_uint8_blocks * 32],
                                                       name="vcmp_out_uint8",
                                                       scope=tik.scope_ubuf)
        self.vcmp_out_fp16 = self.tik_instance.Tensor("float16", [self.vcmp_out_uint8_blocks * 32],
                                                      name="vcmp_out_fp16",
                                                      scope=tik.scope_ubuf)
        self.vcmax_result = self.tik_instance.Tensor("float16", [
            self.vcmax_result_shape,
        ],
                                                     name="vcmax_result",
                                                     scope=tik.scope_ubuf)

        # scalar init
        self.index_set_scalar = self.tik_instance.Scalar(dtype="int32", name="index_set_scalar")
        self.index_scalar = self.tik_instance.Scalar(dtype="int32", name="index_scalar")
        self.this_set_number = self.tik_instance.Scalar(dtype="int32", name="this_set_number")
        self.vcmax_scalar_max = self.tik_instance.Scalar(dtype="float16", name="vcmax_scalar_max", init_value=0.0)
        self.vcmax_scalar_max_fp32 = self.tik_instance.Scalar(dtype="float32", name="vcmax_scalar_max_fp32")
        self.vcmax_scalar_max_int32 = self.tik_instance.Scalar(dtype="int32", name="vcmax_scalar_max_int32")
        self.vcmax_scalar_cnt = self.tik_instance.Scalar(dtype="int32", name="vcmax_scalar_cnt", init_value=0)
        self.vcmax_scalar_index = self.tik_instance.Scalar(dtype="uint16", name="vcmax_scalar_index", init_value=0)
        self.vcmax_scalar_index_offset = self.tik_instance.Scalar(dtype="int32", name="vcmax_scalar_index_offset")
        self.index_offset_scalar = self.tik_instance.Scalar(dtype="int32", name="index_offset_scalar")
        self.scalar_neg_1 = self.tik_instance.Scalar(dtype="int32", name="scalar_neg_1", init_value=-1)
        self.scalar_int1 = self.tik_instance.Scalar(dtype="int32", name="scalar_int1", init_value=1)
        self.scalar_int0 = self.tik_instance.Scalar(dtype="int32", name="scalar_int0", init_value=0)
        self.max_timesp_index = self.tik_instance.Scalar(dtype="uint32", name="max_timesp_index")
        self.max_timesp_value = self.tik_instance.Scalar(dtype="int32", name="max_timesp_value")
        self.min_timesp_index = self.tik_instance.Scalar(dtype="uint32", name="max_timesp_index")
        self.exchange_tag_index = self.tik_instance.Scalar(dtype="int32", name="exchange_tag_index")
        self.cache_in_way_index = self.tik_instance.Scalar(dtype="int32", name="cache_in_way_index")
        self.miss_index_offset_wsp = self.tik_instance.Scalar(dtype="int32", name="miss_index_offset_wsp", init_value=1)
        self.cache_count = self.tik_instance.Scalar(dtype="int32", name="cache_count", init_value=0)
        self.miss_count = self.tik_instance.Scalar(dtype="int32", name="miss_count", init_value=0)
        self.exchange_nums = self.tik_instance.Scalar(dtype="int32", name="exchange_nums")
        self.last_timestamp_scalar = self.tik_instance.Scalar(dtype="int32", name="now_timestamp")
        self.miss_index = self.tik_instance.Scalar(dtype="int32", name="miss_index")
        self.fp32_max_scalar = self.tik_instance.Scalar(dtype="float32", name="fp32_max", init_value=FP32_MAX)

    def every_core_process_sets(self):
        """
        every_core_process_sets
        """
        self.set_number_each_core = math.ceil(self.set_number / self.aicore_num)
        self.use_core_nums = math.ceil(self.set_number / self.set_number_each_core)
        for i in range(0, self.use_core_nums - 1):
            self.sets_list.append(
                [x for x in range(i * self.set_number_each_core, (i + 1) * self.set_number_each_core)])
        self.sets_list.append([x for x in range((self.use_core_nums - 1) * self.set_number_each_core, self.set_number)])

    def get_least_multiple(self, a, b):
        """
        get_least_multiple
        """
        x = math.ceil(a / b) * b

        return x

    def cal_data_mov_loops(self, tag_sets):
        """
        cal_data_mov_loops
        """
        # Calculate the number of cycles required
        self.index_list_loop_times = math.ceil(self.index_list_len / self.index_list_len_every_loop)
        self.index_list_len_last_loop = self.index_list_len - \
                                        (self.index_list_loop_times-1) * self.index_list_len_every_loop
        self.tag_set_nums_per_loop = 1
        self.tag_loop_times = tag_sets

    def vector_compute(self, dst_ub, src_ub0, compute_type, compute_blocks, src_ub1=None, scalar=None):
        """
        vector_compute
        """
        self.get_vector_mask_max(dst_ub, src_ub0)
        compute_ele_nums = int(self._get_ele_block(src_ub0.dtype) * compute_blocks)
        if compute_type == "vsort32":
            self.vector_mask_max = 32
        compute_instr_loops = compute_ele_nums // (self.vector_mask_max * REPEAT_TIMES_MAX)
        compute_offset = 0
        if compute_instr_loops > 0:
            with self.tik_instance.for_range(0, compute_instr_loops) as \
                    instr_loops_index:
                compute_offset = instr_loops_index * self.vector_mask_max * REPEAT_TIMES_MAX * self.blk_stride
                if compute_type == "vconv":
                    self.vconv_instr_gen(self.vector_mask_max, compute_offset, dst_ub, src_ub0, REPEAT_TIMES_MAX)
                if compute_type in ["vand", "vreduce", "vcmpv_eq", "vsort32"]:
                    self.double_in_instr_gen(self.vector_mask_max, compute_offset, dst_ub, src_ub0, src_ub1,
                                             REPEAT_TIMES_MAX, compute_type)
                if compute_type in ["vshr", "vshl", "vector_dup", "vcmpvs_eq", "vcmax"]:
                    self.tensor_scalar_instr_gen(self.vector_mask_max, compute_offset, dst_ub, src_ub0,
                                                 REPEAT_TIMES_MAX, scalar, compute_type)
            compute_offset = compute_instr_loops * self.vector_mask_max * REPEAT_TIMES_MAX * self.blk_stride
        repeat_time = (compute_ele_nums % (self.vector_mask_max * REPEAT_TIMES_MAX) // self.vector_mask_max)
        if repeat_time > 0:
            if compute_type == "vconv":
                self.vconv_instr_gen(self.vector_mask_max, compute_offset, dst_ub, src_ub0, repeat_time)
            if compute_type in ["vand", "vreduce", "vcmpv_eq", "vsort32"]:
                self.double_in_instr_gen(self.vector_mask_max, compute_offset, dst_ub, src_ub0, src_ub1, repeat_time,
                                         compute_type)
            if compute_type in ["vshr", "vshl", "vector_dup", "vcmpvs_eq", "vcmax"]:
                self.tensor_scalar_instr_gen(self.vector_mask_max, compute_offset, dst_ub, src_ub0, repeat_time, scalar,
                                             compute_type)
            compute_offset = compute_offset + repeat_time * self.vector_mask_max * self.blk_stride
        last_num = compute_ele_nums % self.vector_mask_max
        if last_num > 0:
            if compute_type == "vconv":
                self.vconv_instr_gen(last_num, compute_offset, dst_ub, src_ub0, REMAIN_REPEAT)
            if compute_type in ["vand", "vreduce", "vcmpv_eq", "vsort32"]:
                self.double_in_instr_gen(last_num, compute_offset, dst_ub, src_ub0, src_ub1, REMAIN_REPEAT,
                                         compute_type)
            if compute_type in ["vshr", "vshl", "vector_dup", "vcmpvs_eq", "vcmax"]:
                self.tensor_scalar_instr_gen(last_num, compute_offset, dst_ub, src_ub0, REMAIN_REPEAT, scalar,
                                             compute_type)

    def vconv_instr_gen(self, mask, offset, dst_ub, src_ub0, repeat_times):
        """
        vconv_instr_gen
        """
        if src_ub0.dtype == "uint8" and dst_ub.dtype == "float16":
            return self.tik_instance.vconv(mask, "", dst_ub[offset], src_ub0[offset], repeat_times, self.blk_stride,
                                           self.blk_stride, self.dst_rep_stride, self.src_rep_stride)
        if src_ub0.dtype == "int32" and dst_ub.dtype == "float32":
            return self.tik_instance.vconv(mask, "", dst_ub[offset], src_ub0[offset], repeat_times, self.blk_stride,
                                           self.blk_stride, self.dst_rep_stride, self.src_rep_stride)

    def double_in_instr_gen(self, mask, offset, dst_ub, src_ub0, src_ub1, repeat_times, compute_type):
        """
        double_in_instr_gen
        """
        tik_fun = None
        if compute_type == "vand":
            tik_fun = self.tik_instance.vand
            return tik_fun(
                mask,
                dst_ub[offset],
                src_ub0[offset],
                src_ub1[offset],
                repeat_times,
                self.blk_stride,
                self.blk_stride,
                self.blk_stride,
                self.dst_rep_stride,
                self.src_rep_stride,
                self.src_rep_stride,
            )
        if compute_type == "vreduce":
            tik_fun = self.tik_instance.vreduce
            src0_blk_stride = self.blk_stride
            src0_rep_stride = self.src_rep_stride
            src1_rep_stride = 0
            if repeat_times == 1:
                mask_mode = "counter"
            else:
                mask_mode = "normal"
            det_ub_offset = offset // 2
            return tik_fun(mask,
                           dst_ub[det_ub_offset],
                           src_ub0[offset],
                           src_ub1,
                           repeat_times,
                           src0_blk_stride,
                           src0_rep_stride,
                           src1_rep_stride,
                           mask_mode=mask_mode)
        if compute_type == "vcmpv_eq":
            tik_fun = self.tik_instance.vcmpv_eq
            return tik_fun(
                dst_ub[offset],
                src_ub0[offset],
                src_ub1[offset],
                repeat_times,
                self.blk_stride,
                self.blk_stride,
                self.src_rep_stride,
                self.src_rep_stride,
            )
        if compute_type == "vsort32":
            tik_fun = self.tik_instance.vsort32
            det_ub_offset = offset * 2
            return tik_fun(dst_ub[det_ub_offset], src_ub0[offset], src_ub1[offset], repeat_times)

    def tensor_scalar_instr_gen(self, mask, offset, dst_ub, src_ub, repeat_times, scalar, compute_type):
        """
        tensor_scalar_instr_gen
        """
        tik_fun = None
        if compute_type == "vshr":
            tik_fun = self.tik_instance.vshr
        if compute_type == "vshl":
            tik_fun = self.tik_instance.vshl
        if compute_type == "vector_dup":
            return self.tik_instance.vector_dup(mask, dst_ub[offset], scalar, repeat_times, self.blk_stride,
                                                self.dst_rep_stride)
        if compute_type == "vcmax":
            return self.tik_instance.vcmax(
                mask,
                dst_ub[offset],
                src_ub[offset],
                repeat_times,
                self.blk_stride,
                self.blk_stride,
                self.src_rep_stride,
                max_cnt_index=[self.vcmax_scalar_max, self.vcmax_scalar_cnt, self.vcmax_scalar_index])

        if compute_type == "vcmpvs_eq":
            dst_ub_offset = offset // 8
            return self.tik_instance.vcmpvs_eq(dst_ub[dst_ub_offset], src_ub[offset], scalar, repeat_times,
                                               self.blk_stride, self.src_rep_stride)
        return tik_fun(mask, dst_ub[offset], src_ub[offset], scalar, repeat_times, self.blk_stride, self.blk_stride,
                       self.dst_rep_stride, self.src_rep_stride)

    def get_vector_mask_max(self, dst_ub, src_ub):
        """
        get_vector_mask_max
        """
        self.vector_mask_max = min(self.vector_mask_max_dict.get(dst_ub.dtype),
                                   self.vector_mask_max_dict.get(src_ub.dtype))
        self.blk_stride = SHAPE_SIZE_ONE
        self.dst_rep_stride = self.vector_mask_max // self._get_ele_block(dst_ub.dtype) * self.blk_stride
        self.src_rep_stride = self.vector_mask_max // self._get_ele_block(src_ub.dtype) * self.blk_stride

    def isPower(self, k):
        """
        input is or not 2**n
        """
        if k < 1:
            return False
        m = k & (k - 1)
        return m == 0

    def get_mod_vand_scalar(self):
        """
        get_mod_vand_scalar
        """
        n = int(math.log(self.set_number, 2))
        x = 0
        for i in range(0, n):
            x = x + 2**i
        return int(x)

    def vmrgsort_recursion(self, index_process_list):
        """
        vmrgsort_recursion
        """
        if len(index_process_list) > 4:
            first_ele = index_process_list[0]
            n_same = 0
            for i in range(0, len(index_process_list)):
                if first_ele == index_process_list[i]:
                    n_same = i + 1
                else:
                    break
            four_src_repeats = n_same // 4
            if four_src_repeats != 0:
                ele_count_list = [first_ele * 32 for i in range(0, 4)]
                self.tik_instance.vmrgsort(self.vsort_result_ub_fp32,
                                           (self.vsort_result_ub_fp32[0:ele_count_list[0] * 2],
                                            self.vsort_result_ub_fp32[ele_count_list[0] * 2:ele_count_list[0] * 4],
                                            self.vsort_result_ub_fp32[ele_count_list[0] * 4:ele_count_list[0] * 6],
                                            self.vsort_result_ub_fp32[ele_count_list[0] * 6:ele_count_list[0] * 8]),
                                           ele_count_list, False, four_src_repeats)
                del index_process_list[0:four_src_repeats * 4]
                new_index_process_list = [4 * first_ele for i in range(0, four_src_repeats)] + index_process_list
                self.vmrgsort_recursion(new_index_process_list)
            else:
                ele_count_list = [
                    index_process_list[0] * 32, index_process_list[1] * 32, index_process_list[2] * 32,
                    index_process_list[3] * 32
                ]
                self.tik_instance.vmrgsort(
                    self.vsort_result_ub_fp32,
                    (self.vsort_result_ub_fp32[0:ele_count_list[0] * 2],
                     self.vsort_result_ub_fp32[ele_count_list[0] * 2:ele_count_list[0] * 2 + ele_count_list[1] * 2],
                     self.vsort_result_ub_fp32[ele_count_list[0] * 2 + ele_count_list[1] * 2:ele_count_list[0] * 2 +
                                               ele_count_list[1] * 2 + ele_count_list[2] * 2],
                     self.vsort_result_ub_fp32[ele_count_list[0] * 2 + ele_count_list[1] * 2 +
                                               ele_count_list[2] * 2:ele_count_list[0] * 2 + ele_count_list[1] * 2 +
                                               ele_count_list[2] * 2 + ele_count_list[3] * 2]), ele_count_list, False,
                    1)
                ori_index_process_list = [
                    index_process_list[0] + index_process_list[1] + index_process_list[2] + index_process_list[3]
                ]
                del index_process_list[0:4]
                new_index_process_list = ori_index_process_list + index_process_list
                self.vmrgsort_recursion(new_index_process_list)
        # singe repeat
        else:
            if len(index_process_list) == 4:
                ele_count_list = [
                    index_process_list[0] * 32, index_process_list[1] * 32, index_process_list[2] * 32,
                    index_process_list[3] * 32
                ]
                self.tik_instance.vmrgsort(
                    self.vsort_result_ub_fp32,
                    (self.vsort_result_ub_fp32[0:ele_count_list[0] * 2],
                     self.vsort_result_ub_fp32[ele_count_list[0] * 2:ele_count_list[0] * 2 + ele_count_list[1] * 2],
                     self.vsort_result_ub_fp32[ele_count_list[0] * 2 + ele_count_list[1] * 2:ele_count_list[0] * 2 +
                                               ele_count_list[1] * 2 + ele_count_list[2] * 2],
                     self.vsort_result_ub_fp32[ele_count_list[0] * 2 + ele_count_list[1] * 2 +
                                               ele_count_list[2] * 2:ele_count_list[0] * 2 + ele_count_list[1] * 2 +
                                               ele_count_list[2] * 2] + ele_count_list[3] * 2), ele_count_list, False,
                    1)
            elif len(index_process_list) == 3:
                ele_count_list = [index_process_list[0] * 32, index_process_list[1] * 32, index_process_list[2] * 32]
                self.tik_instance.vmrgsort(
                    self.vsort_result_ub_fp32,
                    (self.vsort_result_ub_fp32[0:ele_count_list[0] * 2],
                     self.vsort_result_ub_fp32[ele_count_list[0] * 2:ele_count_list[0] * 2 + ele_count_list[1] * 2],
                     self.vsort_result_ub_fp32[ele_count_list[0] * 2 + ele_count_list[1] * 2:ele_count_list[0] * 2 +
                                               ele_count_list[1] * 2 + ele_count_list[2] * 2]), ele_count_list, False,
                    1)
            elif len(index_process_list) == 2:
                ele_count_list = [index_process_list[0] * 32, index_process_list[1] * 32]
                self.tik_instance.vmrgsort(
                    self.vsort_result_ub_fp32,
                    (self.vsort_result_ub_fp32[0:ele_count_list[0] * 2],
                     self.vsort_result_ub_fp32[ele_count_list[0] * 2:ele_count_list[0] * 2 + ele_count_list[1] * 2]),
                    ele_count_list, False, 1)
            else:
                pass

    def sort_index(self, index, score, result, actual_shape):
        """
        sort index
        """
        index_len = index.shape[0]
        index_process_list = [1 for i in range(0, index_len // 32)]
        set_max_value_nums = index_len - actual_shape
        if set_max_value_nums != 0:
            for i in range(0, set_max_value_nums):
                score[actual_shape + i].set_as(self.fp32_max_scalar)
        vsort_blocks = index_len // 8
        self.vector_compute(result, score, "vsort32", vsort_blocks, src_ub1=index)
        self.vmrgsort_recursion(index_process_list)

    def lru_compute(self):
        """
        lru_bboxv2_compute
        """
        with self.tik_instance.for_range(0, self.use_core_nums, block_num=self.use_core_nums) as core_id:
            tag_gm_offset = core_id * (self.set_number_each_core * self.way_number)
            if self.set_number_each_core == len(self.sets_list[-1]):
                self.lru_compute_each_core(tag_gm_offset, self.set_number_each_core, core_id)
            else:
                with self.tik_instance.if_scope(core_id < (self.use_core_nums - 1)):
                    self.lru_compute_each_core(tag_gm_offset, self.set_number_each_core, core_id)
                with self.tik_instance.else_scope():
                    self.lru_compute_each_core(tag_gm_offset, len(self.sets_list[-1]), core_id)
            self.tik_instance.block_barrier(self.sync_wsp)
            with self.tik_instance.if_scope(core_id == 0):
                not_in_cache_index_list_ub = self.index_list_ub
                move_offset = self.exchange_nums
                move_offset.set_as(0)
                for n in range(0, self.set_number):
                    self.tik_instance.data_move_pad(not_in_cache_index_list_ub,
                                                    self.miss_index_wsp[n * (2 + self.index_list_len)], 1, 8, 0, 0)
                    self.cache_count.set_as(not_in_cache_index_list_ub[0])
                    self.miss_count.set_as(not_in_cache_index_list_ub[1])
                    with self.tik_instance.if_scope((self.cache_count + self.miss_count) > self.way_number):
                        self.tik_instance.data_move_pad(
                            not_in_cache_index_list_ub,
                            self.miss_index_wsp[n * (2 + self.index_list_len) + self.cache_count + 2], 1,
                            4 * (self.cache_count + self.miss_count - self.way_number), 0, 0)
                        self.tik_instance.data_move_pad(self.not_in_cache_index_list_gm[move_offset],
                                                        not_in_cache_index_list_ub, 1,
                                                        4 * (self.cache_count + self.miss_count - self.way_number), 0,
                                                        0)
                        move_offset.set_as(move_offset + self.miss_count - self.way_number)
        # this "global_variable_link" flag suggest ccec.py do link without "-r" option
        # which will result in global variable in cce file with wrong address
        tbe_context.get_context().add_compile_info("global_variable_link", True)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.index_list_gm, self.data_gm, self.cache_gm, self.tag_gm],
                                   outputs=[
                                       self.data_out_gm, self.cache_out_gm, self.tag_out_gm, self.index_offset_gm,
                                       self.not_in_cache_index_list_gm, self.not_in_cache_count_gm
                                   ])

        return self.tik_instance

    def lru_compute_each_core(self, tag_core_offset, tag_sets, core_id):
        """
        lru_compute_each_core
        """
        # index_list for loop
        self.cal_data_mov_loops(tag_sets)
        self.ub_scalar_init()
        with self.tik_instance.for_range(0, self.tag_loop_times) as j:
            tag_loop_offset = tag_core_offset + j * self.tag_set_nums_per_loop * self.way_number
            self.tag_process_each_loop(tag_loop_offset)
            self.this_set_number.set_as(core_id * self.set_number_each_core + j)
            with self.tik_instance.for_range(0, self.index_list_loop_times) as i:
                index_list_loop_offset = i * self.index_list_len_every_loop
                with self.tik_instance.if_scope(i < (self.index_list_loop_times - 1)):
                    self.index_list_process_each_loop(index_list_loop_offset, self.index_list_len_every_loop)
                with self.tik_instance.else_scope():
                    self.index_list_process_each_loop(index_list_loop_offset, self.index_list_len_last_loop)
            self.tag_after_process_each_loop(core_id, j)

    def tag_process_each_loop(self, tag_offset):
        """
        tag_process_each_loop
        """
        tag_blocks = math.ceil(self.tag_set_nums_per_loop * self.way_number / self.tag_ele_per_block)
        self.tik_instance.data_move(self.tag_ub, self.tag_gm[tag_offset], SMMU_ID, DATA_MOV_NBURST, tag_blocks,
                                    DATA_MOV_STRIDE, DATA_MOV_STRIDE)
        self.tik_instance.data_move(self.timestamp_ub, self.time_stamp_wsp[tag_offset], SMMU_ID, DATA_MOV_NBURST,
                                    tag_blocks, DATA_MOV_STRIDE, DATA_MOV_STRIDE)
        self.tik_instance.data_move(self.sorted_index_ub, self.sorted_index_gm, SMMU_ID, DATA_MOV_NBURST,
                                    self.sorted_index_blocks, DATA_MOV_STRIDE, DATA_MOV_STRIDE)

    def index_list_process_each_loop(self, index_list_offset, index_list_len):
        """
        tag_set_nums : actual tag sets
        """
        index_list_blocks = math.ceil(index_list_len / self.index_list_ele_per_block)

        self.tik_instance.data_move(self.index_list_ub, self.index_list_gm[index_list_offset], SMMU_ID, DATA_MOV_NBURST,
                                    index_list_blocks, DATA_MOV_STRIDE, DATA_MOV_STRIDE)
        if self.index_list_dtype in ["int32", "uint32"]:
            self.vector_compute(self.index_set_ub,
                                self.index_list_ub,
                                "vshr",
                                index_list_blocks,
                                scalar=int(math.log(self.pre_route_number, 2)))
            self.vector_compute(self.index_vand_ub,
                                self.index_vand_ub,
                                "vector_dup",
                                index_list_blocks,
                                scalar=self.get_mod_vand_scalar())
            index_set_ub_int16 = self.index_set_ub.reinterpret_cast_to("int16")
            index_vand_ub_int16 = self.index_vand_ub.reinterpret_cast_to("int16")
            self.vector_compute(index_set_ub_int16,
                                index_set_ub_int16,
                                "vand",
                                index_list_blocks,
                                src_ub1=index_vand_ub_int16)
            index_set_ub_int32 = index_set_ub_int16.reinterpret_cast_to(self.index_list_dtype)
        # index for loop
        with self.tik_instance.for_range(0, index_list_len) as index_id:
            self.index_set_scalar.set_as(index_set_ub_int32[index_id])
            with self.tik_instance.if_scope(self.index_set_scalar == self.this_set_number):
                vcmp_out_uint16 = self.vcmp_out_uint8.reinterpret_cast_to("uint16")
                self.vector_compute(vcmp_out_uint16,
                                    vcmp_out_uint16,
                                    "vector_dup",
                                    self.vcmp_out_uint8_blocks,
                                    scalar=0)
                vcmp_out_uint8_all_zero = vcmp_out_uint16.reinterpret_cast_to("uint8")
                self.index_scalar.set_as(self.index_list_ub[index_id])
                self.vector_compute(vcmp_out_uint8_all_zero,
                                    self.tag_ub,
                                    "vcmpvs_eq",
                                    math.ceil(self.way_number_vector_compute_align / self.tag_ele_per_block),
                                    scalar=self.index_scalar)
                self.vector_compute(self.vcmp_out_fp16, vcmp_out_uint8_all_zero, "vconv", self.vcmp_out_uint8_blocks)
                self.tik_instance.vcmax(32, self.vcmax_result, self.vcmp_out_fp16, 1, 1, 1, 8)
                self.vcmax_scalar_index.set_as(self.vcmax_result[1].reinterpret_cast_to("uint16"))
                self.vcmax_scalar_max.set_as(self.vcmax_result[0])
                self.tik_instance.scalar_conv('', self.vcmax_scalar_max_fp32, self.vcmax_scalar_max)
                self.tik_instance.scalar_conv('round', self.vcmax_scalar_max_int32, self.vcmax_scalar_max_fp32)
                with self.tik_instance.if_scope(self.vcmax_scalar_max_int32 == 0):
                    self.miss_count.set_as(self.miss_count + 1)
                    self.not_in_cache_count_ub.set_as(self.scalar_int1)
                    self.count_ub_for_wsp[0].set_as(self.miss_count)
                    self.index_offset_ub[0].set_as(self.scalar_neg_1)
                    self.tik_instance.data_move_pad(self.index_offset_gm[index_list_offset + index_id],
                                                    self.index_offset_ub, 1, 4, 0, 0)
                    self.tik_instance.data_move_pad(self.miss_index_wsp[self.this_set_number, 1], self.count_ub_for_wsp,
                                                    1, 4, 0, 0)
                    # ub reuse for miss_index
                    self.index_offset_ub[0].set_as(self.index_list_ub[index_id])
                    self.tik_instance.data_move_pad(self.miss_index_wsp[self.this_set_number, self.miss_count + 1],
                                                    self.index_offset_ub, 1, 4, 0, 0)
                    self.tik_instance.set_atomic_add(4)
                    self.tik_instance.data_move_pad(self.not_in_cache_count_gm, self.not_in_cache_count_ub, 1, 4, 0, 0)

                    self.tik_instance.set_atomic_add(0)
                with self.tik_instance.else_scope():
                    self.cache_count.set_as(self.cache_count + 1)
                    for m in range(0, 8):
                        with self.tik_instance.if_scope(self.vcmax_scalar_max_int32 == 2**m):
                            self.vcmax_scalar_index_offset.set_as(m)
                    self.count_ub_for_wsp[0].set_as(self.cache_count)
                    self.cache_in_way_index.set_as(self.vcmax_scalar_index * 8 + self.vcmax_scalar_index_offset)
                    self.index_offset_scalar.set_as(
                        (self.cache_in_way_index + self.this_set_number * self.way_number) * self.embedding_size)
                    self.last_timestamp_scalar.set_as(self.timestamp_ub[self.cache_in_way_index])
                    self.timestamp_ub[self.cache_in_way_index].set_as(self.last_timestamp_scalar + 1)
                    self.index_offset_ub[0].set_as(self.index_offset_scalar)
                    self.tik_instance.data_move_pad(self.index_offset_gm[index_list_offset + index_id],
                                                    self.index_offset_ub, 1, 4, 0, 0)
                    self.tik_instance.data_move_pad(self.miss_index_wsp[self.this_set_number, 0], self.count_ub_for_wsp,
                                                    1, 4, 0, 0)

    def tag_after_process_each_loop(self, core_id, tag_loop_id):
        """
        tag_after_process_each_loop
        """
        #  sort time_stamp
        self.vector_compute(self.timestamp_ub_fp32, self.timestamp_ub, "vconv", self.sorted_index_blocks)
        self.sort_index(self.sorted_index_ub, self.timestamp_ub_fp32, self.vsort_result_ub_fp32, self.way_number)
        with self.tik_instance.if_scope(self.cache_count + self.miss_count <= self.way_number):
            self.exchange_nums.set_as(self.miss_count)
        with self.tik_instance.if_scope(self.cache_count + self.miss_count > self.way_number):
            self.exchange_nums.set_as(self.way_number - self.cache_count)
        with self.tik_instance.for_range(0, self.exchange_nums) as k:
            # get miss index
            get_miss_index_ub = self.index_offset_ub
            self.tik_instance.data_move_pad(get_miss_index_ub, self.miss_index_wsp[self.this_set_number, k + 2], 1, 4,
                                            0, 0)
            self.miss_index.set_as(get_miss_index_ub[0])
            self.min_timesp_index.set_as(self.vsort_result_ub_fp32[(self.sorted_index_shape - k) * 2 -
                                                                   1].reinterpret_cast_to("uint32"))
            cache_embedding_bursts = self.embedding_size * tbe_platform.get_bit_len(self.cache_dtype) // 8
            self.tik_instance.data_move_pad(
                self.cache_exchange_ub,
                self.cache_gm[(self.this_set_number * self.way_number + self.min_timesp_index) * self.embedding_size],
                1, cache_embedding_bursts, 0, 0)
            self.exchange_tag_index.set_as(self.tag_ub[self.min_timesp_index])
            self.tik_instance.data_move_pad(self.data_out_gm[self.exchange_tag_index * self.embedding_size],
                                            self.cache_exchange_ub, 1, cache_embedding_bursts, 0, 0)
            self.tik_instance.data_move_pad(self.cache_exchange_ub, self.data_gm[self.miss_index * self.embedding_size],
                                            1, cache_embedding_bursts, 0, 0)
            self.tik_instance.data_move_pad(
                self.cache_out_gm[(self.this_set_number * self.way_number + self.min_timesp_index) *
                                  self.embedding_size], self.cache_exchange_ub, 1, cache_embedding_bursts, 0, 0)
            tag_refresh_ub = self.count_ub_for_wsp
            self.last_timestamp_scalar.set_as(self.timestamp_ub[self.min_timesp_index])
            tag_refresh_ub[0].set_as(get_miss_index_ub[0])
            self.tik_instance.data_move_pad(
                self.tag_out_gm[((self.this_set_number * self.way_number) + self.min_timesp_index)], tag_refresh_ub, 1,
                4, 0, 0)
            tag_refresh_ub[0].set_as(self.last_timestamp_scalar + 1)
            self.tik_instance.data_move_pad(
                self.time_stamp_wsp[((self.this_set_number * self.way_number) + self.min_timesp_index)], tag_refresh_ub,
                1, 4, 0, 0)
        # 2 scalar init 0
        self.cache_count.set_as(0)
        self.miss_count.set_as(0)

    def check_param(self):
        """
        check_param
        """
        para_check.check_shape(self.index_list_shape, min_rank=1, max_rank=1, param_name="index_list")
        para_check.check_shape(self.data_shape, min_rank=2, max_rank=2, param_name="data")
        para_check.check_shape(self.tag_shape, min_rank=1, max_rank=1, param_name="tag")
        para_check.check_shape(self.cache_shape, min_rank=1, max_rank=1, param_name="cahce")

        if self.data_shape[-1] != self.embedding_size:
            error_info = {}
            error_info['errCode'] = 'E80000'
            error_info['op_name'] = 'lru'
            error_info['param_name'] = "lru's data input last dim"
            error_info['expected_value'] = self.embedding_size
            error_info['real_value'] = self.data_shape[-1]
            raise RuntimeError(
                error_info, "In op[{op_name}], the parameter[{param_name}] "
                "should be [{expected_value}], "
                "but actually is [{real_value}].".format(**error_info))
        if self.tag_shape[0] != self.set_number * self.way_number:
            error_info = {}
            error_info['errCode'] = 'E80000'
            error_info['op_name'] = 'lru'
            error_info['param_name'] = "lru's tag size"
            error_info['expected_value'] = self.set_number * self.way_number
            error_info['real_value'] = self.tag_shape[0]
            raise RuntimeError(
                error_info, "In op[{op_name}], the parameter[{param_name}] "
                "should be [{expected_value}], "
                "but actually is [{real_value}].".format(**error_info))
        if self.cache_shape[0] != self.tag_shape[0] * self.embedding_size:
            error_info = {}
            error_info['errCode'] = 'E80000'
            error_info['op_name'] = 'lru'
            error_info['param_name'] = "lru's cahce size"
            error_info['expected_value'] = self.tag_shape[0] * self.embedding_size
            error_info['real_value'] = self.cache_shape[0]
            raise RuntimeError(
                error_info, "In op[{op_name}], the parameter[{param_name}] "
                "should be [{expected_value}], "
                "but actually is [{real_value}].".format(**error_info))
        check_list_input_list = ["int32"]
        check_list_data = ["float16", "float32"]
        para_check.check_dtype(self.index_list_dtype, check_list_input_list, param_name="input_list")
        para_check.check_dtype(self.tag_dtype, check_list_input_list, param_name="tag")
        para_check.check_dtype(self.cache_dtype, check_list_data, param_name="cache")
        para_check.check_dtype(self.data_dtype, check_list_data, param_name="data")
        if not self.isPower(self.set_number):
            error_info = {}
            error_info['errCode'] = 'E80000'
            error_info['op_name'] = 'lru'
            error_info['param_name'] = "set_number"
            error_info['expected_value'] = "2**n"
            error_info['real_value'] = self.set_number
            raise RuntimeError(
                error_info, "In op[{op_name}], the parameter[{param_name}] "
                "should be [{expected_value}], but actually "
                "is [{real_value}].".format(**error_info))
        if not self.isPower(self.pre_route_number):
            error_info = {}
            error_info['errCode'] = 'E80000'
            error_info['op_name'] = 'lru'
            error_info['param_name'] = "pre_route_number"
            error_info['expected_value'] = "2**n"
            error_info['real_value'] = self.pre_route_number
            raise RuntimeError(
                error_info, "In op[{op_name}], the parameter[{param_name}] "
                "should be [{expected_value}], but actually "
                "is [{real_value}].".format(**error_info))


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def lru_cache_v2(index_list,
                 data,
                 cache,
                 tag,
                 is_last_call,
                 out_data,
                 out_cache,
                 out_tag,
                 index_offset_list,
                 not_in_cache_index_list,
                 not_in_cache_number,
                 pre_route_count,
                 kernel_name="lru_cache_v2"):
    """
    index_list:exchange index list
    data:host data
    cache:gm cache
    tag:cache's tag
    is_last_call: if is last call write all cache to data
    out_data:output data
    out_cache:output gm cache
    out_tag:output cache's tag
    index_offset_list,
    not_in_cache_index_list,
    not_in_cache_number,
    pre_route_count,
    """
    lru_compute = LRU(index_list, data, cache, tag, is_last_call, out_data, out_cache, out_tag, index_offset_list,
                      not_in_cache_index_list, not_in_cache_number, pre_route_count, kernel_name)
    return lru_compute.lru_compute()
