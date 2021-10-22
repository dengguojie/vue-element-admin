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
sort
"""


from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl import common_util
from impl import constant_util

# max int32 value
MAX_INT32_VALUE = 2 ** 31 - 1
# workspace size
WORKSPACE_SIZE = 2 ** 30
# temp proposal gm cache
TEMP_PROPOSAL_SIZE = 2 ** 30
# tiling param num
TILING_ARG_NUM = 24
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# bytes of one block
BLOCK_BYTES = 32
# proposal num
PROPOSAL_NUM = 8
# max proposal num
MAX_PROPOSAL_NUM = 2048
# col_per_part num
COL_PAR_LEN = 3600
# col loop time
COL_LOOP_TIME = 5
# big shape 2048 align
NUM_BLOCK = 2048
# block
BLOCK = 16
# max repeat
REPEAT_MAX = 255
# repeat max
DATA_MAX = 4080
# min value for descend
MIN_VAL = -65504
# max value for ascend
MAX_VAL = 65504


# pylint: disable=too-many-branches,too-many-statements,too-many-locals
# pylint: disable=too-many-arguments
def tik_func_vector_dup(tik_instance, _ub, value, dup_len):
    """
    tik_func_vector

    Parameters:
    ----------
    tik_instance : tik_instance.
        tik_instance
    _ub: ub
        vector ub
    value: value
        vector value
    dup_len: int
        vector data len

    Returns
    -------
    None
    """
    do_dtype = _ub.dtype
    # get how many byte
    byte_num_one = common_util.get_data_size(do_dtype)
    block_num = constant_util.BLOCK_SIZE // byte_num_one
    vector_num = block_num * constant_util.REPEAT_STRIDE_EIGHT

    repeat_255_scalar = tik_instance.Scalar(init_value=(dup_len // (vector_num * REPEAT_MAX)))
    repeat_remain_scalar = tik_instance.Scalar(
        init_value=((dup_len - repeat_255_scalar * vector_num * REPEAT_MAX) // vector_num))

    repeat_tail = tik_instance.Scalar(
        init_value=(dup_len - vector_num * REPEAT_MAX * repeat_255_scalar - repeat_remain_scalar * vector_num))

    with tik_instance.if_scope(repeat_255_scalar > 0):
        with tik_instance.for_range(0, repeat_255_scalar, name='vadd_i0') as i:
            tik_instance.vector_dup(vector_num, _ub[vector_num * REPEAT_MAX * i], value, REPEAT_MAX, 1, 8)

    with tik_instance.if_scope(repeat_remain_scalar > 0):
        tik_instance.vector_dup(vector_num, _ub[vector_num * REPEAT_MAX * repeat_255_scalar], value,
                                repeat_remain_scalar, 1, 8)
    with tik_instance.if_scope(repeat_tail > 0):
        offset = vector_num * REPEAT_MAX * repeat_255_scalar + repeat_remain_scalar * vector_num
        tik_instance.vector_dup(repeat_tail, _ub[offset], value, 1, 1, 8)


# pylint: disable=no-self-use,too-many-arguments
def tik_func_element_func(tik_instance, function, out_dst, src0, src1, copy_num,
                          dst_blk=1, src0_blk=1, src1_blk=1, dst_rep=8, src0_rep=8,
                          src1_rep=8):
    """
    tik_func_vcomple
    """
    # dtype
    do_dtype = out_dst.dtype
    byte_num_one = common_util.get_data_size(do_dtype)
    block_num = constant_util.BLOCK_SIZE // byte_num_one
    vector_num = block_num * constant_util.REPEAT_STRIDE_EIGHT

    repeat_255_scalar = tik_instance.Scalar(init_value=(copy_num // (vector_num * REPEAT_MAX)))
    repeat_remain_scalar = tik_instance.Scalar(
        init_value=((copy_num - repeat_255_scalar * vector_num * REPEAT_MAX) // vector_num))

    repeat_tail = tik_instance.Scalar(
        init_value=(copy_num - vector_num * REPEAT_MAX * repeat_255_scalar - repeat_remain_scalar * vector_num))
    tik_fun = None
    if function == "vmin":
        tik_fun = tik_instance.vmin
    elif function == "vmax":
        tik_fun = tik_instance.vmax
    elif function == "vmul":
        tik_fun = tik_instance.vmul
    elif function == "vadd":
        tik_fun = tik_instance.vadd
    elif function == "vsub":
        tik_fun = tik_instance.vsub

    with tik_instance.if_scope(repeat_255_scalar > 0):
        with tik_instance.for_range(0, repeat_255_scalar, name='vadd_i0') as i:
            offset = vector_num * REPEAT_MAX * i
            tik_fun(vector_num,
                    out_dst[offset],
                    src0[offset],
                    src1[offset],
                    REPEAT_MAX,
                    dst_blk, src0_blk, src1_blk,
                    dst_rep, src0_rep, src1_rep)

    with tik_instance.if_scope(repeat_remain_scalar > 0):
        offset = repeat_255_scalar * vector_num * REPEAT_MAX
        tik_fun(vector_num,
                out_dst[offset],
                src0[offset],
                src1[offset],
                repeat_remain_scalar,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)

    with tik_instance.if_scope(repeat_tail > 0):
        offset = vector_num * REPEAT_MAX * repeat_255_scalar + repeat_remain_scalar * vector_num
        tik_fun(repeat_tail,
                out_dst[offset],
                src0[offset],
                src1[offset],
                1,
                dst_blk, src0_blk, src1_blk,
                dst_rep, src0_rep, src1_rep)


# pylint: disable=too-many-arguments,no-self-use
def emit_vextract(tik_instance, dst, src, mode, cnt, dst_offset=0, src_offset=0):
    """
    emit_vextract
    """
    repeat_255 = cnt // (16 * 255)
    repeat_remain = (cnt - repeat_255 * 16 * 255) // 16

    with tik_instance.if_scope(repeat_255 > 0):
        with tik_instance.for_range(0, repeat_255, name='i0') as i:
            tik_instance.vextract(dst[dst_offset + i * 255 * 16], src[src_offset + i * 255 * 16 * 8], 255, mode)
    with tik_instance.if_scope(repeat_remain > 0):
        tik_instance.vextract(dst[dst_offset + 255 * 16 * repeat_255], src[src_offset + 255 * 16 * 8 * repeat_255],
                              repeat_remain, mode)


# pylint: disable=too-many-arguments,no-self-use
def emit_vec_conv(tik_instance, dst, src, cnt, dst_offset=0, src_offset=0):
    """
    emit_vec_conv
    """
    repeat_255_scalar = tik_instance.Scalar(init_value=(cnt // (16 * 255)))
    repeat_remain_scalar = tik_instance.Scalar(init_value=((cnt - repeat_255_scalar * 16 * 255) // 16))

    with tik_instance.if_scope(repeat_255_scalar > 0):
        with tik_instance.for_range(0, repeat_255_scalar, name='vconv_i0') as i:
            tik_instance.vec_conv(BLOCK, "round", dst[dst_offset + i * 255 * 16],
                                  src[src_offset + i * 255 * 16], 255, 2, 1)
    with tik_instance.if_scope(repeat_remain_scalar > 0):
        tik_instance.vec_conv(BLOCK, "round", dst[dst_offset + 255 * 16 * repeat_255_scalar],
                              src[src_offset + 255 * 16 * repeat_255_scalar], repeat_remain_scalar, 2, 1)


# pylint: disable=too-many-arguments,too-many-instance-attributes,unused-argument
# pylint: disable=too-many-public-methods,invalid-name,too-many-lines
# pylint: disable=attribute-defined-outside-init,too-many-branches,too-many-statements
class Sort(object):
    """
       Function: use to store sort base parameters
       Modify : 2021-04-13
    """

    def __init__(self, x, y1, y2, axis, descending, kernel_name):
        """
        Init Sort parameters

        Parameters
        ----------
        x: dict
            A Tensor. Must be one of the following types: float16
        y1: dict
            A Tensor. Must be one of the following types: float16
        y2: dict
            A Tensor. Must be one of the following types: int32
        axis: int
            the dimension to sort along
        descending: bool
            controls the sorting order (ascending or descending) default value is "False".
        kernel_name: str
            cce kernel name, default value is "sort".
        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.ai_core_num = tbe_platform.get_soc_spec(
            tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(
            tbe_platform.UB_SIZE) - RESERVED_UB_SIZE)

        self.input_dtype = x.get("dtype").lower()
        # get data each block
        self.data_byte_size = tbe_platform.get_bit_len(self.input_dtype)
        self.data_each_block = BLOCK_BYTES // self.data_byte_size

        self.indices_byte_size = tbe_platform.get_bit_len("int32")
        self.indices_each_block = BLOCK_BYTES // self.indices_byte_size

        # attr
        self.axis = axis
        self.descending = descending

        # init tik gm
        self.input_gm_list = []
        self.output_gm_list = []
        self.input_gm = None
        self.tiling_gm = None
        self.temp_proposal_gm = None
        self.out_gm1 = None
        self.out_gm2 = None
        self.data_out = None
        self.data_indices = None

        self.init_tik_mem()

        # init ub para
        self.tiling_ub = None
        self.max_data = self.tik_instance.Scalar("int32", name="max_data")
        self.max_data.set_as(DATA_MAX)
        self.repeat_max = self.tik_instance.Scalar("int32", name="repeat_max")
        self.repeat_max.set_as(REPEAT_MAX)

        # input ub tensor size
        self.src_ub_size = 44800
        self.des_ub_size = 44800
        self.ub_align_size = 5600

    # pylint: disable=no-self-use,too-many-arguments
    def init_tik_mem(self):
        """
        init tik mem
        :return:
        """
        self.tiling_gm = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)

        self.input_gm = self.tik_instance.Tensor(self.input_dtype, (MAX_INT32_VALUE,), name="input_gm",
                                                 scope=tik.scope_gm)

        self.data_out = self.tik_instance.Tensor(self.input_dtype, [WORKSPACE_SIZE, ], name="data_out",
                                                 scope=tik.scope_gm, is_workspace=True)

        self.data_indices = self.tik_instance.Tensor("int32", [WORKSPACE_SIZE, ], name="data_indices",
                                                     scope=tik.scope_gm, is_workspace=True)

        self.temp_proposal_gm = self.tik_instance.Tensor(self.input_dtype, [TEMP_PROPOSAL_SIZE, ],
                                                         name="temp_proposal_gm",
                                                         scope=tik.scope_gm, is_workspace=True)

        self.out_gm1 = self.tik_instance.Tensor(self.input_dtype, (MAX_INT32_VALUE,), name="out_gm1",
                                                scope=tik.scope_gm)

        self.out_gm2 = self.tik_instance.Tensor("int32", (MAX_INT32_VALUE,), name="out_gm2", scope=tik.scope_gm)

        self.input_gm_list = [self.input_gm]
        self.output_gm_list = [self.out_gm1, self.out_gm2]

    # pylint: disable=no-self-use,too-many-arguments
    def tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from sort tiling

        Returns
        -------
        None
        """
        # tiling mode
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.tiling_mode.set_as(self.tiling_ub[0])

        self.used_core_num = self.tik_instance.Scalar("int32", name="used_core_num")
        self.used_core_num.set_as(self.tiling_ub[1])
        # round
        self.round = self.tik_instance.Scalar("int32", name="round")
        self.round.set_as(self.tiling_ub[2])
        # origin unsorted last dim num
        self.dim_num = self.tik_instance.Scalar("int32", name="dim_num")
        self.dim_num.set_as(self.tiling_ub[3])
        # 32B align last dim num
        self.dim_align_num = self.tik_instance.Scalar("int32", name="dim_align_num")
        self.dim_align_num.set_as(self.tiling_ub[4])
        # merge loop_times
        self.loop_times = self.tik_instance.Scalar("int32", name="loop_times")
        self.loop_times.set_as(self.tiling_ub[5])

        self.batch_num_per_aicore = self.tik_instance.Scalar("int32", name="batch_num_per_aicore")
        self.batch_num_per_aicore.set_as(self.tiling_ub[6])

        self.batch_tail = self.tik_instance.Scalar("int32", name="batch_tail")
        self.batch_tail.set_as(self.tiling_ub[7])

        self.col_tail_loop = self.tik_instance.Scalar("int32", name="col_tail_loop")
        self.col_tail_loop.set_as(self.tiling_ub[8])

        self.num_block_align = self.tik_instance.Scalar("int32", name="num_block_align")
        self.num_block_align.set_as(self.tiling_ub[9])

    # pylint: disable=no-self-use,too-many-arguments
    def generate_proposal_more(self, tik_instance, src_ub, des_ub, input_gm, num_16, num, version, dtype, offset_in):
        """
        generate from unsorted region
        :param tik_instance:
        :param src_ub:
        :param des_ub:
        :param input_gm:
        :param num_16:
        :param num:
        :param version:
        :param dtype:
        :param offset_in:
        :return:
        """
        tik_instance.data_move(des_ub, input_gm[offset_in], 0, 1, num_16 // BLOCK, 0, 0)

        proposal_loop = tik_instance.Scalar(dtype="int32", init_value=0)
        proposal_loop.set_as((self.dim_align_num - 1 + MAX_PROPOSAL_NUM) // MAX_PROPOSAL_NUM)

        data_out_ub_ = tik_instance.Tensor(dtype, [BLOCK], name="data_out_ub_", scope=tik.scope_ubuf)
        data_indices_ub_int_ = tik_instance.Tensor("int32", [BLOCK], name="data_indices_ub_int_",
                                                   scope=tik.scope_ubuf)
        with tik_instance.for_range(0, MAX_PROPOSAL_NUM) as i2:
            data_indices_ub_int_.set_as(i2)
            tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
            des_ub[num_16 * 5 + i2].set_as(data_out_ub_[0])

        with tik_instance.for_range(0, proposal_loop) as loop_id:
            data_indices_ub_int_.set_as(loop_id)
            tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
            proposal_id = tik_instance.Scalar(dtype=dtype)
            proposal_id.set_as(data_out_ub_[0])

            tik_func_vector_dup(tik_instance, des_ub[6 * num_16], proposal_id, MAX_PROPOSAL_NUM)

            # move index 0 to head des_ub
            tik_instance.data_move(des_ub[num_16 + loop_id * MAX_PROPOSAL_NUM], des_ub[5 * num_16], 0, 1,
                                   MAX_PROPOSAL_NUM // BLOCK, 0, 0)
            # move index 1 to second des_ub
            tik_instance.data_move(des_ub[3 * num_16 + loop_id * MAX_PROPOSAL_NUM], des_ub[6 * num_16], 0, 1,
                                   MAX_PROPOSAL_NUM // BLOCK, 0, 0)

        with tik_instance.if_scope(num_16 > DATA_MAX):
            # index0
            tik_instance.vconcat(src_ub[0], des_ub[num_16 * 3], self.repeat_max, 0)
            tik_instance.vconcat(src_ub[self.max_data * PROPOSAL_NUM], des_ub[num_16 * 3 + self.max_data],
                                 (num_16 % self.max_data) // BLOCK, 0)
            # index 1
            tik_instance.vconcat(src_ub[0], des_ub[num_16], self.repeat_max, 1)
            tik_instance.vconcat(src_ub[self.max_data * PROPOSAL_NUM], des_ub[num_16 + self.max_data],
                                 (num_16 % self.max_data) // BLOCK, 1)
            # value
            tik_instance.vconcat(src_ub[0], des_ub, self.repeat_max, 4)
            tik_instance.vconcat(src_ub[self.max_data * PROPOSAL_NUM], des_ub[self.max_data],
                                 (num_16 % self.max_data) // BLOCK, 4)

        with tik_instance.else_scope():
            tik_instance.vconcat(src_ub[0], des_ub[num_16 * 3], num_16 // BLOCK, 0)

            tik_instance.vconcat(src_ub[0], des_ub[num_16], num_16 // BLOCK, 1)

            tik_instance.vconcat(src_ub[0], des_ub, num_16 // BLOCK, 4)

        return src_ub, des_ub

    # pylint: disable=no-self-use,too-many-arguments
    def generate_proposal_less(self, tik_instance, src_ub, des_ub, input_gm, num_16, num, version, dtype, offset_in):
        """
        generate from unsorted region
        :param tik_instance:
        :param src_ub:
        :param des_ub:
        :param input_gm:
        :param num_16:
        :param num:
        :param version:
        :param dtype:
        :param offset_in:
        :return:
        """
        tik_instance.data_move(des_ub, input_gm[offset_in], 0, 1, num_16 // BLOCK, 0, 0)

        if version == "cloud":
            idx = tik_instance.Scalar(dtype="float32", init_value=num)
            with tik_instance.for_range(0, num) as i2:
                idx.set_as(idx - 1)
                src_ub[(num - 1 - i2) * PROPOSAL_NUM].set_as(idx)
        elif version == "mini":
            data_out_ub_ = tik_instance.Tensor(dtype, [BLOCK], name="data_out_ub_", scope=tik.scope_ubuf)
            data_indices_ub_int_ = tik_instance.Tensor("int32", [BLOCK], name="data_indices_ub_int_",
                                                       scope=tik.scope_ubuf)
            with tik_instance.for_range(0, num) as i2:
                data_indices_ub_int_.set_as(num - 1 - i2)
                tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
                src_ub[(num - 1 - i2) * PROPOSAL_NUM].set_as(data_out_ub_[0])
        else:
            raise RuntimeError("Unexcepted version.")

        with tik_instance.if_scope(num_16 > DATA_MAX):
            tik_instance.vconcat(src_ub[0], des_ub, self.repeat_max, 4)
            tik_instance.vconcat(src_ub[self.max_data * PROPOSAL_NUM], des_ub[self.max_data],
                                 (num_16 % self.max_data) // BLOCK, 4)
        with tik_instance.else_scope():
            tik_instance.vconcat(src_ub[0], des_ub, num_16 // BLOCK, 4)

        return src_ub, des_ub

    # pylint: disable=no-self-use,too-many-arguments
    def vbs16(self, tik_instance, num, total, src_ub, des_ub, descending):
        """
        Function: Sort every 16 numsi in UB, des_ub is sorted after vbs16
        Modify : 2021-04-13

        Init base parameters
        Parameters
        ----------
        num: The number of effective object.
        total: The number of all object (16 alignment).
        input_ub: UB
        ----------
        """
        max_scalar = tik_instance.Scalar('float16', init_value=65504)
        min_scalar = tik_instance.Scalar('float16', init_value=-65504)
        # Add ineffective object for 16 alignment
        if descending:
            with tik_instance.for_range(0, total - num) as i:
                src_ub[(num + i) * PROPOSAL_NUM + 4].set_as(min_scalar)
        else:
            with tik_instance.for_range(0, total - num) as i:
                src_ub[(num + i) * PROPOSAL_NUM + 4].set_as(max_scalar)

        n_repeat_total = total // BLOCK
        with self.tik_instance.if_scope(n_repeat_total > REPEAT_MAX):
            tik_instance.vrpsort16(dst=des_ub[0], src=src_ub[0], repeat_times=self.repeat_max)

            tik_instance.vrpsort16(dst=des_ub[self.repeat_max * BLOCK * PROPOSAL_NUM],
                                   src=src_ub[self.repeat_max * BLOCK * PROPOSAL_NUM],
                                   repeat_times=n_repeat_total - REPEAT_MAX)
        with self.tik_instance.else_scope():
            tik_instance.vrpsort16(dst=des_ub[0], src=src_ub[0], repeat_times=n_repeat_total)

        return src_ub, des_ub

    # pylint: disable=no-self-use,too-many-arguments
    def sort_region(self, tik_instance, src_ub, dst_ub, last_dim, total_region_list, loop_time, region_offset=0):
        """
        sort region
        :param tik_instance:
        :param src_ub:
        :param dst_ub:
        :param last_dim:
        :param total_region_list:
        :param region_offset:
        :return: dst_ub is sorted
        """
        region_list_reg = tik_instance.Scalar(init_value=total_region_list, dtype="int32", name="region_list_reg")
        with tik_instance.for_range(0, loop_time) as i:
            with tik_instance.if_scope(i % 2 == 0):
                self._merge(tik_instance, src_ub, dst_ub, last_dim, region_list_reg, (i + 1), region_offset)
            with tik_instance.else_scope():
                self._merge(tik_instance, dst_ub, src_ub, last_dim, region_list_reg, (i + 1), region_offset)
            region_list_reg.set_as((region_list_reg + 3) // 4)

    # pylint: disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements,no-self-use
    @staticmethod
    def _merge(tik_instance, src_ub, dst_ub, last_dim, total_region_list, level, region_offset=0):
        """
        _merge_recur
        merge multi sorted region proposal list to one sorted region proposal list
        dst_ub is sorted
        """
        mid_var = total_region_list // 4
        loops = tik_instance.Scalar(init_value=(mid_var), dtype="int32", name="loops")
        remain = tik_instance.Scalar(init_value=(total_region_list - loops * 4), dtype="int32", name="remain")

        level_reg = tik_instance.Scalar(init_value=level, dtype="int32", name="level_reg")

        merge_n0_reg = tik_instance.Scalar(init_value=1, name="merge_n0_reg")
        merge_n0_reg.set_as(merge_n0_reg << (4 + 2 * level_reg - 2))

        merge_n1_reg = tik_instance.Scalar(init_value=merge_n0_reg, name="merge_n1_reg")
        merge_n2_reg = tik_instance.Scalar(init_value=merge_n0_reg, name="merge_n2_reg")
        merge_n3_reg = tik_instance.Scalar(init_value=merge_n0_reg, name="merge_n3_reg")
        merge_repeat = tik_instance.Scalar(init_value=loops, name="merge_repeat")

        need_tail_process = tik_instance.Scalar(init_value=0, name="need_tail_process")
        merge_left = tik_instance.Scalar(init_value=0, name="merge_left")
        n012 = tik_instance.Scalar(init_value=0, name="n012")

        with tik_instance.if_scope(tik.all(loops > 0, remain == 0)):
            with tik_instance.if_scope(merge_n0_reg * 4 * loops > last_dim):
                merge_repeat.set_as(loops - 1)
                n012.set_as(merge_n0_reg + merge_n1_reg + merge_n2_reg)
                merge_left.set_as(last_dim - ((merge_n0_reg * 4 * (loops - 1)) + n012))
                need_tail_process.set_as(1)

        with tik_instance.if_scope(merge_repeat > 0):
            src_list = [
                src_ub[region_offset], src_ub[region_offset + merge_n0_reg * 8],
                src_ub[region_offset + merge_n0_reg * 8 + merge_n1_reg * 8],
                src_ub[region_offset + merge_n0_reg * 8 + merge_n1_reg * 8 + merge_n2_reg * 8]
            ]
            tik_instance.vmrgsort4(dst_ub[region_offset], src_list,
                                   (merge_n0_reg, merge_n1_reg, merge_n2_reg, merge_n3_reg), False, 15, merge_repeat)

        with tik_instance.if_scope(need_tail_process == 1):
            tail_offset = tik_instance.Scalar(init_value=(4 * merge_n0_reg * merge_repeat * 8), name="tail_offset")
            src_list = [
                src_ub[region_offset + tail_offset], src_ub[region_offset + tail_offset + merge_n0_reg * 8],
                src_ub[region_offset + tail_offset + merge_n0_reg * 8 + merge_n1_reg * 8],
                src_ub[region_offset + tail_offset + merge_n0_reg * 8 + merge_n1_reg * 8 + merge_n2_reg * 8]
            ]
            tik_instance.vmrgsort4(dst_ub[region_offset + tail_offset], src_list,
                                   (merge_n0_reg, merge_n1_reg, merge_n2_reg, merge_left), False, 15, 1)

        offset_reg = tik_instance.Scalar(dtype="int32", init_value=0, name="offset_reg")
        offset_reg.set_as(4 * loops * merge_n0_reg)

        with tik_instance.if_scope(remain == 3):
            merge_n2_reg.set_as(last_dim - (offset_reg + merge_n0_reg + merge_n1_reg))

            src_list = [
                src_ub[region_offset + offset_reg * 8], src_ub[region_offset + offset_reg * 8 + merge_n0_reg * 8],
                src_ub[region_offset + offset_reg * 8 + merge_n0_reg * 8 + merge_n1_reg * 8], src_ub[0]
            ]

            tik_instance.vmrgsort4(dst_ub[region_offset + offset_reg * 8], src_list,
                                   (merge_n0_reg, merge_n1_reg, merge_n2_reg, 16), False, 7, 1)
        with tik_instance.if_scope(remain == 2):
            merge_n1_reg.set_as(last_dim - (offset_reg + merge_n0_reg))
            src_list = [
                src_ub[region_offset + offset_reg * 8], src_ub[region_offset + offset_reg * 8 + merge_n0_reg * 8],
                src_ub[0], src_ub[0]
            ]

            tik_instance.vmrgsort4(dst_ub[region_offset + offset_reg * 8], src_list,
                                   (merge_n0_reg, merge_n1_reg, 16, 16), False, 3, 1)

        with tik_instance.if_scope(remain == 1):
            merge_n0_reg.set_as(last_dim - offset_reg)
            num_blocks_write_reg = tik_instance.Scalar(init_value=((merge_n0_reg * 16 + 31) // 32), name="merge_n1_reg")
            tik_instance.data_move(dst_ub[region_offset + offset_reg * 8], src_ub[region_offset + offset_reg * 8], 0, 1,
                                   num_blocks_write_reg, 0, 0)

    # pylint: disable=no-self-use,too-many-arguments
    def sort_compute_less(self, tik_instance, dtype, num_16, i0, descending, num, data_out, data_indices, input_gm):
        """
        Function: sort the proposal id num is more than 4000
        Modify : 2021-04-15

        Attention : This way is unstable (can't compare two scalar).
        Init base parameters
        Parameters
        ----------
        dtype, num_16, i0, descending, num, distance, shape, big_distance, L : for index compute
        data_out, data_indices, input_gm : for data move
        ----------
        """

        src_ub = self.tik_instance.Tensor(self.input_dtype, [self.src_ub_size], name="src_ub",
                                          scope=tik.scope_ubuf)
        des_ub = self.tik_instance.Tensor(self.input_dtype, [self.des_ub_size], name="des_ub",
                                          scope=tik.scope_ubuf)

        version = tik.Dprofile().get_product_name()
        offset_in = i0 * num
        offset_out = i0 * num_16

        # 1. generate proposal,src_ub is proposal
        with tik_instance.if_scope(self.dim_align_num > MAX_PROPOSAL_NUM):
            src_ub, des_ub = self.generate_proposal_more(tik_instance, src_ub, des_ub,
                                                         input_gm, num_16, num, version,
                                                         dtype, offset_in)
        with tik_instance.else_scope():
            src_ub, des_ub = self.generate_proposal_less(tik_instance, src_ub, des_ub,
                                                         input_gm, num_16, num, version,
                                                         dtype, offset_in)

        # 2. sort proposal 16 by 16,des_ub is sorted ub
        src_ub, des_ub = self.vbs16(tik_instance, num, num_16, src_ub, des_ub, descending)

        # 3.merge proposal
        with tik_instance.if_scope(self.dim_align_num > 16):
            # if dim is bigger than 16, des_ub to be sort,sort result is put in src_ub
            self.sort_region(tik_instance, des_ub, src_ub, self.dim_align_num,
                             self.dim_align_num // 16, self.loop_times, region_offset=0)
            # src_ub is sorted ub
            with tik_instance.if_scope(self.dim_align_num > MAX_PROPOSAL_NUM):
                data_out, data_indices = self.moveout_more(tik_instance, descending, num_16, num, data_out, offset_out,
                                                           src_ub,
                                                           des_ub, data_indices, version, self.ub_align_size)
            with tik_instance.else_scope():
                data_out, data_indices = self.moveout_less(tik_instance, descending, num_16, num, data_out, offset_out,
                                                           src_ub,
                                                           des_ub, data_indices, version, self.ub_align_size)
        with tik_instance.else_scope():
            # des_ub is sorted ub
            data_out, data_indices = self.moveout_less(tik_instance, descending, num_16, num, data_out, offset_out,
                                                       des_ub,
                                                       src_ub, data_indices, version, self.ub_align_size)

        return data_out, data_indices

    # pylint: disable=no-self-use,too-many-arguments
    def sort_in_ub(self, tik_instance, src_ub, dst_ub, idx_ub, num, i, ori_offset, offset, descending):
        """
        Function: sort proposal 2048 by 2048 in ub
        Modify : 2021-05-22
        Attention : This way is unstable (can't compare two scalar).

        """

        repeat_times = NUM_BLOCK // BLOCK

        # 1. Move data from OUT to UB
        tik_instance.data_move(dst_ub, self.input_gm[ori_offset + i * NUM_BLOCK], 0, 1, repeat_times, 0, 0)

        # generate index ub
        index_0_ub = tik_instance.Tensor("float16", [NUM_BLOCK], name="index_0_ub", scope=tik.scope_ubuf)
        temp_float16_ub = tik_instance.Tensor("float16", [BLOCK], name="temp_float16_ub", scope=tik.scope_ubuf)
        temp_ub_int = tik_instance.Tensor("int32", [BLOCK], name="temp_ub_int",
                                          scope=tik.scope_ubuf)
        temp_ub_int[0].set_as(i)
        tik_instance.vec_conv(1, "none", temp_float16_ub, temp_ub_int, 1, 0, 0, deqscale=1.0)
        temp_scalar = self.tik_instance.Scalar("float16", name="temp_scalar")
        temp_scalar.set_as(temp_float16_ub[0])
        tik_func_vector_dup(tik_instance, index_0_ub, temp_scalar, NUM_BLOCK)

        # append big value to ub memory
        with tik_instance.if_scope(num < (i + 1) * NUM_BLOCK):
            # aline for NUM_BLOCK
            aline = NUM_BLOCK - num % NUM_BLOCK
            if descending:
                big_value_scalar = tik_instance.Scalar('float16', init_value=MIN_VAL)
            # descend
            else:
                big_value_scalar = tik_instance.Scalar('float16', init_value=MAX_VAL)
            # Add ineffective object for 16 alignment
            with tik_instance.for_range(0, aline % BLOCK) as j:
                dst_ub[num % NUM_BLOCK + j].set_as(big_value_scalar)
            # Add ineffective object for NUM_BLOCK alignment
            with tik_instance.if_scope(aline > BLOCK - 1):
                tik_instance.vec_dup(BLOCK, dst_ub[num % NUM_BLOCK + aline % BLOCK], big_value_scalar,
                                     aline // BLOCK, 1)

        # index // 2048 for index 0
        tik_instance.vconcat(src_ub[0], index_0_ub[0], repeat_times, 0)
        # index % 2048 for index 1
        tik_instance.vconcat(src_ub[0], idx_ub[0], repeat_times, 1)
        tik_instance.vconcat(src_ub[0], dst_ub[0], repeat_times, 4)

        # 2. sort proposal 16 by 16,des_ub is sorted ub
        tik_instance.vrpsort16(dst=dst_ub[0], src=src_ub[0], repeat_times=repeat_times)

        # 3. merge sort ,src_ub is sorted ub
        loop_time = 5
        self.sort_region(tik_instance, dst_ub, src_ub, NUM_BLOCK,
                         NUM_BLOCK // 16, loop_time, region_offset=0)

        # 4. Move Data from UB to OUT
        tik_instance.data_move(self.temp_proposal_gm[offset + i * NUM_BLOCK * PROPOSAL_NUM], src_ub[0], 0, 1,
                               NUM_BLOCK * PROPOSAL_NUM // BLOCK, 0, 0)

    def sort_in_gm(self, tik_instance, num_gm, input_ub, offset):
        """
        Function: sort in gm.
        Modify : 2020-11-16

        Init base parameters
        Parameters
        ----------
        num_gm, offset : for index compute
        temp, input_ub : for data move
        ----------
        """
        src_pos_ub = tik_instance.Scalar("int32")
        dest_pos_ub = tik_instance.Scalar("int32")

        with tik_instance.for_range(0, num_gm - 1) as tail:
            src_pos_ub.set_as(0)
            dest_pos_ub.set_as(NUM_BLOCK * 2 * PROPOSAL_NUM)

            tik_instance.data_move(input_ub[src_pos_ub + NUM_BLOCK * PROPOSAL_NUM], self.temp_proposal_gm[offset], 0, 1,
                                   (NUM_BLOCK * PROPOSAL_NUM) // BLOCK, 0, 0)
            with tik_instance.for_range(1, num_gm - tail) as i:
                tik_instance.data_move(input_ub[src_pos_ub],
                                       self.temp_proposal_gm[offset + NUM_BLOCK * i * PROPOSAL_NUM], 0, 1,
                                       (NUM_BLOCK * PROPOSAL_NUM) // BLOCK, 0, 0)

                tik_instance.vmrgsort4(input_ub[dest_pos_ub],
                                       [input_ub[src_pos_ub], input_ub[src_pos_ub + NUM_BLOCK * PROPOSAL_NUM],
                                        input_ub[0], input_ub[0]], [NUM_BLOCK, NUM_BLOCK, 0, 0],
                                       if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)

                tik_instance.data_move(self.temp_proposal_gm[offset + NUM_BLOCK * (i - 1) * PROPOSAL_NUM],
                                       input_ub[dest_pos_ub], 0, 1,
                                       (NUM_BLOCK * PROPOSAL_NUM) // BLOCK, 0, 0)

                dest_pos_ub.set_as(src_pos_ub)
                src_pos_ub.set_as(NUM_BLOCK * 2 * PROPOSAL_NUM - dest_pos_ub)

            # Move Data from UB to GM
            tik_instance.data_move(self.temp_proposal_gm[offset + NUM_BLOCK * (num_gm - tail - 1) * PROPOSAL_NUM],
                                   input_ub[src_pos_ub + NUM_BLOCK * PROPOSAL_NUM], 0, 1,
                                   (NUM_BLOCK * PROPOSAL_NUM) // BLOCK, 0, 0)

    def pick_up(self, tik_instance, offset, core_idx, num_2048, data_out, data_indices, src_ub, dst_ub, num_gm,
                descending, version):
        """
        Function: pick value from proposal.
        Modify : 2021-05-13
        ----------
        temp, offset, core_idx, num_2048 : for index compute
        data_out, input_ub2, num_gm : for data move
        ----------
        """
        # dest position in UB
        repeat_times = NUM_BLOCK // BLOCK

        with tik_instance.for_range(0, num_gm) as i:
            tik_instance.data_move(src_ub[0], self.temp_proposal_gm[offset + NUM_BLOCK * i * PROPOSAL_NUM], 0, 1,
                                   (NUM_BLOCK * PROPOSAL_NUM) // BLOCK, 0, 0)
            int_list_1 = tik_instance.Tensor("int32", [NUM_BLOCK], name="int_list_1", scope=tik.scope_ubuf)
            int_list_2 = tik_instance.Tensor("int32", [NUM_BLOCK], name="int_list_2", scope=tik.scope_ubuf)
            int_list_3 = tik_instance.Tensor("int32", [NUM_BLOCK], name="int_list_3", scope=tik.scope_ubuf)
            int_list_4 = tik_instance.Tensor("int32", [NUM_BLOCK], name="int_list_4", scope=tik.scope_ubuf)

            tik_instance.vector_dup(BLOCK, int_list_4, NUM_BLOCK, repeat_times, 1, 2)

            tik_instance.vextract(dst_ub[0], src_ub[0], repeat_times, 0)
            tik_instance.vextract(dst_ub[NUM_BLOCK], src_ub[0], repeat_times, 1)
            tik_instance.vec_conv(BLOCK, "round", int_list_1, dst_ub[0], repeat_times, 2, 1)
            tik_instance.vec_conv(BLOCK, "round", int_list_2, dst_ub[NUM_BLOCK], repeat_times, 2, 1)

            tik_instance.vec_mul(BLOCK, int_list_3, int_list_1, int_list_4, repeat_times, 2, 2, 2)
            tik_instance.vec_add(BLOCK, int_list_1, int_list_2, int_list_3, repeat_times, 2, 2, 2)

            # data is continuous in GM & gather scattered data together
            if version == "cloud":
                tik_instance.vextract(dst_ub[0], src_ub[0], repeat_times, 4)
            elif version == "mini":
                with tik_instance.for_range(0, NUM_BLOCK) as i2:
                    dst_ub[i2].set_as(src_ub[i2 * PROPOSAL_NUM + 4])
            else:
                raise RuntimeError("Unexcepted version.")

            # move output (float16) from UB to GM
            # ascend
            with tik_instance.if_scope(descending is False):
                with tik_instance.for_range(0, NUM_BLOCK) as i2:
                    int_list_2[i2].set_as(int_list_1[NUM_BLOCK - i2 - 1])
                    dst_ub[NUM_BLOCK + i2].set_as(dst_ub[NUM_BLOCK - i2 - 1])
                tik_instance.data_move(data_indices[core_idx * num_2048 + NUM_BLOCK * (num_gm - i - 1)],
                                       int_list_2, 0, 1, 2 * repeat_times, 0, 0)
                tik_instance.data_move(data_out[core_idx * num_2048 + NUM_BLOCK * (num_gm - i - 1)],
                                       dst_ub[NUM_BLOCK], 0, 1, repeat_times, 0, 0)
            with tik_instance.else_scope():
                tik_instance.data_move(data_indices[core_idx * num_2048 + NUM_BLOCK * i], int_list_1, 0, 1,
                                       2 * repeat_times, 0, 0)

                tik_instance.data_move(data_out[core_idx * num_2048 + NUM_BLOCK * i], dst_ub[0], 0, 1,
                                       repeat_times, 0, 0)
        return data_out, data_indices

    # pylint: disable=no-self-use,too-many-arguments
    def sort_compute_batch(self, tik_instance, dtype, num_16, core_idx, descending, num, data_out, data_indices,
                           input_gm):
        """
        Function: sort the proposal every 2048
        Modify : 2021-05-13

        Attention : This way is unstable (can't compare two scalar).
        Init base parameters
        Parameters
        ----------
        dtype, num_16, i0, descending, num, distance, shape, big_distance, L : for index compute
        data_out, data_indices, input_gm : for data move
        ----------
        """
        idx_ub = tik_instance.Tensor(dtype, [NUM_BLOCK], name="idx_ub", scope=tik.scope_ubuf)

        version = tik.Dprofile().get_product_name()
        if version == "cloud":
            idx = tik_instance.Scalar(dtype="float32", init_value=0)
            with tik_instance.for_range(0, NUM_BLOCK) as i2:
                idx_ub[i2].set_as(idx)
                idx.set_as(idx + 1)
        elif version == "mini":
            data_out_ub_ = tik_instance.Tensor(dtype, [BLOCK], name="data_out_ub_", scope=tik.scope_ubuf)
            data_indices_ub_int_ = tik_instance.Tensor("int32", [BLOCK], name="data_indices_ub_int_",
                                                       scope=tik.scope_ubuf)
            with tik_instance.for_range(0, NUM_BLOCK) as i2:
                data_indices_ub_int_.set_as(i2)
                tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
                idx_ub[i2].set_as(data_out_ub_[0])
        else:
            raise RuntimeError("Unexcepted version.")

        num_gm = self.num_block_align // NUM_BLOCK

        offset = (core_idx % self.used_core_num) * num_gm * NUM_BLOCK * PROPOSAL_NUM

        ori_offset = core_idx * num

        with tik_instance.new_stmt_scope():
            src_ub = tik_instance.Tensor(dtype, [NUM_BLOCK * 2 * PROPOSAL_NUM], name="src_ub", scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor(dtype, [NUM_BLOCK * 2 * PROPOSAL_NUM], name="dst_ub", scope=tik.scope_ubuf)

            # 1.sort proposal every 2048
            with tik_instance.for_range(0, num_gm) as i:
                self.sort_in_ub(tik_instance, src_ub, dst_ub, idx_ub, num, i, ori_offset, offset, descending)

        # 2.use gm to sort
        with tik_instance.new_stmt_scope():
            input_ub = tik_instance.Tensor(dtype, [NUM_BLOCK * 4 * PROPOSAL_NUM], name="input_ub", scope=tik.scope_ubuf)
            self.sort_in_gm(tik_instance, num_gm, input_ub, offset)

        with tik_instance.new_stmt_scope():
            src_ub = tik_instance.Tensor(dtype, [NUM_BLOCK * 2 * PROPOSAL_NUM], name="src_ub", scope=tik.scope_ubuf)
            dst_ub = tik_instance.Tensor(dtype, [NUM_BLOCK * 2 * PROPOSAL_NUM], name="dst_ub", scope=tik.scope_ubuf)

            # 3.move data to workspace gm
            data_out, data_indices = self.pick_up(tik_instance, offset, core_idx, self.num_block_align, data_out,
                                                  data_indices, src_ub, dst_ub, num_gm, descending, version)

        return data_out, data_indices

    # pylint: disable=no-self-use,too-many-arguments
    @staticmethod
    def moveout_more(tik_instance, descending, num_16, num, data_out, offset_out, src_ub, des_ub, data_indices,
                     version, ub_align_size):
        """
        Function: Move UB to GM, and trans y2 from fp16 to int32.
        Modify : 2021-04-13

        Attention : This way is unstable (can't compare two scalar).
        Init base parameters
        Parameters
        ----------
        descending, offset_out, num_16, num, dest_pos_ub : for index compute
        data_out, input_ub, data_indices : for data move
        ----------
        """
        # src_ub is sorted proposal

        int_list_0 = tik_instance.Tensor("int32", [ub_align_size], name="int_list_0", scope=tik.scope_ubuf)
        int_list_1 = tik_instance.Tensor("int32", [ub_align_size], name="int_list_1", scope=tik.scope_ubuf)
        int_list_const = tik_instance.Tensor("int32", [ub_align_size], name="int_list_const", scope=tik.scope_ubuf)

        if version == "mini":
            # extract index 0
            emit_vextract(tik_instance, des_ub, src_ub, 0, num_16, dst_offset=0, src_offset=0)
            # extract index 1
            emit_vextract(tik_instance, des_ub, src_ub, 1, num_16, dst_offset=num_16, src_offset=0)
            with tik_instance.for_range(0, num) as i2:
                des_ub[i2 + num_16 * 2].set_as(src_ub[i2 * PROPOSAL_NUM + 4])
        elif version == "cloud":
            # extract index 0
            emit_vextract(tik_instance, des_ub, src_ub, 0, num_16, dst_offset=0, src_offset=0)
            # extract index 1
            emit_vextract(tik_instance, des_ub, src_ub, 1, num_16, dst_offset=num_16, src_offset=0)
            # extract sorted value
            emit_vextract(tik_instance, des_ub, src_ub, 4, num_16, dst_offset=2 * num_16, src_offset=0)
        else:
            raise RuntimeError("Unexcepted version.")

        emit_vec_conv(tik_instance, int_list_0, des_ub, num_16, dst_offset=0, src_offset=0)
        tik_func_vector_dup(tik_instance, int_list_const, 2048, num_16)
        # index0*2048
        tik_func_element_func(tik_instance, "vmul", int_list_1, int_list_0, int_list_const, num_16)
        emit_vec_conv(tik_instance, int_list_const, des_ub, num_16, dst_offset=0, src_offset=num_16)
        # index0*2048 + index1
        tik_func_element_func(tik_instance, "vadd", int_list_0, int_list_const, int_list_1, num_16)

        # ascend reversed the ub data & move output
        with tik_instance.if_scope(descending is False):
            with tik_instance.for_range(0, num) as i2:
                des_ub[3 * num_16 + i2].set_as(des_ub[3 * num_16 - 1 - i2])
                int_list_1[i2].set_as(int_list_0[num_16 - 1 - i2])
            # move sorted value
            tik_instance.data_move(data_out[offset_out], des_ub[num_16 * 3], 0, 1, num_16 // BLOCK, 0, 0)
            # move index to out
            tik_instance.data_move(data_indices[offset_out], int_list_1, 0, 1, num_16 // 8, 0, 0)
        # descend
        with tik_instance.else_scope():
            # move sorted value
            tik_instance.data_move(data_out[offset_out], des_ub[num_16 * 2], 0, 1, num_16 // BLOCK, 0, 0)
            # move index to out
            tik_instance.data_move(data_indices[offset_out], int_list_0, 0, 1, num_16 // 8, 0, 0)

        return data_out, data_indices

    # pylint: disable=no-self-use,too-many-arguments
    def moveout_less(self, tik_instance, descending, num_16, num, data_out, offset_out, src_ub, des_ub, data_indices,
                     version, ub_align_size):
        """
        Function: Move UB to GM, and trans y2 from fp16 to int32.
        Modify : 2021-04-13

        Attention : This way is unstable (can't compare two scalar).
        Init base parameters
        Parameters
        ----------
        descending, offset_out, num_16, num, dest_pos_ub : for index compute
        data_out, input_ub, data_indices : for data move
        ----------
        """
        int_list = tik_instance.Tensor("int32", [ub_align_size], name="data_indices_ub_list", scope=tik.scope_ubuf)

        # ascend
        # data is continuous in GM & gather scattered data together
        with tik_instance.if_scope(descending is False):
            with tik_instance.for_range(0, num) as i2:
                des_ub[i2].set_as(src_ub[(num_16 - 1 - i2) * PROPOSAL_NUM + 4])
                des_ub[i2 + num_16].set_as(src_ub[(num_16 - 1 - i2) * PROPOSAL_NUM])

            # conv indices (float16->int32) , and move from UB to GM
            with tik_instance.if_scope(num_16 > DATA_MAX):
                tik_instance.vec_conv(BLOCK, "round", int_list, des_ub[num_16], self.repeat_max, 2, 1)
                tik_instance.vec_conv(BLOCK, "round", int_list[self.max_data], des_ub[num_16 + DATA_MAX],
                                      (num_16 % DATA_MAX) // BLOCK, 2, 1)
            with tik_instance.else_scope():
                tik_instance.vec_conv(BLOCK, "round", int_list, des_ub[num_16], num_16 // BLOCK, 2, 1)

            # move output (float16) from UB to GM
            tik_instance.data_move(data_out[offset_out], des_ub[0], 0, 1, num_16 // BLOCK, 0, 0)
            tik_instance.data_move(data_indices[offset_out], int_list, 0, 1, num_16 // 8, 0, 0)

        # descend
        with tik_instance.else_scope():
            # data is continuous in GM & gather scattered data together
            if version == "mini":
                with tik_instance.for_range(0, num) as i2:
                    des_ub[i2].set_as(src_ub[i2 * PROPOSAL_NUM + 4])
                    des_ub[i2 + num_16].set_as(src_ub[i2 * PROPOSAL_NUM])
            elif version == "cloud":
                with tik_instance.if_scope(num_16 > DATA_MAX):
                    tik_instance.vextract(des_ub[0], src_ub[0], self.repeat_max, 4)
                    tik_instance.vextract(des_ub[self.max_data], src_ub[self.max_data * PROPOSAL_NUM],
                                          (num_16 % self.max_data) // BLOCK, 4)

                    tik_instance.vextract(des_ub[num_16], src_ub[0], self.repeat_max, 0)
                    tik_instance.vextract(des_ub[num_16 + DATA_MAX],
                                          src_ub[self.max_data * PROPOSAL_NUM],
                                          (num_16 % self.max_data) // BLOCK, 0)
                with tik_instance.else_scope():
                    tik_instance.vextract(des_ub[0], src_ub[0], num_16 // BLOCK, 4)
                    tik_instance.vextract(des_ub[num_16], src_ub[0], num_16 // BLOCK, 0)
            else:
                raise RuntimeError("Unexcepted version.")

            # if num_16 > DATA_MAX:
            with tik_instance.if_scope(num_16 > DATA_MAX):
                tik_instance.vec_conv(BLOCK, "round", int_list, des_ub[num_16], self.repeat_max, 2, 1)
                tik_instance.vec_conv(BLOCK, "round", int_list[self.max_data], des_ub[num_16 + DATA_MAX],
                                      (num_16 % self.max_data) // BLOCK, 2, 1)
            with tik_instance.else_scope():
                tik_instance.vec_conv(BLOCK, "round", int_list, des_ub[num_16], num_16 // BLOCK, 2, 1)
            # move output (float16) from UB to GM
            tik_instance.data_move(data_out[offset_out], des_ub[0], 0, 1, num_16 // BLOCK, 0, 0)
            tik_instance.data_move(data_indices[offset_out], int_list, 0, 1, num_16 // 8, 0, 0)

        return data_out, data_indices

    def workspace_to_gm(self):
        """
        Function: move sorted result from workspace to finally gm
        Modify : 2021-05-13
        """
        num_gm = self.num_block_align // NUM_BLOCK
        with self.tik_instance.if_scope(self.dim_num <= NUM_BLOCK):
            with self.tik_instance.for_range(0, self.round) as i:
                float_ub = self.tik_instance.Tensor("float16", [NUM_BLOCK], name="float_ub", scope=tik.scope_ubuf)
                int_ub = self.tik_instance.Tensor("int32", [NUM_BLOCK], name="int_ub", scope=tik.scope_ubuf)

                self.tik_instance.data_move(float_ub[0], self.data_out[i * self.dim_align_num], 0, 1,
                                            self.dim_align_num // 16, 0, 0)
                self.tik_instance.data_move(self.out_gm1[i * self.dim_num], float_ub[0], 0, 1,
                                            self.dim_align_num // 16, 0, 0)

                self.tik_instance.data_move(int_ub[0], self.data_indices[i * self.dim_align_num], 0, 1,
                                            self.dim_align_num // 8, 0, 0)
                self.tik_instance.data_move(self.out_gm2[i * self.dim_num], int_ub[0], 0, 1,
                                            self.dim_align_num // 8, 0, 0)

        with self.tik_instance.else_scope():
            repeat_times = NUM_BLOCK // BLOCK
            with self.tik_instance.for_range(0, self.round) as i:
                with self.tik_instance.for_range(0, num_gm) as j:
                    float_ub = self.tik_instance.Tensor("float16", [NUM_BLOCK], name="float_ub", scope=tik.scope_ubuf)
                    int_ub = self.tik_instance.Tensor("int32", [NUM_BLOCK], name="int_ub", scope=tik.scope_ubuf)
                    self.tik_instance.data_move(float_ub[0], self.data_out[i * self.num_block_align + j * NUM_BLOCK], 0,
                                                1,
                                                repeat_times, 0, 0)
                    self.tik_instance.data_move(self.out_gm1[i * self.dim_num + j * NUM_BLOCK], float_ub[0], 0, 1,
                                                repeat_times, 0, 0)

                    self.tik_instance.data_move(int_ub[0], self.data_indices[i * self.num_block_align + j * NUM_BLOCK],
                                                0, 1,
                                                2 * repeat_times, 0, 0)
                    self.tik_instance.data_move(self.out_gm2[i * self.dim_num + j * NUM_BLOCK], int_ub[0], 0, 1,
                                                2 * repeat_times, 0, 0)

    def sort_compute_tiling(self):
        """
        sort compute tiling
        :return:
        """
        self.tiling_ub = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="tiling_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, TILING_ARG_NUM // 8, 0, 0)
        self.tiling_args()

        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as core_index:
            with self.tik_instance.if_scope(self.tiling_mode == 1):
                with self.tik_instance.for_range(0, self.batch_num_per_aicore) as k:
                    self.data_out, self.data_indices = self.sort_compute_less(self.tik_instance,
                                                                              self.input_dtype,
                                                                              self.dim_align_num,
                                                                              core_index + k * self.ai_core_num,
                                                                              self.descending,
                                                                              self.dim_num,
                                                                              self.data_out,
                                                                              self.data_indices,
                                                                              self.input_gm)
                with self.tik_instance.if_scope(core_index < self.batch_tail):
                    self.data_out, self.data_indices = self.sort_compute_less(self.tik_instance,
                                                                              self.input_dtype,
                                                                              self.dim_align_num,
                                                                              self.batch_num_per_aicore * \
                                                                              self.ai_core_num + core_index,
                                                                              self.descending,
                                                                              self.dim_num,
                                                                              self.data_out,
                                                                              self.data_indices,
                                                                              self.input_gm)
            with self.tik_instance.if_scope(self.tiling_mode == 2):
                with self.tik_instance.for_range(0, self.batch_num_per_aicore) as k:
                    self.data_out, self.data_indices = self.sort_compute_batch(self.tik_instance,
                                                                               self.input_dtype,
                                                                               self.dim_align_num,
                                                                               core_index + k * self.ai_core_num,
                                                                               self.descending,
                                                                               self.dim_num,
                                                                               self.data_out,
                                                                               self.data_indices,
                                                                               self.input_gm)
                with self.tik_instance.if_scope(core_index < self.batch_tail):
                    self.data_out, self.data_indices = self.sort_compute_batch(self.tik_instance,
                                                                               self.input_dtype,
                                                                               self.dim_align_num,
                                                                               self.batch_num_per_aicore * \
                                                                               self.ai_core_num + core_index,
                                                                               self.descending,
                                                                               self.dim_num,
                                                                               self.data_out,
                                                                               self.data_indices,
                                                                               self.input_gm)

        # move result data from gm workspace to gm
        self.workspace_to_gm()

    def sort_compute_operator(self):
        """
        sort_compute_operator
        :return:
        """
        self.sort_compute_tiling()

    def build_tik_instance(self, kernel_name_value):
        """
        build_tik_instance
        """
        opt_config = {"out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(
            kernel_name=kernel_name_value,
            inputs=self.input_gm_list,
            outputs=self.output_gm_list,
            flowtable=[self.tiling_gm],
            config=opt_config)

        tbe_context.get_context().add_compile_info("vars", {"ub_size": self.ub_size_bytes,
                                                            "core_num": self.ai_core_num})
        return self.tik_instance


# pylint: disable=unused-argument
@register_operator("Sort")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def sort(x, y1, y2,
         axis=-1,
         descending=False,
         kernel_name="sort"):
    """
    sort interface

    Parameters
    ----------
        x: dict
            A Tensor. Must be one of the following types: float16
        y1: dict
            A Tensor. Must be one of the following types: float16
        y2: dict
            A Tensor. Must be one of the following types: int32
        axis: int
            the dimension to sort along
        descending: bool
            controls the sorting order (ascending or descending) default value is "False".
        kernel_name: str
            cce kernel name, default value is "sort".

    Returns
    -------
    compile info
    """
    obj = Sort(x, y1, y2, axis, descending, kernel_name)
    obj.sort_compute_operator()
    return obj.build_tik_instance(kernel_name)
