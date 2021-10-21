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
masked_fill_range
"""
import te.platform as tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check

# the number of bits per byte
THREAD_NUM = 2
# the number of data contained in each coordinate box
DEFAULT_NBURST = 1
# data type of fp16
FP16 = "float16"
# data type of fp32
FP32 = "float32"
# data type of fp16
INT32 = "int32"
# data type of fp32
INT8 = "int8"
# number of element of fp16 and fp32 data type in one mask
VEC_MASK = {FP32: 64, FP16: 128, INT32: 64, INT8: 256}
# number of element of fp16 and fp32 data type in one mask
DATA_TYPE_SIZE = {FP32: 4, FP16: 2, INT32: 4, INT8: 1}
# BLOCK_NUM
BLOCK_NUMBER_SIZE = {FP32: 8, FP16: 16, INT32: 8, INT8: 32}
# tiling mode
TILING_MODE = {"TILING_D": 1, "TILING_N": 2, "TILING_NC": 3}
# max repeat times
MAX_REPEAT_TIME = 65536
# max mask repeat times
MAX_MASK_REPEAT_TIME = 255


def _align_with_value(value, align_num):
    return ((value + align_num - 1) // align_num) * align_num


def _ceil_with_value(value, align_num):
    return (value + align_num - 1) // align_num


class MaskedFillRange():
    """
    use to store MaskedFillRange basic parameters
    """

    # 'pylint: disable=too-many-arguments
    def __init__(self, x, start, end, value, y, axis, kernel_name):
        self.input_x_shape = x.get("shape")
        self.input_x_ori_shape = x.get("ori_shape")
        self.input_x_dtype = x.get("dtype").lower()
        self.input_x_format = x.get("format")
        self.input_dim = len(self.input_x_shape)
        self.start_shape = start.get("shape")
        self.start_ori_shape = start.get("ori_shape")
        self.start_dtype = start.get("dtype")
        self.end_shape = end.get("shape")
        self.end_dtype = end.get("dtype")
        self.end_ori_shape = end.get("ori_shape")
        self.value_shape = value.get("shape")
        self.value_dtype = value.get("dtype")
        self.value_ori_shape = value.get("ori_shape")
        self.output_y_shape = y.get("shape")
        self.output_y_dtype = y.get("dtype").lower()
        self.output_y_ori_shape = y.get("ori_shape")
        self.axis = axis
        self.absolute_axis = self._get_absolute_axis(axis)
        self.loop_cycle = self.start_shape[0]
        self.kernel_name = kernel_name

        self.tik_inst = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.soc_version = tbe_platform.get_soc_spec(tbe_platform.SOC_VERSION)
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self._input_param_check()
        self._init_gm_tensor()

        self.each_core_max_ub_size = _align_with_value(
            self.ub_size * DATA_TYPE_SIZE[FP16] // DATA_TYPE_SIZE[self.input_x_dtype] // 8,
            VEC_MASK[self.input_x_dtype])
        self.tiling_param_list = self._get_tiling_param()

    def _input_param_check(self):
        para_check.check_dtype(self.input_x_dtype, ["float32", "float16", "int32", "int8"])
        para_check.check_dtype(self.output_y_dtype, ["float32", "float16", "int32", "int8"])
        para_check.check_format(self.input_x_format, ["ND"])

        if self.input_x_dtype != self.output_y_dtype or self.input_x_dtype != self.value_dtype:
            error_info = {'errCode': 'E60000',
                          'param_value1': self.input_x_dtype,
                          'param_value2': self.value_dtype,
                          'param_value3': self.output_y_dtype,
                          'op_name': 'MaskedFillRange'}
            raise RuntimeError(error_info, "In op[%s], the data type of input [%s], value [%s], output [%s] "
                                           "must be equal."
                               % (error_info['op_name'], error_info['param_value1'], error_info['param_value2'],
                                  error_info['param_value3']))

        if self.absolute_axis >= self.input_dim:
            error_info = {'errCode': 'E60001',
                          'param_value1': self.axis,
                          'param_value2': self.input_dim,
                          'op_name': 'MaskedFillRange'}
            raise RuntimeError(error_info, "In op[%s], the axis [%d] is out of range input dim[%d] "
                               % (error_info['op_name'], error_info['param_value1'], error_info['param_value2']))

        if self.soc_version not in (tbe_platform.ASCEND_910,):
            error_info = {'errCode': 'E60002',
                          'param_value1': self.soc_version,
                          'op_name': 'MaskedFillRange'}
            raise RuntimeError(error_info, "In op[%s], soc version [%s] is out of range [ASCEND310/710/910] "
                               % (error_info['op_name'], error_info['param_value1']))

        if self._get_input_x_len() > 1024 * 1024 * 1024:
            error_info = {'errCode': 'E60003',
                          'param_value1': self._get_input_x_len(),
                          'op_name': 'MaskedFillRange'}
            raise RuntimeError(error_info, "In op[%s], input x length [%d] is out of range [1024 * 1024 * 1024] "
                               % (error_info['op_name'], error_info['param_value1']))

    def _get_absolute_axis(self, axis):
        """
        :param axis: input axis, [-2, -1, 0, 1, 2]
        :return: absolute axis
        """
        real_axis = axis
        if axis < 0:
            real_axis = self.input_dim + axis
        return real_axis

    def _get_input_x_len(self):
        x_len = self.input_x_shape[0]
        for i in range(1, self.input_dim):
            x_len = x_len * self.input_x_shape[i]
        return x_len

    def _is_single_dim_tiling(self):
        if self.input_dim == 1:
            return True
        elif self.input_dim == 2:
            if self.input_x_shape[0] == 1 or self.input_x_shape[1] == 1:
                return True
        else:
            if ((self.input_x_shape[0] * self.input_x_shape[1] == 1) or
                    (self.input_x_shape[0] * self.input_x_shape[2] == 1) or
                    (self.input_x_shape[1] * self.input_x_shape[2] == 1)):
                return True
        return False

    def _get_tiling_param_d(self):
        x_len = self._get_input_x_len()
        block_num = BLOCK_NUMBER_SIZE[self.input_x_dtype] * self.core_num
        if x_len <= block_num:
            self.core_num = 1
            each_core_len = _align_with_value(x_len, BLOCK_NUMBER_SIZE[self.input_x_dtype])
            each_core_cycle = _ceil_with_value(each_core_len, self.each_core_max_ub_size)
            last_core_res_len = 0
            last_core_cycle = 1
        else:
            each_core_len = _align_with_value(x_len // self.core_num, BLOCK_NUMBER_SIZE[self.input_x_dtype])
            self.core_num = x_len // each_core_len
            each_core_cycle = _ceil_with_value(each_core_len, self.each_core_max_ub_size)
            last_core_res_len = x_len - self.core_num * each_core_len
            if x_len % each_core_len > 0:
                self.core_num = self.core_num + 1
            last_core_cycle = _ceil_with_value(last_core_res_len, self.each_core_max_ub_size)

        batch_cycle = 1
        return {"tiling_mode": TILING_MODE["TILING_D"],
                "core_num": self.core_num,
                "loop_cycle": batch_cycle,
                "each_core_len": each_core_len,
                "each_core_cycle": each_core_cycle,
                "last_core_res_len": last_core_res_len,
                "last_core_cycle": last_core_cycle}

    def _get_tiling_param_n(self):
        len_n = self.input_x_shape[0]
        len_res = self._get_input_x_len() // len_n
        if len_res == 1:
            return self._get_tiling_param_d()
        else:
            if len_n <= self.core_num:
                self.core_num = len_n
                batch_cycle = 1
            else:
                batch_cycle = _ceil_with_value(len_n, self.core_num)
            each_core_len = batch_cycle * len_res
            each_core_cycle = batch_cycle
            each_core_res_len = (len_n - batch_cycle * self.core_num) * len_res
            last_core_cycle = len_n - batch_cycle * self.core_num
            return {"tiling_mode": TILING_MODE["TILING_N"],
                    "core_num": self.core_num,
                    "batch_cycle": batch_cycle,
                    "each_core_len": each_core_len,
                    "each_core_cycle": each_core_cycle,
                    "last_core_res_len": each_core_res_len,
                    "last_core_cycle": last_core_cycle}

    def _get_tiling_param_nc(self):
        len_n = self.input_x_shape[0]
        len_c = self.input_x_shape[1]
        len_n_c = len_n * len_c
        len_res = self._get_input_x_len() // len_n_c
        if len_n_c <= self.core_num:
            self.core_num = len_n_c
            batch_cycle = 1
        else:
            batch_cycle = _ceil_with_value(len_n_c, self.core_num)
            self.core_num = _ceil_with_value(len_n_c, batch_cycle)
        each_core_len = batch_cycle * len_res
        each_core_cycle = _ceil_with_value(each_core_len, self.each_core_max_ub_size)
        last_core_res_len = self._get_input_x_len() - each_core_len * (self.core_num - 1)
        last_core_cycle = _ceil_with_value(last_core_res_len, self.each_core_max_ub_size)
        return {"tiling_mode": TILING_MODE["TILING_NC"],
                "core_num": self.core_num,
                "batch_cycle": batch_cycle,
                "each_core_len": each_core_len,
                "each_core_cycle": each_core_cycle,
                "last_core_res_len": last_core_res_len,
                "last_core_cycle": last_core_cycle}

    def _get_tiling_param_nd(self):
        len_n = self.input_x_shape[0]
        len_d = self.input_x_shape[1]
        len_n_d = len_n * len_d
        core_num_d = self.core_num // len_n
        if len_n_d <= self.core_num:
            core_num = len_n_d
            batch_cycle = 1
        else:
            core_num = core_num_d * len_n
            batch_cycle = _ceil_with_value(len_n_d, core_num)
        each_core_len = batch_cycle
        each_core_cycle = _ceil_with_value(each_core_len, self.each_core_max_ub_size)
        last_core_res_len = len_d - each_core_len * (core_num_d - 1)
        last_core_cycle = _ceil_with_value(last_core_res_len, self.each_core_max_ub_size)
        return {"tiling_mode": TILING_MODE["TILING_NC"],
                "core_num": core_num,
                "batch_cycle": batch_cycle,
                "each_core_len": each_core_len,
                "each_core_cycle": each_core_cycle,
                "last_core_res_len": last_core_res_len,
                "last_core_cycle": last_core_cycle}

    def _get_tiling_param(self):
        if self._is_single_dim_tiling() is True:
            tiling_param = self._get_tiling_param_d()
        elif self.input_dim <= 3:
            if self.input_x_shape[0] >= (self.core_num // 2):
                tiling_param = self._get_tiling_param_n()
            else:
                if self.input_dim == 2:
                    tiling_param = self._get_tiling_param_nd()
                else:
                    tiling_param = self._get_tiling_param_nc()
        else:
            raise RuntimeError("tiling mode is not support yet!")
        return tiling_param

    def _data_move_src_buf_function(self, loop_id):
        """data_move_mte2_function

        Parameters
        ----------
        loop_id : int
            loop index

        Returns
        -------
        result : list
            [start_ub, end_ub, value_ub]
        """
        start_block_num = _ceil_with_value(self.start_shape[1], BLOCK_NUMBER_SIZE[self.start_dtype])
        start_ub = self.tik_inst.Tensor(
            self.start_dtype, (start_block_num * BLOCK_NUMBER_SIZE[self.start_dtype],),
            name="start_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(start_ub, self.start_gm[loop_id, 0], 0,
                                DEFAULT_NBURST, start_block_num, 0, 0)

        end_block_num = _ceil_with_value(self.end_shape[1], BLOCK_NUMBER_SIZE[self.end_dtype])
        end_ub = self.tik_inst.Tensor(
            self.end_dtype, (end_block_num * BLOCK_NUMBER_SIZE[self.end_dtype],),
            name="end_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(end_ub, self.end_gm[loop_id, 0], 0,
                                DEFAULT_NBURST, end_block_num, 0, 0)

        value_block_num = _ceil_with_value(self.value_shape[0], BLOCK_NUMBER_SIZE[self.value_dtype])
        value_ub = self.tik_inst.Tensor(
            self.value_dtype, (value_block_num * BLOCK_NUMBER_SIZE[self.value_dtype],),
            name="value_ub", scope=tik.scope_ubuf)
        self.tik_inst.data_move(value_ub, self.mask_value_gm[loop_id], 0,
                                DEFAULT_NBURST, value_block_num, 0, 0)

        src_ub_list = {"start_ub": start_ub, "end_ub": end_ub, "value_ub": value_ub}
        return src_ub_list

    def _data_move_mte_function(self, dst_buf, src_buf, data_length, block_number):
        """
        _data_move_mte_function

        Parameters
        ----------
        dst_buf : destination buffer
        src_buf : source buffer
        data_length: data move length
        block_number: block length

        Returns
        -------
        result
        """
        repeat_times = _ceil_with_value(data_length, block_number)
        if repeat_times > 0:
            if repeat_times > MAX_REPEAT_TIME:
                loop_num = repeat_times // MAX_REPEAT_TIME
                loop_res = repeat_times - loop_num * MAX_REPEAT_TIME
                loop_iter_num = MAX_REPEAT_TIME * block_number
                with self.tik_inst.for_range(0, loop_num) as loop_i:
                    self.tik_inst.data_move(dst_buf[loop_iter_num * loop_i],
                                            src_buf[loop_iter_num * loop_i],
                                            0, DEFAULT_NBURST, repeat_times, 0, 0)
                self.tik_inst.data_move(dst_buf[loop_iter_num * loop_num],
                                        src_buf[loop_iter_num * loop_num],
                                        0, DEFAULT_NBURST, loop_res, 0, 0)
            else:
                self.tik_inst.data_move(dst_buf, src_buf, 0, DEFAULT_NBURST, repeat_times, 0, 0)

    def _init_gm_tensor(self):
        """
        init_gm_tensor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.input_x_gm = self.tik_inst.Tensor(
            self.input_x_dtype, self.input_x_ori_shape, name="x", scope=tik.scope_gm)
        self.start_gm = self.tik_inst.Tensor(
            self.start_dtype, self.start_ori_shape, name="start", scope=tik.scope_gm)
        self.end_gm = self.tik_inst.Tensor(
            self.end_dtype, self.end_ori_shape, name="end", scope=tik.scope_gm)
        self.mask_value_gm = self.tik_inst.Tensor(
            self.value_dtype, self.value_ori_shape, name="value", scope=tik.scope_gm)
        self.output_y_gm = self.tik_inst.Tensor(
            self.output_y_dtype, self.output_y_ori_shape, name="y", scope=tik.scope_gm)

    def _fill_mask_info_1d(self, input_ub, start_address, length):
        """
        fill_mask_info_1d

        Parameters
        input_ub: input UB buffer
        start_address: masked fill start address
        length: masked fill length

        Returns
        -------
        None
        """
        start_mask_id = self.tik_inst.Scalar(INT32, name="start_mask_id")
        end_mask_id = self.tik_inst.Scalar(INT32, name="end_mask_id")
        scalar_value = self.tik_inst.Scalar(self.value_dtype, "scalar_value")
        with self.tik_inst.for_range(0, self.loop_cycle) as loop_id:
            src_ub_list = self._data_move_src_buf_function(loop_id)
            scalar_value.set_as(src_ub_list["value_ub"][0])
            start_mask_id.set_as(src_ub_list["start_ub"][0])
            end_mask_id.set_as(src_ub_list["end_ub"][0])
            start_id, end_id = self._get_start_end_pos(
                start_mask_id, end_mask_id, start_address, length)
            self._fill_single_mask_info(input_ub, start_id, end_id, scalar_value)

    def _fill_mask_info_3d_tiling_n(self, input_ub, start_address, length, batch_id, channel_id):
        """
        fill_mask in NCD shape with tiling batch

        Parameters
        input_ub: input UB buffer
        start_address: masked fill start address
        length: masked fill length
        batch_id: batch index
        channel_id: channel index
        Returns
        -------
        None
        """
        start_mask_id = self.tik_inst.Scalar(self.start_dtype, name="start_mask_id")
        end_mask_id = self.tik_inst.Scalar(self.end_dtype, name="end_mask_id")
        scalar_value = self.tik_inst.Scalar(self.value_dtype, "scalar_value")
        with self.tik_inst.for_range(0, self.loop_cycle) as loop_id:
            src_ub_list = self._data_move_src_buf_function(loop_id)
            scalar_value.set_as(src_ub_list["value_ub"][0])
            start_mask_id.set_as(src_ub_list["start_ub"][batch_id])
            end_mask_id.set_as(src_ub_list["end_ub"][batch_id])

            if self.absolute_axis == 0:
                with self.tik_inst.if_scope(tik.all(batch_id >= start_mask_id, batch_id < end_mask_id)):
                    self._fill_single_mask_info(input_ub, 0, length, scalar_value)
            elif self.absolute_axis == 1:
                with self.tik_inst.if_scope(tik.all(channel_id >= start_mask_id, channel_id < end_mask_id)):
                    self._fill_single_mask_info(input_ub, 0, length, scalar_value)
            else:
                start_id, end_id = self._get_start_end_pos(
                    start_mask_id, end_mask_id, start_address, length)
                self._fill_single_mask_info(input_ub, start_id, end_id, scalar_value)

    def _fill_single_mask_info(self, input_ub, start_idx, end_idx, value):
        with self.tik_inst.if_scope(end_idx > start_idx):
            scalar_len = self.tik_inst.Scalar(INT32, "scalar_len")
            start_align_id = self.tik_inst.Scalar(INT32, "start_align_id")
            end_align_id = self.tik_inst.Scalar(INT32, "end_align_id")
            block_num = self.tik_inst.Scalar(INT32, "block_num")

            scalar_len.set_as(end_idx - start_idx)
            block_num.set_as(BLOCK_NUMBER_SIZE[self.input_x_dtype])
            start_align_id.set_as((start_idx + block_num - 1) // block_num * block_num)
            end_align_id.set_as((end_idx // block_num) * block_num)

            with self.tik_inst.if_scope(end_align_id > start_align_id):
                with self.tik_inst.for_range(start_idx, start_align_id) as loop_i:
                    input_ub[loop_i].set_as(value)
                with self.tik_inst.for_range(end_align_id, end_idx) as loop_j:
                    input_ub[loop_j].set_as(value)
                if self.input_x_dtype == INT8:
                    self._vector_dup_int8_function(input_ub[start_align_id], end_align_id - start_align_id, value)
                else:
                    self._vector_dup_function(input_ub[start_align_id], end_align_id - start_align_id, value,
                                              self.input_x_dtype)
            with self.tik_inst.else_scope():
                with self.tik_inst.for_range(start_idx, end_idx) as loop_i:
                    input_ub[loop_i].set_as(value)

    def _vector_dup_function(self, input_ub, length, value, data_type):
        repeat_mask_time = self.tik_inst.Scalar(INT32, "repeat_mask_time")
        max_loop_cycle = self.tik_inst.Scalar(INT32, "max_loop_cycle")
        res_max_repeat_time = self.tik_inst.Scalar(INT32, "res_max_repeat_time")
        res_len = self.tik_inst.Scalar(INT32, "res_len")
        block_num = BLOCK_NUMBER_SIZE[data_type]
        vec_mask = VEC_MASK[data_type]
        max_mask_len = vec_mask * MAX_MASK_REPEAT_TIME

        max_loop_cycle.set_as(length // max_mask_len)
        with self.tik_inst.if_scope(max_loop_cycle > 0):
            with self.tik_inst.for_range(0, max_loop_cycle) as loop_i:
                self.tik_inst.vector_dup(vec_mask, input_ub[loop_i * max_mask_len],
                                         value, MAX_MASK_REPEAT_TIME, 0, 8)

        res_len.set_as(length - max_loop_cycle * max_mask_len)
        with self.tik_inst.if_scope(res_len > 0):
            repeat_mask_time.set_as(res_len // vec_mask)
            res_max_repeat_time.set_as((res_len - repeat_mask_time * vec_mask) // block_num)
            with self.tik_inst.if_scope(repeat_mask_time > 0):
                self.tik_inst.vector_dup(vec_mask,
                                         input_ub[max_loop_cycle * max_mask_len],
                                         value, repeat_mask_time, 0, 8)
            with self.tik_inst.if_scope(res_max_repeat_time > 0):
                self.tik_inst.vector_dup(block_num,
                                         input_ub[repeat_mask_time * vec_mask + max_loop_cycle * max_mask_len],
                                         value, res_max_repeat_time, 0, 1)

    def _vector_dup_int8_function(self, input_ub, length, value):
        max_loop_cycle = self.tik_inst.Scalar(INT32, "max_loop_cycle")
        block_num = BLOCK_NUMBER_SIZE[INT8]
        max_mask_len = BLOCK_NUMBER_SIZE[INT8]

        input_fp16_ub = self.tik_inst.Tensor(FP16, (max_mask_len,),
                                             name="input_fp16_ub", scope=tik.scope_ubuf)
        value_fp16_ub = self.tik_inst.Tensor(FP16, (BLOCK_NUMBER_SIZE[FP16],),
                                             name="value_fp16_ub", scope=tik.scope_ubuf)
        value_int8_ub = self.tik_inst.Tensor(INT8, (max_mask_len,),
                                             name="value_int8_ub", scope=tik.scope_ubuf)

        value_int8_ub[0] = value
        self.tik_inst.vconv(BLOCK_NUMBER_SIZE[FP16], 'none', value_fp16_ub, value_int8_ub, 1, 1, 1, 1, 1)

        scalar_value = self.tik_inst.Scalar(FP16, "scalar_value")
        scalar_value.set_as(value_fp16_ub[0])

        max_loop_cycle.set_as(length // max_mask_len)
        with self.tik_inst.if_scope(max_loop_cycle > 0):
            with self.tik_inst.for_range(0, max_loop_cycle) as loop_i:
                self.tik_inst.vector_dup(block_num, input_fp16_ub, scalar_value, 1, 0, 1)
                self.tik_inst.vconv(block_num, 'none', input_ub[loop_i * max_mask_len],
                                    input_fp16_ub, 1, 1, 1, 1, 1)

    def _get_start_end_pos(self, start_mask_id, end_mask_id, start_address, length):
        start_id = self.tik_inst.Scalar(INT32, name="start_id", init_value=0)
        end_id = self.tik_inst.Scalar(INT32, name="end_id", init_value=0)
        start_id.set_as(start_address)
        end_id.set_as(start_address + length)
        with self.tik_inst.if_scope(tik.any(end_id < start_mask_id, start_id > end_mask_id)):
            start_id.set_as(start_address)
            end_id.set_as(start_address)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(start_id < start_mask_id):
                start_id.set_as(start_mask_id)
            with self.tik_inst.if_scope(end_id > end_mask_id):
                end_id.set_as(end_mask_id)
        return start_id - start_address, end_id - start_address

    def _masked_fill_range_nd_function(self, dst_buf, src_ub, src_gm, size, start_address, length,
                                       batch_id, channel_id, data_type, input_dim):
        self._data_move_mte_function(src_ub, src_gm, size, BLOCK_NUMBER_SIZE[data_type])
        if input_dim == 1:
            self._fill_mask_info_1d(src_ub, start_address, length)
        else:
            self._fill_mask_info_3d_tiling_n(src_ub, start_address, length, batch_id, channel_id)
        self._data_move_mte_function(dst_buf, src_ub, size, BLOCK_NUMBER_SIZE[data_type])

    def _each_core_process_1_d(self, block_id):
        core_num = self.tiling_param_list["core_num"]
        each_core_len = self.tiling_param_list["each_core_len"]
        each_core_cycle = self.tiling_param_list["each_core_cycle"]
        last_core_res_len = self.tiling_param_list["last_core_res_len"]
        last_core_cycle = self.tiling_param_list["last_core_cycle"]
        each_core_one_cycle_len = _align_with_value(each_core_len // each_core_cycle,
                                                    BLOCK_NUMBER_SIZE[self.input_x_dtype])
        last_core_res_one_cycle_len = 0
        if last_core_cycle > 0:
            last_core_res_one_cycle_len = _align_with_value(last_core_res_len // last_core_cycle,
                                                            BLOCK_NUMBER_SIZE[self.input_x_dtype])
        input_x_ub = self.tik_inst.Tensor(self.input_x_dtype, (self.each_core_max_ub_size,),
                                          name="input_x_ub", scope=tik.scope_ubuf)

        start_address = self.tik_inst.Scalar(INT32, name="start_address")

        with self.tik_inst.if_scope(tik.all(block_id > 1, block_id == (core_num - 1))):
            if last_core_res_len > 0:
                with self.tik_inst.for_range(0, last_core_cycle) as last_cycle_id:
                    start_address.set_as(block_id * each_core_len + last_cycle_id * last_core_res_one_cycle_len)
                    self._masked_fill_range_nd_function(self.output_y_gm[start_address], input_x_ub,
                                                        self.input_x_gm[start_address],
                                                        last_core_res_one_cycle_len, start_address,
                                                        last_core_res_one_cycle_len, 0, 0, self.input_x_dtype,
                                                        self.input_dim)
        with self.tik_inst.else_scope():
            with self.tik_inst.for_range(0, each_core_cycle) as cycle_id:
                start_address.set_as(block_id * each_core_len + cycle_id * each_core_one_cycle_len)
                self._masked_fill_range_nd_function(self.output_y_gm[start_address], input_x_ub,
                                                    self.input_x_gm[start_address],
                                                    each_core_one_cycle_len, start_address,
                                                    each_core_one_cycle_len, 0, 0, self.input_x_dtype, self.input_dim)

    def _each_core_process_2_d_tiling_n_d(self, block_id, len_n, len_d, data_type):
        core_num = self.tiling_param_list["core_num"]
        each_core_len = self.tiling_param_list["each_core_len"]
        each_core_cycle = self.tiling_param_list["each_core_cycle"]
        last_core_res_len = self.tiling_param_list["last_core_res_len"]
        last_core_cycle = self.tiling_param_list["last_core_cycle"]
        input_x_ub = self.tik_inst.Tensor(data_type, (self.each_core_max_ub_size,),
                                          name="input_x_ub", scope=tik.scope_ubuf)
        start_address = self.tik_inst.Scalar(INT32, name="start_address")
        cur_channel = self.tik_inst.Scalar(INT32, name="cur_channel")
        dim_address = self.tik_inst.Scalar(INT32, name="dim_address")

        core_num_d = core_num // len_n
        length = len_d
        cur_channel.set_as(block_id * each_core_len)
        batch_id = cur_channel // len_d
        channel_id = cur_channel % len_d
        start_address.set_as(batch_id * length + channel_id)
        thread_num_loop = 1

        with self.tik_inst.if_scope(batch_id < len_n):
            with self.tik_inst.if_scope(
                    tik.all(block_id > 0, (block_id % (core_num_d - 1)) == 0, last_core_res_len > 0)):
                if last_core_cycle > 0:
                    if last_core_cycle % THREAD_NUM == 0:
                        thread_num_loop = THREAD_NUM
                    each_loop_len = _ceil_with_value(last_core_res_len, last_core_cycle)
                    dim_address.set_as(channel_id)
                    with self.tik_inst.for_range(0, last_core_cycle, thread_num=thread_num_loop) as loop_d:
                        self._masked_fill_range_nd_function(
                            self.output_y_gm[start_address + loop_d * self.each_core_max_ub_size],
                            input_x_ub, self.input_x_gm[start_address + loop_d * self.each_core_max_ub_size],
                            each_loop_len, dim_address, each_loop_len, batch_id, channel_id, data_type, self.input_dim)
            with self.tik_inst.else_scope():
                each_loop_len = _ceil_with_value(each_core_len, each_core_cycle)
                if each_core_cycle % THREAD_NUM == 0:
                    thread_num_loop = THREAD_NUM
                with self.tik_inst.if_scope(tik.all(block_id > 0, (block_id % core_num_d) == 0)):
                    dim_address.set_as(start_address % len_d)
                with self.tik_inst.else_scope():
                    dim_address.set_as(channel_id)
                with self.tik_inst.for_range(0, each_core_cycle, thread_num=thread_num_loop) as loop_d:
                    self._masked_fill_range_nd_function(
                        self.output_y_gm[start_address + loop_d * self.each_core_max_ub_size],
                        input_x_ub, self.input_x_gm[start_address + loop_d * self.each_core_max_ub_size],
                        each_loop_len, dim_address, each_loop_len, batch_id, channel_id, data_type, self.input_dim)

    def _each_core_process_2_d_tiling_n(self, block_id, len_d, data_type):
        core_num = self.tiling_param_list["core_num"]
        batch_cycle = self.tiling_param_list["batch_cycle"]
        each_core_len = self.tiling_param_list["each_core_len"]
        each_core_cycle = self.tiling_param_list["each_core_cycle"]
        last_core_res_len = self.tiling_param_list["last_core_res_len"]
        last_core_cycle = self.tiling_param_list["last_core_cycle"]
        input_x_ub = self.tik_inst.Tensor(data_type, (self.each_core_max_ub_size,),
                                          name="input_x_ub", scope=tik.scope_ubuf)
        length = len_d
        start_address = self.tik_inst.Scalar(INT32, name="start_address")
        loop_d_num = len_d // self.each_core_max_ub_size
        loop_d_res = len_d - loop_d_num * self.each_core_max_ub_size
        res_address = loop_d_num * self.each_core_max_ub_size
        thread_num_loop = 1

        with self.tik_inst.if_scope(tik.all(block_id == (core_num - 1), last_core_res_len > 0)):
            if last_core_cycle > 0:
                if last_core_cycle % THREAD_NUM == 0:
                    thread_num_loop = THREAD_NUM
                with self.tik_inst.for_range(0, last_core_cycle, thread_num=thread_num_loop) as last_cycle_id:
                    start_address.set_as(block_id * each_core_len + last_cycle_id * length)
                    if loop_d_num > 0:
                        with self.tik_inst.for_range(0, loop_d_num) as loop_d:
                            self._masked_fill_range_nd_function(
                                self.output_y_gm[start_address + loop_d * self.each_core_max_ub_size],
                                input_x_ub, self.input_x_gm[start_address + loop_d * self.each_core_max_ub_size],
                                self.each_core_max_ub_size, loop_d * self.each_core_max_ub_size,
                                self.each_core_max_ub_size, block_id * batch_cycle + last_cycle_id, 0, data_type,
                                self.input_dim)
                    self._masked_fill_range_nd_function(
                        self.output_y_gm[start_address + res_address],
                        input_x_ub, self.input_x_gm[start_address + res_address],
                        loop_d_res, res_address, loop_d_res, block_id * batch_cycle + last_cycle_id, 0, data_type,
                        self.input_dim)
        with self.tik_inst.else_scope():
            if each_core_cycle % THREAD_NUM == 0:
                thread_num_loop = THREAD_NUM
            with self.tik_inst.for_range(0, each_core_cycle, thread_num=thread_num_loop) as cycle_id:
                start_address.set_as(block_id * each_core_len + cycle_id * length)
                if loop_d_num > 0:
                    with self.tik_inst.for_range(0, loop_d_num) as loop_d:
                        self._masked_fill_range_nd_function(
                            self.output_y_gm[start_address + loop_d * self.each_core_max_ub_size],
                            input_x_ub, self.input_x_gm[start_address + loop_d * self.each_core_max_ub_size],
                            self.each_core_max_ub_size, loop_d * self.each_core_max_ub_size,
                            self.each_core_max_ub_size, block_id * batch_cycle + cycle_id, 0, data_type,
                            self.input_dim)
                self._masked_fill_range_nd_function(
                    self.output_y_gm[start_address + res_address],
                    input_x_ub, self.input_x_gm[start_address + res_address],
                    loop_d_res, res_address, loop_d_res, block_id * batch_cycle + cycle_id, 0, data_type,
                    self.input_dim)

    def _each_core_process_2_d(self, block_id):
        tiling_mode = self.tiling_param_list["tiling_mode"]
        if tiling_mode == TILING_MODE["TILING_N"]:
            self._each_core_process_2_d_tiling_n(block_id, self.input_x_shape[1], self.input_x_dtype)
        else:
            if self.absolute_axis == 1:
                self.absolute_axis = 2
            self._each_core_process_2_d_tiling_n_d(block_id, self.input_x_ori_shape[0],
                                                   self.input_x_ori_shape[1], self.input_x_dtype)

    def _each_core_process_3_d_tiling_n(self, block_id, len_c, len_d, data_type):
        core_num = self.tiling_param_list["core_num"]
        batch_cycle = self.tiling_param_list["batch_cycle"]
        each_core_len = self.tiling_param_list["each_core_len"]
        each_core_cycle = self.tiling_param_list["each_core_cycle"]
        last_core_res_len = self.tiling_param_list["last_core_res_len"]
        last_core_cycle = self.tiling_param_list["last_core_cycle"]
        input_x_ub = self.tik_inst.Tensor(data_type, (self.each_core_max_ub_size,),
                                          name="input_x_ub", scope=tik.scope_ubuf)
        length = len_c * len_d
        start_address = self.tik_inst.Scalar(INT32, name="start_address")
        loop_d_num = len_d // self.each_core_max_ub_size
        loop_d_res = len_d - loop_d_num * self.each_core_max_ub_size
        res_address = loop_d_num * self.each_core_max_ub_size

        if len_c % THREAD_NUM != 0:
            thread_num_loop = 1
        else:
            thread_num_loop = THREAD_NUM

        with self.tik_inst.if_scope(tik.all(block_id == (core_num - 1), last_core_res_len > 0)):
            with self.tik_inst.for_range(0, last_core_cycle) as last_cycle_id:
                with self.tik_inst.for_range(0, len_c, thread_num=thread_num_loop) as loop_c:
                    start_address.set_as(block_id * each_core_len + last_cycle_id * length + loop_c * len_d)
                    if loop_d_num > 0:
                        with self.tik_inst.for_range(0, loop_d_num) as loop_d:
                            self._masked_fill_range_nd_function(
                                self.output_y_gm[start_address + loop_d * self.each_core_max_ub_size],
                                input_x_ub, self.input_x_gm[start_address + loop_d * self.each_core_max_ub_size],
                                self.each_core_max_ub_size, loop_d * self.each_core_max_ub_size,
                                self.each_core_max_ub_size, block_id * batch_cycle + last_cycle_id, loop_c, data_type,
                                self.input_dim)
                    self._masked_fill_range_nd_function(
                        self.output_y_gm[start_address + res_address],
                        input_x_ub, self.input_x_gm[start_address + res_address],
                        loop_d_res, res_address, loop_d_res, block_id * batch_cycle + last_cycle_id, loop_c, data_type,
                        self.input_dim)
        with self.tik_inst.else_scope():
            with self.tik_inst.for_range(0, each_core_cycle) as cycle_id:
                with self.tik_inst.for_range(0, len_c, thread_num=thread_num_loop) as loop_c:
                    start_address.set_as(block_id * each_core_len + cycle_id * length + loop_c * len_d)
                    if loop_d_num > 0:
                        with self.tik_inst.for_range(0, loop_d_num) as loop_d:
                            self._masked_fill_range_nd_function(
                                self.output_y_gm[start_address + loop_d * self.each_core_max_ub_size],
                                input_x_ub, self.input_x_gm[start_address + loop_d * self.each_core_max_ub_size],
                                self.each_core_max_ub_size, loop_d * self.each_core_max_ub_size,
                                self.each_core_max_ub_size, block_id * batch_cycle + cycle_id, loop_c, data_type,
                                self.input_dim)
                    self._masked_fill_range_nd_function(
                        self.output_y_gm[start_address + res_address],
                        input_x_ub, self.input_x_gm[start_address + res_address],
                        loop_d_res, res_address, loop_d_res, block_id * batch_cycle + cycle_id, loop_c, data_type,
                        self.input_dim)

    def _each_core_process_3_d_tiling_n_c(self, block_id, len_c, len_d, data_type):
        core_num = self.tiling_param_list["core_num"]
        batch_cycle = self.tiling_param_list["batch_cycle"]
        last_core_res_len = self.tiling_param_list["last_core_res_len"]
        input_x_ub = self.tik_inst.Tensor(data_type, (self.each_core_max_ub_size,),
                                          name="input_x_ub", scope=tik.scope_ubuf)
        length = len_c * len_d
        start_address = self.tik_inst.Scalar(INT32, name="start_address")
        loop_d_num = len_d // self.each_core_max_ub_size
        loop_d_res = len_d - loop_d_num * self.each_core_max_ub_size
        res_address = loop_d_num * self.each_core_max_ub_size
        cur_channel = self.tik_inst.Scalar(INT32, name="cur_channel")
        thread_num_loop = 1

        with self.tik_inst.if_scope(tik.all(block_id == (core_num - 1), last_core_res_len > 0)):
            if last_core_res_len > 0:
                res_batch_cycle = last_core_res_len // len_d
                if res_batch_cycle % THREAD_NUM == 0:
                    thread_num_loop = THREAD_NUM
                with self.tik_inst.for_range(0, res_batch_cycle, thread_num=thread_num_loop) as loop_c:
                    cur_channel.set_as(block_id * batch_cycle + loop_c)
                    batch_id = cur_channel // len_c
                    channel_id = cur_channel % len_c
                    start_address.set_as(batch_id * length + channel_id * len_d)
                    if loop_d_num > 0:
                        with self.tik_inst.for_range(0, loop_d_num) as loop_d:
                            self._masked_fill_range_nd_function(
                                self.output_y_gm[start_address + loop_d * self.each_core_max_ub_size],
                                input_x_ub, self.input_x_gm[start_address + loop_d * self.each_core_max_ub_size],
                                self.each_core_max_ub_size, loop_d * self.each_core_max_ub_size,
                                self.each_core_max_ub_size, batch_id, channel_id, data_type, self.input_dim)
                    self._masked_fill_range_nd_function(
                        self.output_y_gm[start_address + res_address],
                        input_x_ub, self.input_x_gm[start_address + res_address],
                        loop_d_res, res_address, loop_d_res, batch_id, channel_id, data_type, self.input_dim)
        with self.tik_inst.else_scope():
            if batch_cycle % THREAD_NUM == 0:
                thread_num_loop = THREAD_NUM
            with self.tik_inst.for_range(0, batch_cycle, thread_num=thread_num_loop) as loop_c:
                cur_channel.set_as(block_id * batch_cycle + loop_c)
                batch_id = cur_channel // len_c
                channel_id = cur_channel % len_c
                start_address.set_as(batch_id * length + channel_id * len_d)
                if loop_d_num > 0:
                    with self.tik_inst.for_range(0, loop_d_num) as loop_d:
                        self._masked_fill_range_nd_function(
                            self.output_y_gm[start_address + loop_d * self.each_core_max_ub_size],
                            input_x_ub, self.input_x_gm[start_address + loop_d * self.each_core_max_ub_size],
                            self.each_core_max_ub_size, loop_d * self.each_core_max_ub_size,
                            self.each_core_max_ub_size, batch_id, channel_id, data_type, self.input_dim)
                self._masked_fill_range_nd_function(
                    self.output_y_gm[start_address + res_address],
                    input_x_ub, self.input_x_gm[start_address + res_address],
                    loop_d_res, res_address, loop_d_res, batch_id, channel_id, data_type, self.input_dim)

    def _each_core_process_3_d(self, block_id):
        tiling_mode = self.tiling_param_list["tiling_mode"]
        if tiling_mode == TILING_MODE["TILING_N"]:
            self._each_core_process_3_d_tiling_n(block_id, self.input_x_ori_shape[1], self.input_x_ori_shape[2],
                                                 self.input_x_dtype)
        elif tiling_mode == TILING_MODE["TILING_NC"]:
            self._each_core_process_3_d_tiling_n_c(block_id, self.input_x_ori_shape[1], self.input_x_ori_shape[2],
                                                   self.input_x_dtype)
        else:
            raise RuntimeError("process NCD shape's tiling mode is not support yet!")

    def _each_core_process(self, block_id):
        """
        each_core process enter function
        Parameters
        ----------
        block_id : int, core index

        Returns
        -------
        None
        """
        if self._is_single_dim_tiling() is True:
            self._each_core_process_1_d(block_id)
        elif self.input_dim == 2:
            self._each_core_process_2_d(block_id)
        elif self.input_dim == 3:
            self._each_core_process_3_d(block_id)
        else:
            pass

    def tik_inst_function(self):
        """
        tik_inst_function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_inst.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            self._each_core_process(block_id)
        self.tik_inst.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=[self.input_x_gm, self.start_gm, self.end_gm, self.mask_value_gm], outputs=[self.output_y_gm])


# 'pylint: disable=too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def masked_fill_range(x, start, end, value, y, axis=-1, kernel_name="MaskedFillRange"):
    """
    algorithm: masked_fill_range

    Parameters
    ----------
    x : dict
        input tensor of masked_fill
    start : dict
        shape and dtype of start position along to axis
    end : dict
        shape and dtype of end position along to axis
    value: dict
        shape and dtype of masked_fill value
    y: dict
        shape and dtype of output
    axis: attr
        attribute, generator mask along aixs
    kernel_name : str
        kernel name, default value is "GenRangeMask"

    Returns
    -------
    None
    """
    masked_fill_range_tik = MaskedFillRange(x, start, end, value, y, axis, kernel_name)
    masked_fill_range_tik.tik_inst_function()
    return masked_fill_range_tik.tik_inst
