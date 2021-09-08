# Copyright 2019 Huawei Technologies Co., Ltd
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
scatter_nd
"""
from functools import reduce

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

# max int64 value
MAX_INT64_VALUE = 2**64 - 1
# tiling param num
TILING_ARG_NUM = 36
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# 8 bit
EIGHT_BIT = 8
# bytes of one block
BLOCK_BYTES = 32
# the vector size is 256B
VECTOR_BYTE_SIZE = 256


def ceil_div(value_x, value_y):
    """
    do ceil division
    """
    return (value_x + value_y - 1) // value_y


def check_supported(indices, x, shape, y, kernel_name="ScatterNd"):
    """
    check support dynamiclly
    """
    x_dtype = x.get('dtype')
    indices_shape = list(indices.get('ori_shape'))
    x_shape = list(x.get('ori_shape'))
    shape_shape = list(shape.get('ori_shape'))

    if reduce(lambda a, b: a * b, indices_shape + x_shape + shape_shape) < 0:
        return True, "dynamic shape support"
    if indices_shape[-1] == 1:
        return True,"indices last dim is 1, support dynamic shape"
    data_size = 4
    if x_dtype in ("float32", "int32"):
        data_size = 4
    elif x_dtype in ("float16",):
        data_size = 2
    elif x_dtype in ("int8", "uint8"):
        data_size = 1
    shape_value = list(shape.get('const_value'))
    shape_value += [1]
    update_slice = reduce(lambda a, b: a * b, shape_value[indices_shape[-1]:])
    if data_size > 0 and update_slice > 0 and update_slice < (32 / data_size):
        return False, "ScatterNdD update slice < 32byte, graph not changed."
    return True, ""


# pylint: disable=too-many-public-methods,too-many-arguments,too-many-instance-attributes
# pylint: disable=too-many-lines,attribute-defined-outside-init,too-many-statements,too-many-branches,unused-argument
class ScatterNd():
    """
       Function: use to store scatter_nd base parameters
       Modify : 2021-01-21
    """

    def __init__(self, indices, x, shape, y, kernel_name):
        """
        Init ScatterNd parameters

        Parameters
        ----------
        indices: dict
            the dict of input tensor.
        x: dict
            the dict of input tensor.
        shape: dict
            the dict of input tensor.
        y: dict
            the dict of output tensor.
        kernel_name: str
            cce kernel name, default value is "scatter_nd".

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_dtype = x.get("dtype").lower()
        self.shape_dtype = shape.get("dtype").lower()
        self.out_dtype = y.get("dtype").lower()

        self.check_input_params()

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE)
        self.updates_dtype_bytes_size = tbe_platform.get_bit_len(self.updates_dtype) // EIGHT_BIT
        self.indices_dtype_bytes_size = tbe_platform.get_bit_len(self.indices_dtype) // EIGHT_BIT
        self.updates_data_each_block = BLOCK_BYTES // self.updates_dtype_bytes_size
        self.indices_data_each_block = BLOCK_BYTES // self.indices_dtype_bytes_size
        self.data_num_one_repeat = VECTOR_BYTE_SIZE // self.updates_dtype_bytes_size
        self.support_atomic = 0

        if tbe_platform.api_check_support("tik.set_atomic_add") and self.updates_dtype == "float32":
            self.support_atomic = 1
            self.updates_ub_num = self.ub_size_bytes // 2 // self.updates_dtype_bytes_size
            self.indices_ub_num = self.ub_size_bytes // 2 // self.indices_dtype_bytes_size
        else:
            self.var_ub_num = self.ub_size_bytes // 8 * 3 // self.updates_dtype_bytes_size
            self.updates_ub_num = self.ub_size_bytes // 8 * 3 // self.updates_dtype_bytes_size
            self.indices_ub_num = self.ub_size_bytes // 4 // self.indices_dtype_bytes_size

            self.var_ub = None
            self.var_tile_ub = None
            self.updates_tile_ub = None
            self.var_burst_len = None
            self.each_core_max_indice = None
            self.last_var_num = None
            self.var_offset = None

        self.out_gm = self.tik_instance.Tensor(self.updates_dtype, (MAX_INT64_VALUE,),
                                               name="out_gm",
                                               scope=tik.scope_gm,
                                               is_atomic_add=True)

        self.tiling_gm = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.shape_gm = self.tik_instance.Tensor("int32", (MAX_INT64_VALUE,), name="shape", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype, (MAX_INT64_VALUE,), name="indices_gm", scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.updates_dtype, (MAX_INT64_VALUE,),
                                                   name="updates_gm",
                                                   scope=tik.scope_gm)

        self.updates_ub = None
        self.indices_ub = None
        self.zero_ub = None
        self.var_read_index = None
        self.core_loop_index = None
        self.var_value = None
        self.update_value = None
        self.updates_burst_len = None
        self.indices_tmp = None
        self.offset = None

    def check_input_params(self):
        """
        Check whether the input parameters is valid or not
        """
        indices_support_dtype_list = ("int32", "int64")
        updates_support_dtype_list = ("float32", "int32", "float16")
        shape_support_dtype_list = ("int32",)
        para_check.check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        para_check.check_dtype(self.updates_dtype, updates_support_dtype_list, param_name="updates")
        para_check.check_dtype(self.shape_dtype, shape_support_dtype_list, param_name="shape")
        if self.updates_dtype != self.out_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "x", "y", self.updates_dtype,
                                                                  self.out_dtype)

    def tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from scatter_nd tiling

        Returns
        -------
        None
        """
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_mode")
        self.indice_step = self.tik_instance.Scalar("int64", name="indice_step")
        self.core_num = self.tik_instance.Scalar("int64", name="core_num")
        self.update_data_num = self.tik_instance.Scalar("int64", name="update_data_num")
        self.indices_loop_num = self.tik_instance.Scalar("int64", name="indices_loop_num")
        self.indices_last_num = self.tik_instance.Scalar("int64", name="indices_last_num")
        self.updates_num = self.tik_instance.Scalar("int64", name="updates_num")
        self.updates_loop_num = self.tik_instance.Scalar("int64", name="updates_loop_num")
        self.updates_last_num = self.tik_instance.Scalar("int64", name="updates_last_num")
        self.var_num = self.tik_instance.Scalar("int64", name="var_num")
        self.var_loop_num = self.tik_instance.Scalar("int64", name="var_loop_num")
        self.var_last_num = self.tik_instance.Scalar("int64", name="var_last_num")
        self.var_each_core_burst_len = self.tik_instance.Scalar("int64", name="var_each_core_burst_len")
        self.var_last_core_burst_len = self.tik_instance.Scalar("int64", name="var_last_core_burst_len")
        self.max_indice = self.tik_instance.Scalar("int64", name="max_indice")
        self.var_each_core_data = self.tik_instance.Scalar("int64", name="var_each_core_data")
        self.indices_last_dim = self.tik_instance.Scalar("int64", name="indices_last_dim")
        self.var_each_core_set_zero_loop_num = self.tik_instance.Scalar("int64", name="var_each_core_set_zero_loop_num")
        self.var_each_core_set_zero_last_num = self.tik_instance.Scalar("int64", name="var_each_core_set_zero_last_num")
        self.var_last_core_set_zero_loop_num = self.tik_instance.Scalar("int64", name="var_last_core_set_zero_loop_num")
        self.var_last_core_set_zero_last_num = self.tik_instance.Scalar("int64", name="var_last_core_set_zero_last_num")
        self.indices_each_core_data = self.tik_instance.Scalar("int64", name="indices_each_core_data")
        self.indices_last_core_data = self.tik_instance.Scalar("int64", name="indices_last_core_data")
        self.each_core_indices_loop_num = self.tik_instance.Scalar("int64", name="each_core_indices_loop_num")
        self.each_core_indices_last_num = self.tik_instance.Scalar("int64", name="each_core_indices_last_num")
        self.last_core_indices_loop_num = self.tik_instance.Scalar("int64", name="last_core_indices_loop_num")
        self.last_core_indices_last_num = self.tik_instance.Scalar("int64", name="last_core_indices_last_num")

        self.tiling_mode.set_as(self.tiling_ub[0])
        self.indice_step.set_as(self.tiling_ub[1])
        self.core_num.set_as(self.tiling_ub[2])
        self.update_data_num.set_as(self.tiling_ub[3])
        self.indices_loop_num.set_as(self.tiling_ub[4])
        self.indices_last_num.set_as(self.tiling_ub[5])
        self.updates_num.set_as(self.tiling_ub[6])
        self.updates_loop_num.set_as(self.tiling_ub[7])
        self.updates_last_num.set_as(self.tiling_ub[8])
        self.var_num.set_as(self.tiling_ub[9])
        self.var_loop_num.set_as(self.tiling_ub[10])
        self.var_last_num.set_as(self.tiling_ub[11])
        self.var_each_core_burst_len.set_as(self.tiling_ub[12])
        self.var_last_core_burst_len.set_as(self.tiling_ub[13])
        self.max_indice.set_as(self.tiling_ub[14])
        self.var_each_core_data.set_as(self.tiling_ub[15])
        self.indices_last_dim.set_as(self.tiling_ub[16])
        self.var_each_core_set_zero_loop_num.set_as(self.tiling_ub[24])
        self.var_each_core_set_zero_last_num.set_as(self.tiling_ub[25])
        self.var_last_core_set_zero_loop_num.set_as(self.tiling_ub[26])
        self.var_last_core_set_zero_last_num.set_as(self.tiling_ub[27])
        self.indices_each_core_data.set_as(self.tiling_ub[28])
        self.indices_last_core_data.set_as(self.tiling_ub[29])
        self.each_core_indices_loop_num.set_as(self.tiling_ub[30])
        self.each_core_indices_last_num.set_as(self.tiling_ub[31])
        self.last_core_indices_loop_num.set_as(self.tiling_ub[32])
        self.last_core_indices_last_num.set_as(self.tiling_ub[33])

    def init_ub_tensor(self):
        """
        Compute the ub size of tensors

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not (tbe_platform.api_check_support("tik.set_atomic_add") and self.updates_dtype == "float32"):
            self.var_ub = self.tik_instance.Tensor(self.updates_dtype, (self.var_ub_num,),
                                                   name="var_ub",
                                                   scope=tik.scope_ubuf)
            self.var_tile_ub = self.tik_instance.Tensor(self.updates_dtype, (self.updates_data_each_block,),
                                                        name="var_tile_ub",
                                                        scope=tik.scope_ubuf)
            self.var_burst_len = self.tik_instance.Scalar("int32", name="var_burst_len")
            self.var_burst_len.set_as(0)
            self.each_core_max_indice = self.tik_instance.Scalar("int32", name="each_core_max_indice")
            self.each_core_max_indice.set_as(0)
            self.mask = self.tik_instance.Scalar("int32", name="mask")
            self.mask.set_as(0)
            self.repeat = self.tik_instance.Scalar("int32", name="repeat")
            self.repeat.set_as(0)
            self.var_last_num_off = self.tik_instance.Scalar("int32", name="var_last_num_off")
            self.var_last_num_off.set_as(0)
            self.last_var_num = self.tik_instance.Scalar("int32", name="last_var_num")
            self.last_var_num.set_as(0)
            self.var_offset = self.tik_instance.Scalar("int32", name="var_offset")
            self.var_offset.set_as(0)

        self.updates_tile_ub = self.tik_instance.Tensor(self.updates_dtype, (self.updates_data_each_block,),
                                                        name="updates_tile_ub",
                                                        scope=tik.scope_ubuf)
        self.updates_ub = self.tik_instance.Tensor(self.updates_dtype, (self.updates_ub_num,),
                                                   name="updates_ub",
                                                   scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(self.indices_dtype, (self.indices_ub_num,),
                                                   name="indices_ub",
                                                   scope=tik.scope_ubuf)
        self.zero_ub = self.tik_instance.Tensor(self.updates_dtype, (self.updates_data_each_block,),
                                                name="zero_ub",
                                                scope=tik.scope_ubuf)

        self.var_read_index = self.tik_instance.Scalar("int32", name="var_read_index")
        self.var_read_index.set_as(0)
        self.core_loop_index = self.tik_instance.Scalar("int32", name="core_loop_index")
        self.core_loop_index.set_as(0)
        self.var_value = self.tik_instance.Scalar(self.updates_dtype, name="var_value")
        self.var_value.set_as(0)
        self.update_value = self.tik_instance.Scalar(self.updates_dtype, name="update_value")
        self.update_value.set_as(0)
        self.updates_burst_len = self.tik_instance.Scalar("int32", name="updates_burst_len")
        self.updates_burst_len.set_as(0)
        self.indices_tmp = self.tik_instance.Scalar("int32", name="indices_tmp")
        self.indices_tmp.set_as(0)
        self.offset = self.tik_instance.Scalar("int32", name="offset")
        self.offset.set_as(0)

    def get_var_read_index(self, indices_ub_index):
        """
        Calculate the index of the read var

        Parameters
        ----------
        indices_ub_index: int32
            the index of the currently traversed indices in UB

        Returns
        -------
        None
        """
        indices_ub_index = indices_ub_index * self.indices_last_dim
        self.var_read_index.set_as(0)
        with self.tik_instance.if_scope(self.indices_last_dim == 1):
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.indices_last_dim) as index:
                self.indices_tmp.set_as(self.indices_ub[indices_ub_index + index])
                with self.tik_instance.if_scope(index + 1 < self.indices_last_dim):
                    self.offset.set_as(self.tiling_ub[17 + index])
                    self.var_read_index.set_as(self.var_read_index + self.indices_tmp * self.offset)
                with self.tik_instance.else_scope():
                    self.var_read_index.set_as(self.var_read_index + self.indices_tmp)

    def move_indices(self, indices_in_index, indice_num, mode):
        """
        Move indices, choose branch

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        indices_burst_len = ceil_div(indice_num, self.indices_data_each_block)
        self.tik_instance.data_move(self.indices_ub, self.indices_gm[indices_in_index], 0, 1, indices_burst_len, 0, 0)
        indices_in_index = indices_in_index // self.indices_last_dim

        if mode == 1:
            self.traversing_updates_32b_aligned_and_ub_enough_atomic(indices_in_index, indice_num)
        if mode == 2:
            self.circulate_indices(indices_in_index, indice_num, 2)
        if mode == 3:
            self.traversing_updates_single_core_and_ub_enough_atomic(indices_in_index, indice_num)
        if mode == 4:
            self.traversing_updates_single_core_and_ub_not_enough_atomic(indices_in_index, indice_num)
        if mode == 5:
            self.circulate_indices(indices_in_index, indice_num, 5)
        if mode == 6:
            self.traversing_32b_aligned_ub_store_all_var_and_update(indices_in_index, indice_num)
        if mode == 7:
            self.traversing_32b_aligned_ub_store_all_var(indices_in_index, indice_num)
        if mode == 8:
            self.traversing_32b_aligned_ub_store_all_update(indices_in_index, indice_num)
        if mode == 9:
            self.circulate_indices_not_atomic(indices_in_index, indice_num, 9)
        if mode == 10:
            self.traversing_less_than_one_block_single_core_var_and_update(indices_in_index, indice_num)
        if mode == 11:
            self.traversing_less_than_one_block_single_core_var(indices_in_index, indice_num)
        if mode == 12:
            self.traversing_less_than_one_block_single_core_update(indices_in_index, indice_num)
        if mode == 13:
            self.traversing_less_than_one_block_ub_not_enough(indices_in_index, indice_num)
        if mode == 14:
            self.circulate_indices_not_atomic(indices_in_index, indice_num, 14)
        if mode == 15:
            self.circulate_indices_not_atomic(indices_in_index, indice_num, 15)

    def circulate_indices(self, indices_in_index, indice_num, mode):
        """
        Circulate the index in the indices

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        mode: int32
            which branch
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.var_read_index.set_as(self.var_read_index * self.update_data_num)
                    update_loop_index = (indices_in_index + indices_ub_index) * self.update_data_num
                    self.traversing_updates(update_loop_index, mode)

    def traversing_updates(self, update_loop_index, mode):
        """
        Traversing the index in the updates

        Parameters
        ----------
        update_loop_index: int32
            Updates index on UB
        mode: int32
            which branch
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.updates_loop_num) as updates_loop_index:
            self.update_var(update_loop_index + updates_loop_index * self.updates_ub_num, self.updates_ub_num,
                            self.var_read_index + updates_loop_index * self.updates_ub_num, mode)

        with self.tik_instance.if_scope(self.updates_last_num > 0):
            self.update_var(update_loop_index + self.updates_loop_num * self.updates_ub_num, self.updates_last_num,
                            self.var_read_index + self.updates_loop_num * self.updates_ub_num, mode)

    def update_var(self, updates_loop_index, update_num, var_loop_index, mode):
        """
        Update the update fragment corresponding to the index

        Parameters
        ----------
        updates_loop_index: int32
            Updates index on GM
        update_num: int32
            the number of indexes in the updates on UB
        var_loop_index: int32
            Var index on GM
        Returns
        -------
        None
        """
        if mode == 2:
            self.updates_burst_len.set_as(update_num // self.updates_data_each_block)
            self.tik_instance.data_move(self.updates_ub, self.updates_gm[updates_loop_index], 0, 1,
                                        self.updates_burst_len, 0, 0)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(self.out_gm[var_loop_index], self.updates_ub, 0, 1, self.updates_burst_len, 0,
                                        0)
            self.tik_instance.set_atomic_add(0)

        if mode == 5:
            self.updates_burst_len.set_as(ceil_div(update_num, self.updates_data_each_block))
            self.tik_instance.data_move(self.updates_ub, self.updates_gm[updates_loop_index], 0, 1,
                                        self.updates_burst_len, 0, 0)

            with self.tik_instance.if_scope(update_num % self.updates_data_each_block == 0):
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.out_gm[var_loop_index], self.updates_ub, 0, 1, self.updates_burst_len,
                                            0, 0)
                self.tik_instance.set_atomic_add(0)
            with self.tik_instance.else_scope():
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.out_gm[var_loop_index], self.updates_ub, 0, 1,
                                            self.updates_burst_len - 1, 0, 0)
                self.tik_instance.set_atomic_add(0)
                with self.tik_instance.for_range(0, self.updates_data_each_block) as updates_ub_index:
                    self.update_value.set_as(self.updates_ub[update_num - self.updates_data_each_block +
                                                             updates_ub_index])
                    self.updates_tile_ub[updates_ub_index].set_as(self.update_value)
                self.tik_instance.vec_muls(
                    self.updates_data_each_block - self.update_data_num % self.updates_data_each_block,
                    self.updates_tile_ub, self.updates_tile_ub, 0, 1, 8, 8)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.out_gm[var_loop_index + update_num - self.updates_data_each_block],
                                            self.updates_tile_ub, 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)

    def traversing_updates_32b_aligned_and_ub_enough_atomic(self, indices_in_index, indice_num):
        """
        updateDataNum is 32B aligned, ub can store all updatesNum

        Parameters
        ----------
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        update_burst_len = self.updates_num // self.updates_data_each_block
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, update_burst_len, 0, 0)
        updates_burst_len = self.update_data_num // self.updates_data_each_block

        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move(
                        self.out_gm[self.var_read_index * self.update_data_num],
                        self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num], 0, 1,
                        updates_burst_len, 0, 0)
                    self.tik_instance.set_atomic_add(0)

    def traversing_updates_single_core_and_ub_enough_atomic(self, indices_in_index, indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block, ub can store all updatesNum

        Parameters
        ----------
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.updates_num % self.updates_data_each_block == 0):
            self.updates_burst_len.set_as(self.updates_num // self.updates_data_each_block)
        with self.tik_instance.else_scope():
            self.updates_burst_len.set_as(self.updates_num // self.updates_data_each_block + 1)
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, self.updates_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            self.tik_instance.data_move(self.updates_tile_ub,
                                        self.updates_gm[(indices_in_index + indices_ub_index) * self.update_data_num],
                                        0, 1, 1, 0, 0)
            self.tik_instance.vec_muls(self.updates_data_each_block, self.zero_ub, self.updates_tile_ub, 0, 1, 8, 8)
            self.tik_instance.vec_add(self.update_data_num, self.zero_ub, self.zero_ub, self.updates_tile_ub, 1, 8, 8,
                                      8)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(self.out_gm[self.var_read_index * self.update_data_num], self.zero_ub, 0, 1, 1,
                                        0, 0)
            self.tik_instance.set_atomic_add(0)

    def traversing_updates_single_core_and_ub_not_enough_atomic(self, indices_in_index, indice_num):
        """
        updateDataNum isn't 32B aligned and less than 1 block, ub can't store all updatesNum

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            self.tik_instance.data_move(self.updates_tile_ub,
                                        self.updates_gm[(indices_in_index + indices_ub_index) * self.update_data_num],
                                        0, 1, 1, 0, 0)
            self.tik_instance.vec_muls(self.updates_data_each_block, self.zero_ub, self.updates_tile_ub, 0, 1, 8, 8)
            self.tik_instance.vec_add(self.update_data_num, self.zero_ub, self.zero_ub, self.updates_tile_ub, 1, 8, 8,
                                      8)
            self.tik_instance.set_atomic_add(1)
            self.tik_instance.data_move(self.out_gm[self.var_read_index * self.update_data_num], self.zero_ub, 0, 1, 1,
                                        0, 0)
            self.tik_instance.set_atomic_add(0)

    def traversing_32b_aligned_ub_store_all_var_and_update(self, indices_in_index, indice_num):
        """
        32b aligned ub can store all var and update

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        updates_burst_len = self.updates_num // self.updates_data_each_block
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, updates_burst_len, 0, 0)

        with self.tik_instance.if_scope(self.core_loop_index < self.core_num - 1):
            self.each_core_max_indice.set_as((self.core_loop_index + 1) * self.indice_step)
            self.var_burst_len.set_as(self.var_each_core_burst_len)
        with self.tik_instance.else_scope():
            self.each_core_max_indice.set_as(self.max_indice)
            self.var_burst_len.set_as(self.var_last_core_burst_len)
        self.updates_the_var_mode6(indices_in_index, indice_num, self.each_core_max_indice, self.var_burst_len)

    def updates_the_var_mode6(self, indices_in_index, indice_num, max_indice, burst_len):
        """
        32b aligned ub can store all var and update

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        max_indice: int32
            max index
        burst_len: int32
            move data block
        repeat: int32
            repeat times of vector instruct
        mask: int32
            the mask of vector instruct
        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.var_ub, self.out_gm[self.core_loop_index * self.var_each_core_data], 0, 1,
                                    burst_len, 0, 0)
        self.last_var_num.set_as(self.update_data_num)
        self.var_offset.set_as(0)

        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope(max_indice > self.var_read_index):
                    with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat):
                        with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat * 255):
                            self.tik_instance.vec_add(
                                self.data_num_one_repeat,
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num],
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num],
                                self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num], 255, 8,
                                8, 8)
                            self.var_offset.set_as(255 * self.data_num_one_repeat)
                            self.last_var_num.set_as(self.update_data_num - 255 * self.data_num_one_repeat)
                        self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
                        self.mask.set_as(self.last_var_num % self.data_num_one_repeat)
                        self.tik_instance.vec_add(
                            self.data_num_one_repeat,
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num + self.var_offset],
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num + self.var_offset],
                            self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num +
                                            self.var_offset], self.repeat, 8, 8, 8)
                        with self.tik_instance.if_scope(self.mask != 0):
                            self.tik_instance.vec_add(
                                self.mask, self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                                       self.update_data_num + self.data_num_one_repeat * self.repeat +
                                                       self.var_offset],
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num + self.data_num_one_repeat * self.repeat +
                                            self.var_offset],
                                self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num +
                                                self.data_num_one_repeat * self.repeat + self.var_offset], 1, 8, 8, 8)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_add(
                            self.update_data_num,
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num],
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num],
                            self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num], 1, 8, 8, 8)
        self.tik_instance.data_move(self.out_gm[self.core_loop_index * self.var_each_core_data], self.var_ub, 0, 1,
                                    burst_len, 0, 0)

    def traversing_32b_aligned_ub_store_all_var(self, indices_in_index, indice_num):
        """
        32b aligned ub can store all var

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.core_loop_index < self.core_num - 1):
            self.each_core_max_indice.set_as((self.core_loop_index + 1) * self.indice_step)
            self.var_burst_len.set_as(self.var_each_core_burst_len)
        with self.tik_instance.else_scope():
            self.each_core_max_indice.set_as(self.max_indice)
            self.var_burst_len.set_as(self.var_last_core_burst_len)
        self.updates_the_var_mode7(indices_in_index, indice_num, self.each_core_max_indice, self.var_burst_len)

    def updates_the_var_mode7(self, indices_in_index, indice_num, max_indice, burst_len):
        """
        32b aligned ub can store all var

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        max_indice: int32
            max index
        burst_len: int32
            move data block
        repeat: int32
            repeat times of vector instruct
        mask: int32
            the mask of vector instruct
        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.var_ub, self.out_gm[self.core_loop_index * self.var_each_core_data], 0, 1,
                                    burst_len, 0, 0)
        self.last_var_num.set_as(self.update_data_num)
        self.var_offset.set_as(0)

        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope(max_indice > self.var_read_index):
                    self.tik_instance.data_move(
                        self.updates_ub, self.updates_gm[(indices_in_index + indices_ub_index) * self.update_data_num],
                        0, 1, self.update_data_num // self.updates_data_each_block, 0, 0)
                    with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat):
                        with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat * 255):
                            self.tik_instance.vec_add(
                                self.data_num_one_repeat,
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num],
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num], self.updates_ub, 255, 8, 8, 8)
                            self.var_offset.set_as(255 * self.data_num_one_repeat)
                            self.last_var_num.set_as(self.update_data_num - 255 * self.data_num_one_repeat)
                        self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
                        self.mask.set_as(self.last_var_num % self.data_num_one_repeat)
                        self.tik_instance.vec_add(
                            self.data_num_one_repeat,
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num + self.var_offset],
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num + self.var_offset], self.updates_ub[self.var_offset],
                            self.repeat, 8, 8, 8)
                        with self.tik_instance.if_scope(self.mask != 0):
                            self.tik_instance.vec_add(
                                self.mask, self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                                       self.update_data_num + self.data_num_one_repeat * self.repeat +
                                                       self.var_offset],
                                self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                            self.update_data_num + self.data_num_one_repeat * self.repeat +
                                            self.var_offset],
                                self.updates_ub[self.data_num_one_repeat * self.repeat + self.var_offset], 1, 8, 8, 8)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_add(
                            self.update_data_num,
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num],
                            self.var_ub[(self.var_read_index - self.indice_step * self.core_loop_index) *
                                        self.update_data_num], self.updates_ub, 1, 8, 8, 8)
        self.tik_instance.data_move(self.out_gm[self.core_loop_index * self.var_each_core_data], self.var_ub, 0, 1,
                                    burst_len, 0, 0)

    def traversing_32b_aligned_ub_store_all_update(self, indices_in_index, indice_num):
        """
        32b aligned ub can store all update

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        update_burst_len = self.updates_num // self.updates_data_each_block
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, update_burst_len, 0, 0)
        updates_burst_len = self.update_data_num // self.updates_data_each_block
        self.last_var_num.set_as(self.update_data_num)
        self.var_offset.set_as(0)

        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.tik_instance.data_move(self.var_ub, self.out_gm[self.var_read_index * self.update_data_num], 0,
                                                1, updates_burst_len, 0, 0)
                    with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat):
                        with self.tik_instance.if_scope(self.update_data_num >= self.data_num_one_repeat * 255):
                            self.tik_instance.vec_add(
                                self.data_num_one_repeat, self.var_ub, self.var_ub,
                                self.updates_ub[(indices_ub_index + indices_in_index) * self.update_data_num], 255, 8,
                                8, 8)
                            self.var_offset.set_as(255 * self.data_num_one_repeat)
                            self.last_var_num.set_as(self.update_data_num - 255 * self.data_num_one_repeat)
                        self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
                        self.mask.set_as(self.last_var_num % self.data_num_one_repeat)
                        self.tik_instance.vec_add(
                            self.data_num_one_repeat, self.var_ub[self.var_offset], self.var_ub[self.var_offset],
                            self.updates_ub[(indices_ub_index + indices_in_index) * self.update_data_num +
                                            self.var_offset], self.repeat, 8, 8, 8)
                        with self.tik_instance.if_scope(self.mask != 0):
                            self.tik_instance.vec_add(
                                self.mask, self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                self.updates_ub[(indices_ub_index + indices_in_index) * self.update_data_num +
                                                self.data_num_one_repeat * self.repeat + self.var_offset], 1, 8, 8, 8)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vec_add(
                            self.update_data_num, self.var_ub, self.var_ub,
                            self.updates_ub[(indices_ub_index + indices_in_index) * self.update_data_num], 1, 8, 8, 8)
                    self.tik_instance.data_move(self.out_gm[self.var_read_index * self.update_data_num], self.var_ub, 0,
                                                1, updates_burst_len, 0, 0)

    def circulate_indices_not_atomic(self, indices_in_index, indice_num, mode):
        """
        Circulate the index in the indices

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            with self.tik_instance.if_scope(self.core_loop_index * self.indice_step <= self.var_read_index):
                with self.tik_instance.if_scope((self.core_loop_index + 1) * self.indice_step > self.var_read_index):
                    self.var_read_index.set_as(self.var_read_index * self.update_data_num)
                    update_in_index = (indices_in_index + indices_ub_index) * self.update_data_num
                    if mode == 9:
                        self.traversing_var_mode9(update_in_index)
                    if mode == 14:
                        self.traversing_var_mode14(update_in_index)
                    if mode == 15:
                        self.traversing_var_mode15(update_in_index)

    def traversing_var_mode9(self, update_in_index):
        """
        Traversing the index in the updates of branch 9

        Parameters
        ----------
        update_in_index: int32
            Updates index on GM
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.var_loop_num) as var_loop_index:
            self.move_var(self.var_read_index + var_loop_index * self.var_ub_num, self.var_ub_num,
                          update_in_index + var_loop_index * self.var_ub_num, 9)

        with self.tik_instance.if_scope(self.var_last_num > 0):
            self.move_var(self.var_read_index + self.var_loop_num * self.var_ub_num, self.var_last_num,
                          update_in_index + self.var_loop_num * self.var_ub_num, 9)

    def traversing_var_mode14(self, update_in_index):
        """
        Traversing the index in the updates of branch 14

        Parameters
        ----------
        update_in_index: int32
            Updates index on GM
        Returns
        -------
        None
        """
        var_last_num_once = self.var_last_num // self.updates_data_each_block * self.updates_data_each_block
        self.move_var(self.var_read_index, var_last_num_once, update_in_index, 14)
        var_last_num_twice = self.var_last_num - var_last_num_once
        self.move_var_tail(self.var_read_index + self.var_last_num - self.updates_data_each_block, var_last_num_twice,
                           update_in_index + self.var_last_num - self.updates_data_each_block,
                           self.updates_data_each_block, 14)
        self.tik_instance.data_move(self.out_gm[self.var_read_index], self.var_ub, 0, 1,
                                    var_last_num_once // self.updates_data_each_block, 0, 0)

    def traversing_var_mode15(self, update_in_index):
        """
        Traversing the index in the updates of branch 15

        Parameters
        ----------
        update_in_index: int32
            Updates index on GM
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.var_loop_num) as var_loop_index:
            self.move_var(self.var_read_index + var_loop_index * self.var_ub_num, self.var_ub_num,
                          update_in_index + var_loop_index * self.var_ub_num, 15)

        self.var_last_num_off.set_as(
            (self.var_last_num // self.updates_data_each_block + 1) * self.updates_data_each_block)
        self.move_var_tail(self.var_read_index + self.update_data_num - self.var_last_num_off, self.var_last_num,
                           update_in_index + self.update_data_num - self.var_last_num_off, self.var_last_num_off, 15)

    def move_var(self, var_in_index, var_num, update_in_index, mode):
        """
        Traversing the var

        Parameters
        ----------
        var_in_index: int32
            Var index on UB
        var_num: int32
            Var num move
        update_in_index: int32
            update index on UB
        Returns
        -------
        None
        """
        var_burst_len = var_num // self.updates_data_each_block
        self.tik_instance.data_move(self.var_ub, self.out_gm[var_in_index], 0, 1, var_burst_len, 0, 0)
        self.tik_instance.data_move(self.updates_ub, self.updates_gm[update_in_index], 0, 1, var_burst_len, 0, 0)
        self.last_var_num.set_as(var_num)
        self.mask.set_as(var_num % self.data_num_one_repeat)
        self.var_offset.set_as(0)

        with self.tik_instance.if_scope(var_num >= self.data_num_one_repeat):
            with self.tik_instance.if_scope(var_num >= self.data_num_one_repeat * 255):
                self.tik_instance.vec_add(self.data_num_one_repeat, self.var_ub, self.var_ub, self.updates_ub, 255, 8,
                                          8, 8)
                self.var_offset.set_as(255 * self.data_num_one_repeat)
                self.last_var_num.set_as(var_num - 255 * self.data_num_one_repeat)
            self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
            self.mask.set_as(self.last_var_num % self.data_num_one_repeat)
            self.tik_instance.vec_add(self.data_num_one_repeat, self.var_ub[self.var_offset],
                                      self.var_ub[self.var_offset], self.updates_ub[self.var_offset], self.repeat, 8, 8,
                                      8)
            with self.tik_instance.if_scope(self.mask != 0):
                self.tik_instance.vec_add(self.mask,
                                          self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                          self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                          self.updates_ub[self.data_num_one_repeat * self.repeat + self.var_offset], 1,
                                          8, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vec_add(self.mask, self.var_ub, self.var_ub, self.updates_ub, 1, 8, 8, 8)

        if mode in (9, 15):
            self.tik_instance.data_move(self.out_gm[var_in_index], self.var_ub, 0, 1, var_burst_len, 0, 0)

    def move_var_tail(self, var_in_index, var_num, update_in_index, var_forward_offset, mode):
        """
        Traversing the var tail

        Parameters
        ----------
        var_in_index: int32
            Var index on UB
        var_num: int32
            Var num move
        update_in_index: int32
            update index on UB
        var_forward_offset: int32
            32 aligned of var_num
        Returns
        -------
        None
        """
        if mode == 14:
            self.tik_instance.data_move(self.var_tile_ub, self.out_gm[var_in_index], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.updates_tile_ub, self.updates_gm[update_in_index], 0, 1, 1, 0, 0)
            self.tik_instance.vec_add(self.updates_data_each_block, self.var_tile_ub, self.var_tile_ub,
                                      self.updates_tile_ub, 1, 8, 8, 8)
            self.tik_instance.data_move(self.out_gm[var_in_index], self.var_tile_ub, 0, 1, 1, 0, 0)

        if mode == 15:
            self.var_burst_len.set_as((var_num // self.updates_data_each_block) + 1)
            self.tik_instance.data_move(self.var_ub, self.out_gm[var_in_index], 0, 1, self.var_burst_len, 0, 0)
            self.tik_instance.data_move(self.updates_ub, self.updates_gm[update_in_index], 0, 1, self.var_burst_len, 0,
                                        0)
            self.last_var_num.set_as(var_forward_offset)
            self.var_offset.set_as(0)

            with self.tik_instance.if_scope(var_forward_offset >= self.data_num_one_repeat):
                with self.tik_instance.if_scope(var_forward_offset >= self.data_num_one_repeat * 255):
                    self.tik_instance.vector_dup(var_forward_offset - var_num, self.updates_ub, 0, 1, 1, 8)
                    self.tik_instance.vec_add(self.data_num_one_repeat, self.var_ub, self.var_ub, self.updates_ub, 255,
                                              8, 8, 8)
                    self.var_offset.set_as(255 * self.data_num_one_repeat)
                    self.last_var_num.set_as(var_forward_offset - 255 * self.data_num_one_repeat)
                self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
                self.mask.set_as(self.last_var_num - self.repeat * self.data_num_one_repeat)
                self.tik_instance.vector_dup(var_forward_offset - var_num, self.updates_ub, 0, 1, 1, 8)
                self.tik_instance.vec_add(self.data_num_one_repeat, self.var_ub[self.var_offset],
                                          self.var_ub[self.var_offset], self.updates_ub[self.var_offset], self.repeat,
                                          8, 8, 8)
                with self.tik_instance.if_scope(self.mask != 0):
                    self.tik_instance.vec_add(self.mask,
                                              self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                              self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                              self.updates_ub[self.data_num_one_repeat * self.repeat + self.var_offset],
                                              1, 8, 8, 8)
            with self.tik_instance.else_scope():
                self.tik_instance.vector_dup(var_forward_offset - var_num, self.updates_ub, 0, 1, 1, 8)
                self.tik_instance.vec_add(var_forward_offset, self.var_ub, self.var_ub, self.updates_ub, 1, 8, 8, 8)
            self.tik_instance.data_move(self.out_gm[var_in_index], self.var_ub, 0, 1, self.var_burst_len, 0, 0)

    def traversing_less_than_one_block_single_core_var_and_update(self, indices_in_index, indice_num):
        """
        updateDataNum is less than 1 block, ub can store all var and update

        Parameters
        ----------
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.var_num % self.updates_data_each_block == 0):
            self.var_burst_len.set_as(self.var_num // self.updates_data_each_block)
        with self.tik_instance.else_scope():
            self.var_burst_len.set_as((self.var_num // self.updates_data_each_block) + 1)
        self.tik_instance.data_move(self.var_ub, self.out_gm, 0, 1, self.var_burst_len, 0, 0)

        with self.tik_instance.if_scope(self.updates_num % self.updates_data_each_block == 0):
            self.updates_burst_len.set_as(self.updates_num // self.updates_data_each_block)
        with self.tik_instance.else_scope():
            self.updates_burst_len.set_as((self.updates_num // self.updates_data_each_block) + 1)
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, self.updates_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            self.var_read_index.set_as(self.var_read_index * self.update_data_num)
            with self.tik_instance.for_range(0, self.update_data_num) as ub_index:
                self.var_value.set_as(self.var_ub[self.var_read_index + ub_index])
                self.var_tile_ub[ub_index].set_as(self.var_value)
                self.update_value.set_as(self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num +
                                                         ub_index])
                self.updates_tile_ub[ub_index].set_as(self.update_value)
            self.tik_instance.vec_add(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.updates_tile_ub, 1,
                                      8, 8, 8)
            with self.tik_instance.for_range(0, self.update_data_num) as ub_index:
                self.var_value.set_as(self.var_tile_ub[ub_index])
                self.var_ub[self.var_read_index + ub_index].set_as(self.var_value)

        self.tik_instance.data_move(self.out_gm, self.var_ub, 0, 1, self.var_burst_len, 0, 0)

    def traversing_less_than_one_block_single_core_var(self, indices_in_index, indice_num):
        """
        updateDataNum is less than 1 block, ub can store all var

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.var_num % self.updates_data_each_block == 0):
            self.var_burst_len.set_as(self.var_num // self.updates_data_each_block)
        with self.tik_instance.else_scope():
            self.var_burst_len.set_as((self.var_num // self.updates_data_each_block) + 1)
        self.tik_instance.data_move(self.var_ub, self.out_gm, 0, 1, self.var_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            self.var_read_index.set_as(self.var_read_index * self.update_data_num)
            self.tik_instance.data_move(self.updates_tile_ub,
                                        self.updates_gm[(indices_in_index + indices_ub_index) * self.update_data_num],
                                        0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.update_data_num) as ub_index:
                self.var_value.set_as(self.var_ub[self.var_read_index + ub_index])
                self.var_tile_ub[ub_index].set_as(self.var_value)
            self.tik_instance.vec_add(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.updates_tile_ub, 1,
                                      8, 8, 8)
            with self.tik_instance.for_range(0, self.update_data_num) as ub_index:
                self.var_value.set_as(self.var_tile_ub[ub_index])
                self.var_ub[self.var_read_index + ub_index].set_as(self.var_value)

        self.tik_instance.data_move(self.out_gm, self.var_ub, 0, 1, self.var_burst_len, 0, 0)

    def traversing_less_than_one_block_single_core_update(self, indices_in_index, indice_num):
        """
        updateDataNum is less than 1 block, ub can store all update

        Parameters
        ----------
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.updates_num % self.updates_data_each_block == 0):
            self.updates_burst_len.set_as(self.updates_num // self.updates_data_each_block)
        with self.tik_instance.else_scope():
            self.updates_burst_len.set_as((self.updates_num // self.updates_data_each_block) + 1)
        self.tik_instance.data_move(self.updates_ub, self.updates_gm, 0, 1, self.updates_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            self.var_read_index.set_as(self.var_read_index * self.update_data_num)
            self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index], 0, 1, 1, 0, 0)
            with self.tik_instance.for_range(0, self.updates_data_each_block) as ub_index:
                self.update_value.set_as(self.updates_ub[(indices_in_index + indices_ub_index) * self.update_data_num +
                                                         ub_index])
                self.updates_tile_ub[ub_index].set_as(self.update_value)
            self.tik_instance.vec_add(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.updates_tile_ub, 1,
                                      8, 8, 8)
            self.tik_instance.data_move(self.out_gm[self.var_read_index], self.var_tile_ub, 0, 1, 1, 0, 0)

    def traversing_less_than_one_block_ub_not_enough(self, indices_in_index, indice_num):
        """
        updateDataNum is less than 1 block, ub can't store all var and update

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, indice_num // self.indices_last_dim) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            self.var_read_index.set_as(self.var_read_index * self.update_data_num)
            self.tik_instance.data_move(self.var_tile_ub, self.out_gm[self.var_read_index], 0, 1, 1, 0, 0)
            self.tik_instance.data_move(self.updates_tile_ub,
                                        self.updates_gm[(indices_in_index + indices_ub_index) * self.update_data_num],
                                        0, 1, 1, 0, 0)
            self.tik_instance.vec_add(self.update_data_num, self.var_tile_ub, self.var_tile_ub, self.updates_tile_ub, 1,
                                      8, 8, 8)
            self.tik_instance.data_move(self.out_gm[self.var_read_index], self.var_tile_ub, 0, 1, 1, 0, 0)

    def traversing_var_single_core(self):
        """
        Traversing the index in the var

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.var_each_core_set_zero_loop_num > 0):
            with self.tik_instance.for_range(0, self.var_each_core_set_zero_loop_num) as var_loop_index:
                self.out_gm_set_zero(var_loop_index * self.var_ub_num, self.var_ub_num)

        with self.tik_instance.if_scope(self.var_each_core_set_zero_last_num > 0):
            self.out_gm_set_zero(self.var_each_core_set_zero_loop_num * self.var_ub_num,
                                 self.var_each_core_set_zero_last_num)

    def traversing_var_mul_core(self):
        """
        Traversing the index in the var

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.core_loop_index < self.core_num - 1):
            with self.tik_instance.if_scope(self.var_each_core_set_zero_loop_num > 0):
                with self.tik_instance.for_range(0, self.var_each_core_set_zero_loop_num) as var_loop_index:
                    self.out_gm_set_zero(
                        self.core_loop_index * self.var_each_core_data + var_loop_index * self.var_ub_num,
                        self.var_ub_num)
            with self.tik_instance.if_scope(self.var_each_core_set_zero_last_num > 0):
                self.out_gm_set_zero(
                    self.core_loop_index * self.var_each_core_data +
                    self.var_each_core_set_zero_loop_num * self.var_ub_num, self.var_each_core_set_zero_last_num)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.var_last_core_set_zero_loop_num > 0):
                with self.tik_instance.for_range(0, self.var_last_core_set_zero_loop_num) as var_loop_index:
                    self.out_gm_set_zero(
                        self.core_loop_index * self.var_each_core_data + var_loop_index * self.var_ub_num,
                        self.var_ub_num)
            with self.tik_instance.if_scope(self.var_last_core_set_zero_last_num > 0):
                self.out_gm_set_zero(
                    self.core_loop_index * self.var_each_core_data +
                    self.var_last_core_set_zero_loop_num * self.var_ub_num, self.var_last_core_set_zero_last_num)

    def out_gm_vector_dup_zero(self, var_num):
        """
        calculate out_gm * 0

        Parameters
        ----------
        var_num: int32
            the number of updates
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(var_num >= self.data_num_one_repeat):
            with self.tik_instance.if_scope(var_num >= self.data_num_one_repeat * 255):
                self.tik_instance.vector_dup(self.data_num_one_repeat, self.var_ub, 0, 255, 8, 8)
                self.var_offset.set_as(255 * self.data_num_one_repeat)
                self.last_var_num.set_as(var_num - 255 * self.data_num_one_repeat)
            self.repeat.set_as(self.last_var_num // self.data_num_one_repeat)
            self.mask.set_as(self.last_var_num % self.data_num_one_repeat)
            self.tik_instance.vector_dup(self.data_num_one_repeat, self.var_ub[self.var_offset], 0, self.repeat, 8, 8)
            with self.tik_instance.if_scope(self.mask != 0):
                self.tik_instance.vector_dup(self.mask,
                                             self.var_ub[self.data_num_one_repeat * self.repeat + self.var_offset], 0,
                                             1, 8, 8)
        with self.tik_instance.else_scope():
            self.tik_instance.vector_dup(self.last_var_num, self.var_ub, 0, 1, 8, 8)

    def out_gm_set_zero(self, var_in_index, var_num):
        """
        calculate out_gm * 0

        Parameters
        ----------
        var_in_index: int32
            var index on GM
        var_num: int32
            the number of updates
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(var_num % self.updates_data_each_block == 0):
            self.var_burst_len.set_as(var_num // self.updates_data_each_block)
        with self.tik_instance.else_scope():
            self.var_burst_len.set_as((var_num // self.updates_data_each_block) + 1)

        self.tik_instance.data_move(self.var_ub, self.out_gm[var_in_index], 0, 1, self.var_burst_len, 0, 0)
        self.last_var_num.set_as(var_num)
        self.var_offset.set_as(0)
        self.out_gm_vector_dup_zero(var_num)
        self.tik_instance.data_move(self.out_gm[var_in_index], self.var_ub, 0, 1, self.var_burst_len, 0, 0)

    def traversing_indices(self, mode):
        """
        Traversing the index in the indices

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        max_indices_ub_num = self.indices_ub_num // self.indices_last_dim * self.indices_last_dim

        with self.tik_instance.for_range(0, self.indices_loop_num) as indices_loop_index:
            self.move_indices(indices_loop_index * max_indices_ub_num, max_indices_ub_num, mode)

        with self.tik_instance.if_scope(self.indices_last_num > 0):
            self.move_indices(self.indices_loop_num * max_indices_ub_num, self.indices_last_num, mode)

    def traversing_indices_deepfm(self, mode):
        """
        Traversing the index in the indices for deepfm

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.core_loop_index < self.core_num - 1):
            with self.tik_instance.for_range(0, self.each_core_indices_loop_num) as indices_loop_index:
                self.deepfm_perf(
                    self.core_loop_index * self.indices_each_core_data + indices_loop_index * self.indices_ub_num,
                    self.indices_ub_num, mode)
            with self.tik_instance.if_scope(self.each_core_indices_last_num > 0):
                self.deepfm_perf(
                    self.core_loop_index * self.indices_each_core_data +
                    self.each_core_indices_loop_num * self.indices_ub_num, self.each_core_indices_last_num, mode)
        with self.tik_instance.if_scope(self.core_loop_index == self.core_num - 1):
            with self.tik_instance.for_range(0, self.last_core_indices_loop_num) as indices_loop_index:
                self.deepfm_perf(
                    self.core_loop_index * self.indices_each_core_data + indices_loop_index * self.indices_ub_num,
                    self.indices_ub_num, mode)
            with self.tik_instance.if_scope(self.last_core_indices_last_num > 0):
                self.deepfm_perf(
                    self.core_loop_index * self.indices_each_core_data +
                    self.last_core_indices_loop_num * self.indices_ub_num, self.last_core_indices_last_num, mode)

    def deepfm_perf(self, indices_in_index, indice_num, mode):
        """
        Move indices, deepfm branch

        Parameters
        ----------
        indices_in_index: int32
            Indices index on GM
        indice_num: int32
            the number of indexes in the indices on UB
        Returns
        -------
        None
        """
        indices_burst_len = ceil_div(indice_num, self.indices_data_each_block)
        self.tik_instance.data_move(self.indices_ub, self.indices_gm[indices_in_index], 0, 1, indices_burst_len, 0, 0)

        with self.tik_instance.for_range(0, indice_num) as indices_ub_index:
            self.var_read_index.set_as(self.indices_ub[indices_ub_index])
            self.var_read_index.set_as(self.var_read_index * self.update_data_num)
            updates_read_index = (indices_in_index + indices_ub_index) * self.update_data_num
            if mode == 16:
                self.tik_instance.data_move(self.updates_tile_ub, self.updates_gm[updates_read_index], 0, 1, 1, 0, 0)
                self.tik_instance.vec_muls(self.updates_data_each_block, self.zero_ub, self.updates_tile_ub, 0, 1, 8, 8)
                self.tik_instance.vec_add(self.update_data_num, self.zero_ub, self.zero_ub, self.updates_tile_ub, 1, 8,
                                          8, 8)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.out_gm[self.var_read_index], self.zero_ub, 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)
            if mode == 17:
                self.tik_instance.data_move(self.updates_ub, self.updates_gm[updates_read_index], 0, 1,
                                            self.update_data_num // self.updates_data_each_block, 0, 0)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.out_gm[self.var_read_index], self.updates_ub, 0, 1,
                                            self.update_data_num // self.updates_data_each_block, 0, 0)
                self.tik_instance.set_atomic_add(0)

    def scatter_nd_compute_tiling(self):
        """
        Main process of scatter_nd

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as core_index:
            self.tiling_ub = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,),
                                                      name="tiling_ub",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 9, 0, 0)
            self.tiling_args()

            if tbe_platform.api_check_support("tik.set_atomic_add") and self.updates_dtype == "float32":
                with self.tik_instance.if_scope(self.tiling_mode == 1):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.core_loop_index.set_as(core_index)
                        self.traversing_indices(1)
                with self.tik_instance.if_scope(self.tiling_mode == 2):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.core_loop_index.set_as(core_index)
                        self.traversing_indices(2)
                with self.tik_instance.if_scope(self.tiling_mode == 3):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.traversing_indices(3)
                with self.tik_instance.if_scope(self.tiling_mode == 4):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.traversing_indices(4)
                with self.tik_instance.if_scope(self.tiling_mode == 5):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.core_loop_index.set_as(core_index)
                        self.traversing_indices(5)
                with self.tik_instance.if_scope(self.tiling_mode == 16):
                    self.init_ub_tensor()
                    self.core_loop_index.set_as(core_index)
                    self.traversing_indices_deepfm(16)
                with self.tik_instance.if_scope(self.tiling_mode == 17):
                    self.init_ub_tensor()
                    self.core_loop_index.set_as(core_index)
                    self.traversing_indices_deepfm(17)
            else:
                with self.tik_instance.if_scope(self.tiling_mode == 6):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.core_loop_index.set_as(core_index)
                        self.traversing_indices(6)
                with self.tik_instance.if_scope(self.tiling_mode == 7):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.core_loop_index.set_as(core_index)
                        self.traversing_indices(7)
                with self.tik_instance.if_scope(self.tiling_mode == 8):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.core_loop_index.set_as(core_index)
                        self.traversing_indices(8)
                with self.tik_instance.if_scope(self.tiling_mode == 9):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.core_loop_index.set_as(core_index)
                        self.traversing_indices(9)
                with self.tik_instance.if_scope(self.tiling_mode == 10):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.traversing_indices(10)
                with self.tik_instance.if_scope(self.tiling_mode == 11):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.traversing_indices(11)
                with self.tik_instance.if_scope(self.tiling_mode == 12):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.traversing_indices(12)
                with self.tik_instance.if_scope(self.tiling_mode == 13):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.traversing_indices(13)
                with self.tik_instance.if_scope(self.tiling_mode == 14):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.core_loop_index.set_as(core_index)
                        self.traversing_indices(14)
                with self.tik_instance.if_scope(self.tiling_mode == 15):
                    with self.tik_instance.if_scope(core_index < self.core_num):
                        self.init_ub_tensor()
                        self.core_loop_index.set_as(core_index)
                        self.traversing_indices(15)

    def scatter_nd_operator(self):
        """
        scatter_nd operation

        Parameters
        ----------
        None

        Returns:
        ----------
        compile info
        """
        self.scatter_nd_compute_tiling()
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "core_num": self.ai_core_num,
                "updates_size": self.updates_dtype_bytes_size,
                "indices_size": self.indices_dtype_bytes_size,
                "support_atomic": self.support_atomic
            })
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.indices_gm, self.updates_gm, self.shape_gm),
                                   outputs=(self.out_gm),
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)


# pylint: disable=unused-argument,invalid-name
@register_operator("ScatterNd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def scatter_nd(indices, x, shape, y, kernel_name="ScatterNd"):
    """
    scatter_add interface

    Parameters
    ----------
    indices_dict: input indices shape, dtype and range
    x_dict: input x shape, dtype and range
    shape_dict: input shape shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of scatter_add op

    Returns
    -------
    compile info
    """
    obj = ScatterNd(indices, x, shape, y, kernel_name)
    return obj.scatter_nd_operator()
