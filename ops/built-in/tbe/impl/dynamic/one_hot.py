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
one_hot
"""
# 'pylint: disable=too-many-lines
from impl.util.platform_adapter import tbe_platform as cce
from impl.util.platform_adapter import tik
from impl.util import util_common
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # 16k UB buffer is a reserved space
    RESERVE_SIZE = 16 * 1024
    MAX_INT32 = 2 ** 31 - 1
    SCALAR_TENSOR_SIZE = 32
    TILING_ARG_NUM = 64
    cal_num = 64
    TILING_MODE_1 = 1
    TILING_MODE_2 = 2
    TILING_MODE_3 = 3
    TILING_MODE_4 = 4
    TILING_MODE_5 = 5
    TILING_MODE_6 = 6
    TILING_MODE_7 = 7
    TILING_MODE_8 = 8
    TILING_MODE_9 = 9
    TILING_MODE_10 = 10
    TILING_MODE_11 = 11
    TILING_MODE_12 = 12
    TILING_MODE_13 = 13
    OFF_VALUE_TENSOR_PART = 2
    TOTAL_PART = 3


# 'pylint: disable=too-many-public-methods,too-many-instance-attributes,too-many-arguments
# 'pylint: disable=unused-argument,too-many-statements,too-many-locals,invalid-name
class OneHot:
    """
    The class of OneHot op
    """

    # 'pylint: disable =too-many-arguments,too-many-statements
    def __init__(
            self,
            x,
            depth,
            on_value,
            off_value,
            axis,
            y,
            kernel_name='one_hot'):
        """
        constructor of OneHot

        Parameters
        ----------
        x: dict
            shape and dtype of input indices tensor
        depth: dict
            the int32 scalar which judge the depth of add dim
        on_value: dict
            the value which set_as by the input tensor x
        off_value: dict
            the value which used to fill the off_value_tensor at first
        axis:int
            the attr judged which dim will be add
        y:dict
            dict with keys(range and dtype) of output
        kernel_name: str
            kernel name, default value is "one_hot"

        Returns
        -------
        None
        """
        self.dtype_x = x.get('dtype')
        self.dtype_depth = depth.get('dtype')
        self.dtype_on_value = on_value.get('dtype')
        self.dtype_off_value = off_value.get('dtype')
        self.axis = axis
        self.kernel_name = kernel_name
        self.tiling_dtype = 'int32'

        block_bite_size = 32
        self.max_repeat_time = 255
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.ub_size_bytes = cce.get_soc_spec(cce.UB_SIZE)
        self.dtype_bytes_size_x = cce.get_bit_len(self.dtype_x) // 8
        self.x_each_block = block_bite_size // self.dtype_bytes_size_x
        self.dtype_bytes_size_depth = cce.get_bit_len(
            self.dtype_depth) // 8
        self.depth_each_block = block_bite_size // self.dtype_bytes_size_depth
        self.dtype_bytes_size_on_value = cce.get_bit_len(
            self.dtype_on_value) // 8
        self.on_value_each_block = block_bite_size // self.dtype_bytes_size_on_value
        self.dtype_bytes_size_off_value = cce.get_bit_len(
            self.dtype_off_value) // 8
        self.off_value_each_block = block_bite_size // self.dtype_bytes_size_off_value
        self.vector_mask_max_x = 8 * self.x_each_block
        self.dump_mask_max_off_value = 8 * self.off_value_each_block
        self.dtype_bytes_size_tiling = cce.get_bit_len(
            self.tiling_dtype) // 8
        self.tiling_each_block = block_bite_size // self.dtype_bytes_size_tiling
        self.index_scalar = self.tik_instance.Scalar(
            self.dtype_x, name='index_scalar')
        self.tiling_gm = self.tik_instance.Tensor(
            self.tiling_dtype, (Constant.TILING_ARG_NUM,), name='tiling_gm', scope=tik.scope_gm)
        self.total_core_number = cce.get_soc_spec(cce.CORE_NUM)
        self.numel_shape_x = None
        self.x_gm = None
        self.off_value_ub = None
        self.last_core_index = None
        self.first_dim_x = None
        self.depth_ub = None
        self.on_value_ub = None
        self.on_value_gm = None
        self.x_ub = None
        self.off_value_gm = None
        self.numel_shape_off_value_tensor = None
        self.total_part = None
        self.core_number = None
        self.y_gm = None
        self.off_value = None
        self.max_numel_vec_dup_one_loop = None
        self.last_core_numel = None
        self.on_value = None
        self.last_dim_x = None
        self.depth_gm = None
        self.is_zero_off_value = None
        self.off_value_tensor_ub = None
        self.remain_mask = None
        self.offset_off_value_tensor = None
        self.depth = None
        self.not_last_core_index = None
        self.not_last_core_numel = None
        self.per_part_unused_ub = None
        self.off_value_tensor_part = None
        self.remain_repeat_time = None

    def get_tiling_args(self, tiling_ub):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from one_hot tiling

        Returns
        -------
        None
        """
        self.is_zero_off_value = self.tik_instance.Scalar(
            self.dtype_x, name='is_zero_off_value')
        self.core_number = self.tik_instance.Scalar(
            self.dtype_x, name='core_number')
        self.numel_shape_x = self.tik_instance.Scalar(
            self.dtype_x, name='shape_x')
        self.first_dim_x = self.tik_instance.Scalar(
            self.dtype_x, name='first_dim_x')
        self.last_dim_x = self.tik_instance.Scalar(
            self.dtype_x, name='last_dim_x')
        self.numel_shape_off_value_tensor = self.tik_instance.Scalar(
            self.dtype_x, name='numel_shape_off_value_tensor')
        self.not_last_core_index = self.tik_instance.Scalar(
            self.dtype_x, name='not_last_core_index')
        self.last_core_index = self.tik_instance.Scalar(
            self.dtype_x, name='last_core_index')
        self.not_last_core_numel = self.tik_instance.Scalar(
            self.dtype_x, name='not_last_core_numel')
        self.last_core_numel = self.tik_instance.Scalar(
            self.dtype_x, name='last_core_numel')

        self.is_zero_off_value.set_as(tiling_ub[0])
        self.not_last_core_numel.set_as(tiling_ub[1])
        self.core_number.set_as(tiling_ub[3])
        self.numel_shape_x.set_as(tiling_ub[4])
        self.first_dim_x.set_as(tiling_ub[5])
        self.last_dim_x.set_as(tiling_ub[6])
        self.numel_shape_off_value_tensor.set_as(tiling_ub[7])
        self.last_core_numel.set_as(tiling_ub[8])
        self.not_last_core_index.set_as(tiling_ub[9])
        self.last_core_index.set_as(tiling_ub[10])

    def gm_to_data(self):
        """
        GM size to the data of OneHot OP

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.x_gm = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT32,),
                                             name='x_gm', scope=tik.scope_gm)
        self.y_gm = self.tik_instance.Tensor(self.dtype_on_value, (Constant.MAX_INT32,),
                                             name='y_gm', scope=tik.scope_gm)
        self.on_value_gm = self.tik_instance.Tensor(
            self.dtype_on_value,
            (Constant.SCALAR_TENSOR_SIZE,
             ),
            name='on_value',
            scope=tik.scope_gm)
        self.off_value_gm = self.tik_instance.Tensor(
            self.dtype_off_value,
            (Constant.SCALAR_TENSOR_SIZE,
             ),
            name='off_value',
            scope=tik.scope_gm)
        self.depth_gm = self.tik_instance.Tensor(
            self.dtype_depth, (Constant.SCALAR_TENSOR_SIZE,), name='depth', scope=tik.scope_gm)

    def one_hot_compute_tiling(self):
        """
        Main process of one_hot

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.gm_to_data()
        with self.tik_instance.for_range(0, self.total_core_number, block_num=self.total_core_number) as block_id:
            tiling_ub = self.tik_instance.Tensor(
                self.tiling_dtype, (Constant.TILING_ARG_NUM,), name='tiling_ub', scope=tik.scope_ubuf)
            self.tik_instance.data_move(
                tiling_ub,
                self.tiling_gm,
                0,
                1,
                Constant.SCALAR_TENSOR_SIZE //
                self.tiling_each_block,
                0,
                0)
            self.ub_to_data()
            self.get_tiling_args(tiling_ub)
            with self.tik_instance.if_scope(block_id < self.core_number):
                self.data_move()
                mode_of_cal_with_axis = self.tik_instance.Scalar(
                    self.dtype_x, name='mode_of_cal_with_axis')
                mode_of_cal_with_axis.set_as(tiling_ub[2])
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_1):
                    self.one_hot_last_axis_first_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_2):
                    self.one_hot_last_axis_second_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_3):
                    self.one_hot_last_axis_third_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_4):
                    self.one_hot_last_axis_fourth_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_5):
                    self.one_hot_last_axis_fifth_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_6):
                    self.one_hot_first_axis_first_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_7):
                    self.one_hot_first_axis_second_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_8):
                    self.one_hot_first_axis_third_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_9):
                    self.one_hot_first_axis_fourth_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_10):
                    self.one_hot_middle_axis_first_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_11):
                    self.one_hot_middle_axis_second_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_12):
                    self.one_hot_middle_axis_third_mode(block_id)
                with self.tik_instance.if_scope(mode_of_cal_with_axis == Constant.TILING_MODE_13):
                    self.one_hot_middle_axis_fourth_mode(block_id)
        tbe_context.get_context().add_compile_info('vars', {'core_num': self.total_core_number, 'axis': self.axis})
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name, inputs=[
                self.x_gm, self.depth_gm, self.on_value_gm, self.off_value_gm], outputs=[
                self.y_gm], flowtable=[
                self.tiling_gm])
        return self.tik_instance

    def ub_to_data(self):
        """
        UB size to the data of OneHot OP

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.total_part = Constant.TOTAL_PART
        self.off_value_tensor_part = Constant.OFF_VALUE_TENSOR_PART
        self.per_part_unused_ub = (self.ub_size_bytes - Constant.RESERVE_SIZE) // self.dtype_bytes_size_x // \
            self.total_part // self.x_each_block * self.x_each_block
        self.x_ub = self.tik_instance.Tensor(
            self.dtype_x, (self.per_part_unused_ub,), name='x_ub', scope=tik.scope_ubuf)
        self.off_value_tensor_ub = self.tik_instance.Tensor(
            self.dtype_off_value,
            (self.per_part_unused_ub *
             self.off_value_tensor_part,
             ),
            name='off_value_tensor_ub',
            scope=tik.scope_ubuf)
        self.on_value_ub = self.tik_instance.Tensor(
            self.dtype_on_value,
            (Constant.SCALAR_TENSOR_SIZE,
             ),
            name='on_value_ub',
            scope=tik.scope_ubuf)
        self.off_value_ub = self.tik_instance.Tensor(
            self.dtype_off_value,
            (Constant.SCALAR_TENSOR_SIZE,
             ),
            name='off_value_ub',
            scope=tik.scope_ubuf)
        self.depth_ub = self.tik_instance.Tensor(
            self.dtype_depth, (Constant.SCALAR_TENSOR_SIZE,), name='depth_ub', scope=tik.scope_ubuf)
        self.on_value = self.tik_instance.Scalar(
            self.dtype_on_value, name='on_value_scalar')
        self.off_value = self.tik_instance.Scalar(
            self.dtype_off_value, name='off_value_scalar')
        self.depth = self.tik_instance.Scalar(self.dtype_depth, name='depth_scalar')

    def data_move(self):
        """
        move data of OneHot op from gm to ub

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.tik_instance.data_move(
            self.on_value_ub,
            self.on_value_gm,
            0,
            1,
            Constant.SCALAR_TENSOR_SIZE //
            self.on_value_each_block,
            0,
            0)
        self.on_value.set_as(self.on_value_ub[0])
        self.tik_instance.data_move(
            self.off_value_ub,
            self.off_value_gm,
            0,
            1,
            Constant.SCALAR_TENSOR_SIZE //
            self.off_value_each_block,
            0,
            0)

        self.off_value.set_as(self.off_value_ub[0])
        self.tik_instance.data_move(
            self.depth_ub,
            self.depth_gm,
            0,
            1,
            Constant.SCALAR_TENSOR_SIZE //
            self.depth_each_block,
            0,
            0)
        self.depth.set_as(self.depth_ub[0])

    def vec_dump_off_value_tensor_ub(self, off_value_ub_size):
        """
        the function which vec dump the space of off_value_tensor_ub

        Parameters
        ----------
        off_value_ub_size:
        the size of ub space which should be filled with off_value

        Returns
        -------
        None
        """
        self.max_numel_vec_dup_one_loop = self.max_repeat_time * self.dump_mask_max_off_value
        with self.tik_instance.for_range(0,
                                         off_value_ub_size // self.max_numel_vec_dup_one_loop) as loop:
            self.tik_instance.vec_dup(self.dump_mask_max_off_value,
                                      self.off_value_tensor_ub[loop * self.max_numel_vec_dup_one_loop],
                                      self.off_value, self.max_repeat_time, 8)
        self.remain_repeat_time = off_value_ub_size % self.max_numel_vec_dup_one_loop // self.dump_mask_max_off_value
        self.remain_mask = off_value_ub_size % self.max_numel_vec_dup_one_loop % self.dump_mask_max_off_value
        with self.tik_instance.if_scope(self.remain_repeat_time > 0):
            self.tik_instance.vec_dup(self.dump_mask_max_off_value,
                                      self.off_value_tensor_ub[off_value_ub_size //
                                                               self.max_numel_vec_dup_one_loop *
                                                               self.max_numel_vec_dup_one_loop],
                                      self.off_value, self.remain_repeat_time, 8)
        self.offset_off_value_tensor = off_value_ub_size // self.max_numel_vec_dup_one_loop * \
            self.max_numel_vec_dup_one_loop + off_value_ub_size % \
            self.max_numel_vec_dup_one_loop \
            // self.dump_mask_max_off_value * \
            self.dump_mask_max_off_value
        with self.tik_instance.if_scope(self.remain_mask > 0):
            self.tik_instance.vec_dup(self.remain_mask,
                                      self.off_value_tensor_ub[self.offset_off_value_tensor],
                                      self.off_value,
                                      1,
                                      8)

    def align_to_32_last_block(self, first_index, second_index, id_number):
        """
        align the last block of data move when the axis is -1 or middle

        Parameters
        ----------
        first_index:int
        the index of the origin begin of last block
        second_index:int
        the index of the offset of the y_gm
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(first_index % self.off_value_each_block > 0):
            block_ub = self.tik_instance.Tensor(
                self.dtype_off_value,
                (self.off_value_each_block,
                 ),
                name='block_ub',
                scope=tik.scope_ubuf)
            offset_begin = first_index // self.off_value_each_block * self.off_value_each_block \
                - (self.off_value_each_block - first_index % self.off_value_each_block)
            with self.tik_instance.for_range(offset_begin, first_index) as index:
                block_ub[index -
                         offset_begin].set_as(self.off_value_tensor_ub[index])
            self.tik_instance.data_move(
                self.y_gm[id_number * self.not_last_core_numel * self.depth + second_index + offset_begin],
                block_ub, 0, 1, 1, 0, 0)

    def align_to_32_last_block_first_dim(
            self, first_index, second_index, id_number):
        """
        align the last block of data move when the axis is 0

        Parameters
        ----------
        first_index:int
        the index of the origin begin of last block
        second_index:int
        the index of the offset of the y_gm
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(first_index % self.off_value_each_block > 0):
            block_ub = self.tik_instance.Tensor(
                self.dtype_off_value,
                (self.off_value_each_block,
                 ),
                name='block_ub',
                scope=tik.scope_ubuf)
            offset_begin = first_index // self.off_value_each_block * self.off_value_each_block \
                - (self.off_value_each_block - first_index % self.off_value_each_block)
            with self.tik_instance.for_range(offset_begin, first_index) as index:
                block_ub[index -
                         offset_begin].set_as(self.off_value_tensor_ub[index])
            self.tik_instance.data_move(
                self.y_gm[id_number * self.not_last_core_index * self.numel_shape_x + second_index + offset_begin],
                block_ub, 0, 1, 1, 0, 0)

    # last axis with ub enough for all
    def one_hot_last_axis_first_mode(self, id_number):
        """
        the first calculate mode when the axis is 0

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_numel * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_numel + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_numel + begin)
        move_numel = self.tik_instance.Scalar(dtype='int32', name='move_numel')
        move_numel.set_as(
            ((((end - begin) - 1) // self.x_each_block + 1) * self.x_each_block))
        self.tik_instance.data_move(self.x_ub,
                                    self.x_gm[id_number * self.not_last_core_numel],
                                    0,
                                    1,
                                    move_numel // self.x_each_block,
                                    0,
                                    0)
        off_value_ub_size = self.tik_instance.Scalar(
            dtype='int32', name='off_value_ub_size')
        with self.tik_instance.if_scope((move_numel * self.depth) <
                                        (self.per_part_unused_ub * self.off_value_tensor_part)):
            off_value_ub_size.set_as(move_numel * self.depth)
        with self.tik_instance.else_scope():
            off_value_ub_size.set_as(
                self.per_part_unused_ub *
                self.off_value_tensor_part)
        self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        with self.tik_instance.for_range(0, (end - begin)) as i:
            self.index_scalar.set_as(self.x_ub[i])
            with self.tik_instance.if_scope(tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                self.index_scalar.set_as(i * self.depth + self.index_scalar)
                self.off_value_tensor_ub[self.index_scalar].set_as(self.on_value)

        with self.tik_instance.if_scope((end - begin) * self.depth // self.off_value_each_block > 0):
            self.tik_instance.data_move(self.y_gm[id_number *
                                                  self.not_last_core_numel *
                                                  self.depth], self.off_value_tensor_ub, 0, 1, (end -
                                                                                                begin) *
                                        self.depth //
                                        self.off_value_each_block, 0, 0)
            self.align_to_32_last_block(
                (end - begin) * self.depth, 0, id_number)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.y_gm[id_number *
                                                  self.not_last_core_numel *
                                                  self.depth], self.off_value_tensor_ub, 0, 1, 1, 0, 0)

    # last axis with ub size is more than x and enough to off_value_tensor
    # some lines
    def one_hot_last_axis_second_mode(self, id_number):
        """
        the second calculate mode when the axis is 0

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_numel * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_numel + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_numel + begin)
        move_numel = self.tik_instance.Scalar(dtype='int32', name='move_numel')
        move_numel.set_as(
            ((((end - begin) - 1) // self.x_each_block + 1) * self.x_each_block))
        self.tik_instance.data_move(self.x_ub,
                                    self.x_gm[id_number * self.not_last_core_numel],
                                    0,
                                    1,
                                    move_numel // self.x_each_block,
                                    0,
                                    0)
        line_num = self.tik_instance.Scalar(dtype='int32', name='line_num')
        line_num.set_as(
            self.per_part_unused_ub *
            self.off_value_tensor_part //
            self.depth)
        off_value_ub_size = self.tik_instance.Scalar(
            dtype='int32', name='off_value_ub_size')
        off_value_ub_size.set_as(
            self.per_part_unused_ub *
            self.off_value_tensor_part)
        self.vec_dump_off_value_tensor_ub(off_value_ub_size)

        with self.tik_instance.for_range(0, (end - begin) // line_num) as i:
            with self.tik_instance.for_range(0, line_num) as j:
                self.index_scalar.set_as(self.x_ub[i * line_num + j])
                with self.tik_instance.if_scope(tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                    self.index_scalar.set_as(j * self.depth + self.index_scalar)
                    self.off_value_tensor_ub[self.index_scalar].set_as(self.on_value)
            self.tik_instance.data_move(self.y_gm[id_number *
                                                  self.not_last_core_numel *
                                                  self.depth +
                                                  i *
                                                  self.depth *
                                                  line_num], self.off_value_tensor_ub, 0, 1, self.depth *
                                        line_num //
                                        self.off_value_each_block, 0, 0)
            self.align_to_32_last_block(
                line_num * self.depth,
                i * self.depth * line_num,
                id_number)

            with self.tik_instance.for_range(0, line_num) as j:
                self.index_scalar.set_as(self.x_ub[i * line_num + j])
                with self.tik_instance.if_scope(tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                    self.index_scalar.set_as(j * self.depth + self.index_scalar)
                    self.off_value_tensor_ub[self.index_scalar].set_as(self.off_value)

        with self.tik_instance.if_scope((end - begin) % line_num > 0):
            with self.tik_instance.for_range(0, (end - begin) % line_num) as k:
                self.index_scalar.set_as(
                    self.x_ub[(end - begin) // line_num * line_num + k])
                with self.tik_instance.if_scope(tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                    self.index_scalar.set_as(k * self.depth + self.index_scalar)
                    self.off_value_tensor_ub[self.index_scalar].set_as(self.on_value)
            self.tik_instance.data_move(self.y_gm[id_number *
                                                  self.not_last_core_numel *
                                                  self.depth +
                                                  (end -
                                                   begin) //
                                                  line_num *
                                                  line_num *
                                                  self.depth], self.off_value_tensor_ub, 0, 1, (end -
                                                                                                begin) %
                                        line_num *
                                        self.depth //
                                        self.off_value_each_block, 0, 0)
        self.align_to_32_last_block(
            (end - begin) % line_num * self.depth,
            (end - begin) // line_num * line_num * self.depth,
            id_number)

    # last axis with ub size is more than x and smaller than off_value_tensor
    # one line
    def one_hot_last_axis_third_mode(self, id_number):
        """
        the third calculate mode when the axis is 0

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_numel * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_numel + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_numel + begin)
        move_numel = self.tik_instance.Scalar(dtype='int32', name='move_numel')
        move_numel.set_as(
            ((((end - begin) - 1) // self.x_each_block + 1) * self.x_each_block))
        self.tik_instance.data_move(self.x_ub,
                                    self.x_gm[id_number * self.not_last_core_numel],
                                    0,
                                    1,
                                    move_numel // self.x_each_block,
                                    0,
                                    0)
        part_num = self.tik_instance.Scalar(dtype='int32', name='part_num')
        part_num.set_as(self.per_part_unused_ub * self.off_value_tensor_part)
        self.vec_dump_off_value_tensor_ub(part_num)
        move_times = self.tik_instance.Scalar(dtype='int32', name='move_times')
        move_times.set_as((self.depth * (end - begin)) // part_num)

        with self.tik_instance.for_range(0, move_times) as k:
            self.tik_instance.data_move(self.y_gm[id_number *
                                                  self.not_last_core_numel *
                                                  self.depth +
                                                  k *
                                                  part_num], self.off_value_tensor_ub, 0, 1, part_num //
                                        self.off_value_each_block, 0, 0)
        with self.tik_instance.if_scope(self.depth * (end - begin) % part_num // self.off_value_each_block > 0):
            self.tik_instance.data_move(
                self.y_gm[id_number * self.not_last_core_numel * self.depth + move_times * part_num],
                self.off_value_tensor_ub, 0, 1,
                (self.depth * (end - begin)) % part_num // self.off_value_each_block, 0, 0)
            self.align_to_32_last_block(
                self.depth * (end - begin) % part_num, move_times * part_num, id_number)

        block_index = self.tik_instance.Scalar(
            dtype='int32', name='block_index')
        max_y_index = self.tik_instance.Scalar(
            dtype='int32', name='max_y_index')
        max_y_index.set_as(
            self.depth //
            self.off_value_each_block *
            self.off_value_each_block)

        with self.tik_instance.for_range(0, (end - begin)) as i:
            self.index_scalar.set_as(self.x_ub[i])
            with self.tik_instance.if_scope(tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                with self.tik_instance.if_scope(tik.all(self.depth % self.off_value_each_block > 0,
                                                        self.index_scalar > max_y_index)):
                    block_index.set_as(self.index_scalar % max_y_index + (self.off_value_each_block -
                                      (self.depth - max_y_index)))
                    self.off_value_tensor_ub[block_index].set_as(self.on_value)
                    self.tik_instance.data_move(self.y_gm[id_number * self.not_last_core_numel *
                                                          self.depth + i * self.depth +
                                                          max_y_index - (self.off_value_each_block -
                                                           (self.depth - max_y_index))],
                                                self.off_value_tensor_ub, 0, 1, 1, 0, 0)
                    self.off_value_tensor_ub[block_index].set_as(self.off_value)

                with self.tik_instance.else_scope():
                    block_index.set_as(self.index_scalar % self.off_value_each_block)
                    self.off_value_tensor_ub[block_index].set_as(self.on_value)
                    self.tik_instance.data_move(self.y_gm[id_number * self.not_last_core_numel *
                                                          self.depth + i * self.depth +
                                                          self.index_scalar // self.off_value_each_block *
                                                          self.off_value_each_block], self.off_value_tensor_ub, 0, 1,
                                                1, 0, 0)
                    self.off_value_tensor_ub[block_index].set_as(self.off_value)

    # last axis with ub size is less than x and enough to off_value_tensor
    # some lines
    # 'pylint: disable =too-many-statements
    def one_hot_last_axis_fourth_mode(self, id_number):
        """
        the fourth calculate mode when the axis is 0

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_numel * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_numel + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_numel + begin)

        x_move_times = self.tik_instance.Scalar(
            dtype='int32', name='x_move_times')
        line_num = self.tik_instance.Scalar(dtype='int32', name='line_num')
        line_num.set_as(
            self.per_part_unused_ub *
            self.off_value_tensor_part //
            self.depth)
        x_move_times.set_as(((end - begin) - 1) // self.per_part_unused_ub + 1)

        off_value_ub_size = self.tik_instance.Scalar(
            dtype='int32', name='off_value_ub_size')
        off_value_ub_size.set_as(
            self.per_part_unused_ub *
            self.off_value_tensor_part)
        self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        offset = self.tik_instance.Scalar(dtype='int32', name='offset')

        with self.tik_instance.for_range(0, x_move_times) as i:
            self.tik_instance.data_move(self.x_ub,
                                        self.x_gm[i * self.per_part_unused_ub + id_number * self.not_last_core_numel],
                                        0,
                                        1,
                                        self.per_part_unused_ub // self.x_each_block,
                                        0,
                                        0)
            with self.tik_instance.if_scope(i == x_move_times - 1):
                with self.tik_instance.if_scope((end - begin) % self.per_part_unused_ub // line_num > 0):
                    with self.tik_instance.for_range(0,
                                                     (end - begin) %
                                                     self.per_part_unused_ub // line_num) as k:
                        with self.tik_instance.for_range(0, line_num) as ele:
                            self.index_scalar.set_as(
                                self.x_ub[k * line_num + ele])
                            with self.tik_instance.if_scope(
                                    tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                                self.index_scalar.set_as(
                                    ele * self.depth + self.index_scalar)
                                self.off_value_tensor_ub[self.index_scalar].set_as(
                                    self.on_value)

                        offset.set_as(((x_move_times - 1) * self.per_part_unused_ub + k * line_num) * self.depth)
                        self.tik_instance.data_move(self.y_gm[id_number *
                                                              self.not_last_core_numel *
                                                              self.depth +
                                                              offset], self.off_value_tensor_ub, 0, 1, self.depth *
                                                    line_num //
                                                    self.off_value_each_block, 0, 0)
                        self.align_to_32_last_block(
                            self.depth * line_num, offset, id_number)

                        with self.tik_instance.for_range(0, line_num) as ele:
                            self.index_scalar.set_as(
                                self.x_ub[k * line_num + ele])
                            with self.tik_instance.if_scope(tik.all(self.index_scalar >= 0,
                                                                    self.index_scalar < self.depth)):
                                self.index_scalar.set_as(ele * self.depth + self.index_scalar)
                                self.off_value_tensor_ub[self.index_scalar].set_as(self.off_value)

                remain = self.tik_instance.Scalar(dtype='int32', name='remain')
                remain.set_as(
                    (end - begin) %
                    self.per_part_unused_ub %
                    line_num)
                with self.tik_instance.if_scope(remain > 0):
                    with self.tik_instance.for_range(0, remain) as ele:
                        self.index_scalar.set_as(self.x_ub[
                            (end - begin) % self.per_part_unused_ub //
                            line_num * line_num + ele])
                        with self.tik_instance.if_scope(
                                tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                            self.index_scalar.set_as(ele * self.depth + self.index_scalar)
                            self.off_value_tensor_ub[self.index_scalar].set_as(self.on_value)

                    offset.set_as(((x_move_times - 1) *
                                   self.per_part_unused_ub +
                                   (end - begin) %
                                   self.per_part_unused_ub // line_num * line_num) * self.depth)
                    with self.tik_instance.if_scope(self.depth * remain // self.off_value_each_block > 0):
                        self.tik_instance.data_move(self.y_gm[offset + self.not_last_core_numel *
                                                              id_number * self.depth], self.off_value_tensor_ub,
                                                    0, 1, self.depth * remain // self.off_value_each_block, 0, 0)
                        self.align_to_32_last_block(self.depth * remain, offset, id_number)

            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.per_part_unused_ub // line_num > 0):
                    with self.tik_instance.for_range(0, self.per_part_unused_ub // line_num) as k:
                        with self.tik_instance.for_range(0, line_num) as ele:
                            self.index_scalar.set_as(
                                self.x_ub[k * line_num + ele])
                            with self.tik_instance.if_scope(
                                    tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                                self.index_scalar.set_as(
                                    ele * self.depth + self.index_scalar)
                                self.off_value_tensor_ub[self.index_scalar].set_as(
                                    self.on_value)
                        offset = self.tik_instance.Scalar(
                            dtype='int32', name='offset')
                        offset.set_as(
                            (i * self.per_part_unused_ub + k * line_num) * self.depth)
                        self.tik_instance.data_move(self.y_gm[self.not_last_core_numel *
                                                              id_number *
                                                              self.depth +
                                                              offset], self.off_value_tensor_ub, 0, 1, self.depth *
                                                    line_num //
                                                    self.off_value_each_block, 0, 0)

                        self.align_to_32_last_block(
                            self.depth * line_num, offset, id_number)

                        with self.tik_instance.for_range(0, line_num) as ele:
                            self.index_scalar.set_as(self.x_ub[k * line_num + ele])
                            with self.tik_instance.if_scope(
                                    tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                                self.index_scalar.set_as(ele * self.depth + self.index_scalar)
                                self.off_value_tensor_ub[self.index_scalar].set_as(self.off_value)

                with self.tik_instance.if_scope(self.per_part_unused_ub % line_num > 0):
                    with self.tik_instance.for_range(0, self.per_part_unused_ub % line_num) as ele:
                        self.index_scalar.set_as(self.x_ub[self.per_part_unused_ub // line_num * line_num + ele])
                        with self.tik_instance.if_scope(
                                tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                            self.index_scalar.set_as(ele * self.depth + self.index_scalar)
                            self.off_value_tensor_ub[self.index_scalar].set_as(self.on_value)

                    offset = self.tik_instance.Scalar(dtype='int32', name='offset')
                    offset.set_as((i * self.per_part_unused_ub + self.per_part_unused_ub // line_num *
                                   line_num) * self.depth)
                    self.tik_instance.data_move(self.y_gm[id_number *
                                                          self.not_last_core_numel *
                                                          self.depth +
                                                          offset], self.off_value_tensor_ub, 0, 1,
                                                self.per_part_unused_ub %
                                                line_num *
                                                self.depth //
                                                self.off_value_each_block, 0, 0)
                    self.align_to_32_last_block(self.per_part_unused_ub % line_num * self.depth, offset, id_number)

                    with self.tik_instance.for_range(0, self.per_part_unused_ub % line_num) as ele:
                        self.index_scalar.set_as(self.x_ub[self.per_part_unused_ub // line_num * line_num + ele])
                        with self.tik_instance.if_scope(
                                tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                            self.index_scalar.set_as(ele * self.depth + self.index_scalar)
                            self.off_value_tensor_ub[self.index_scalar].set_as(self.off_value)

    # last axis with ub size is less than x smaller than off_value_tensor one
    # line
    def one_hot_last_axis_fifth_mode(self, id_number):
        """
        the fifth calculate mode when the axis is 0

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_numel * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_numel + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_numel + begin)
        x_move_times = self.tik_instance.Scalar(
            dtype='int32', name='x_move_times')
        part_num = self.tik_instance.Scalar(dtype='int32', name='part_num')
        part_num.set_as(self.per_part_unused_ub * self.off_value_tensor_part)
        x_move_times.set_as((end - begin - 1) // self.per_part_unused_ub + 1)

        self.vec_dump_off_value_tensor_ub(part_num)
        move_times = self.tik_instance.Scalar(dtype='int32', name='move_times')
        move_times.set_as((self.depth * (end - begin)) // part_num)
        with self.tik_instance.for_range(0, move_times) as k:
            self.tik_instance.data_move(self.y_gm[id_number *
                                                  self.not_last_core_numel *
                                                  self.depth +
                                                  k *
                                                  part_num], self.off_value_tensor_ub, 0, 1, part_num //
                                        self.off_value_each_block, 0, 0)
        with self.tik_instance.if_scope(self.depth * (end - begin) % part_num // self.off_value_each_block > 0):
            self.tik_instance.data_move(self.y_gm[id_number * self.not_last_core_numel *
                                                  self.depth + move_times * part_num],
                self.off_value_tensor_ub, 0, 1,
                (self.depth * (end - begin)) % part_num // self.off_value_each_block, 0, 0)
            self.align_to_32_last_block(self.depth * (end - begin) % part_num, move_times * part_num, id_number)

        with self.tik_instance.for_range(0, x_move_times) as i:
            self.tik_instance.data_move(self.x_ub,
                                        self.x_gm[self.not_last_core_numel * id_number + i * self.per_part_unused_ub],
                                        0,
                                        1,
                                        self.per_part_unused_ub // self.x_each_block,
                                        0,
                                        0)
            with self.tik_instance.if_scope(i == x_move_times - 1):
                with self.tik_instance.for_range(0, (end - begin) % self.per_part_unused_ub) as index:
                    self.index_scalar.set_as(self.x_ub[index])
                    with self.tik_instance.if_scope(tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                        with self.tik_instance.for_range(0, Constant.cal_num) as j:
                            with self.tik_instance.if_scope(
                                    tik.all(self.index_scalar < part_num * (j + 1), self.index_scalar >= part_num * j)):
                                self.index_scalar.set_as(self.index_scalar - (j * part_num))
                                self.off_value_tensor_ub[self.index_scalar].set_as(self.on_value)
                                self.tik_instance.data_move(self.y_gm[((end -
                                                                        begin) //
                                                                       self.per_part_unused_ub *
                                                                       self.per_part_unused_ub +
                                                                       index) *
                                                                      self.depth +
                                                                      j *
                                                                      part_num], self.off_value_tensor_ub,
                                                            0, 1, part_num //
                                                            self.off_value_each_block, 0, 0)
                                self.off_value_tensor_ub[self.index_scalar].set_as(self.off_value)
            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, self.per_part_unused_ub) as index:
                    self.index_scalar.set_as(self.x_ub[index])
                    with self.tik_instance.if_scope(tik.all(self.index_scalar >= 0, self.index_scalar < self.depth)):
                        with self.tik_instance.for_range(0, Constant.cal_num) as j:
                            with self.tik_instance.if_scope(
                                    tik.all(self.index_scalar < part_num * (j + 1), self.index_scalar >= part_num * j)):
                                self.index_scalar.set_as(self.index_scalar - (j * part_num))
                                self.off_value_tensor_ub[self.index_scalar].set_as(self.on_value)
                                self.tik_instance.data_move(self.y_gm[id_number *
                                                                      self.not_last_core_numel +
                                                                      (i *
                                                                       self.per_part_unused_ub +
                                                                       index) *
                                                                      self.depth +
                                                                      j *
                                                                      part_num], self.off_value_tensor_ub,
                                                            0, 1, part_num //
                                                            self.off_value_each_block, 0, 0)
                                self.off_value_tensor_ub[self.index_scalar].set_as(self.off_value)

    # first axis with ub enough for all
    def one_hot_first_axis_first_mode(self, id_number):
        """
        the first calculate mode when the axis is -1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_index * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_index + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_index + begin)
        with self.tik_instance.if_scope(self.numel_shape_x // self.x_each_block > 0):
            self.tik_instance.data_move(
                self.x_ub,
                self.x_gm,
                0,
                1,
                self.numel_shape_x //
                self.x_each_block,
                0,
                0)
        with self.tik_instance.if_scope(self.numel_shape_x % self.x_each_block > 0):
            self.tik_instance.data_move(self.x_ub[self.numel_shape_x //
                                                  self.x_each_block *
                                                  self.x_each_block], self.x_gm[self.numel_shape_x //
                                                                                self.x_each_block *
                                                                                self.x_each_block], 0, 1, 1, 0, 0)

        off_value_ub_size = self.tik_instance.Scalar(
            dtype='int32', name='off_value_ub_size')
        with self.tik_instance.if_scope((self.numel_shape_x * (end - begin)) <
                                        (self.per_part_unused_ub * self.off_value_tensor_part)):
            off_value_ub_size.set_as(self.numel_shape_x * (end - begin))
        with self.tik_instance.else_scope():
            off_value_ub_size.set_as(
                self.per_part_unused_ub *
                self.off_value_tensor_part)
        self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        with self.tik_instance.for_range(0, self.numel_shape_x) as i:
            self.index_scalar.set_as(self.x_ub[i])
            with self.tik_instance.if_scope(tik.all(self.index_scalar < end, self.index_scalar >= begin)):
                self.index_scalar.set_as(i +
                                         self.numel_shape_x *
                                         (self.index_scalar -
                                          id_number *
                                          self.not_last_core_index))
                self.off_value_tensor_ub[self.index_scalar].set_as(
                    self.on_value)

        with self.tik_instance.if_scope((end - begin) * self.numel_shape_x // self.off_value_each_block > 0):
            self.tik_instance.data_move(self.y_gm[begin *
                                                  self.numel_shape_x], self.off_value_tensor_ub, 0, 1, (end -
                                                                                                        begin) *
                                        self.numel_shape_x //
                                        self.off_value_each_block, 0, 0)
            self.align_to_32_last_block_first_dim(
                (end - begin) * self.numel_shape_x, 0, id_number)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.y_gm[begin * self.numel_shape_x],
                                        self.off_value_tensor_ub, 0, 1, 1, 0, 0)

    # first axis with ub size is more than x and enough to off_value_tensor
    # some lines
    def one_hot_first_axis_second_mode(self, id_number):
        """
        the second calculate mode when the axis is -1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_index * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_index + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_index + begin)
        with self.tik_instance.if_scope(self.numel_shape_x // self.x_each_block > 0):
            self.tik_instance.data_move(
                self.x_ub,
                self.x_gm,
                0,
                1,
                self.numel_shape_x //
                self.x_each_block,
                0,
                0)
        with self.tik_instance.if_scope(self.numel_shape_x % self.x_each_block > 0):
            self.tik_instance.data_move(self.x_ub[self.numel_shape_x //
                                                  self.x_each_block *
                                                  self.x_each_block], self.x_gm[self.numel_shape_x //
                                                                                self.x_each_block *
                                                                                self.x_each_block], 0, 1, 1, 0, 0)

        off_value_tensor_ub_size = self.tik_instance.Scalar(
            dtype='int32', name='off_value_tensor_ub_size')
        off_value_tensor_ub_size.set_as(
            self.per_part_unused_ub *
            self.off_value_tensor_part)
        self.vec_dump_off_value_tensor_ub(off_value_tensor_ub_size)
        move_times = self.tik_instance.Scalar(dtype='int32', name='move_times')
        move_times.set_as(
            (end -
             begin) *
            self.numel_shape_x //
            off_value_tensor_ub_size)
        offset = self.tik_instance.Scalar(dtype='int32', name='offset')
        with self.tik_instance.for_range(0, move_times) as k:
            self.tik_instance.data_move(self.y_gm[begin * self.numel_shape_x + k * off_value_tensor_ub_size],
                                        self.off_value_tensor_ub, 0, 1,
                                        off_value_tensor_ub_size // self.off_value_each_block, 0, 0)
        with self.tik_instance.if_scope(self.numel_shape_x * (
                end - begin) % off_value_tensor_ub_size // self.off_value_each_block > 0):
            self.tik_instance.data_move(self.y_gm[begin *
                                                  self.numel_shape_x +
                                                  move_times *
                                                  off_value_tensor_ub_size], self.off_value_tensor_ub, 0, 1,
                                        self.numel_shape_x *
                                        (end -
                                         begin) %
                                        off_value_tensor_ub_size //
                                        self.off_value_each_block, 0, 0)
            self.align_to_32_last_block_first_dim(
                self.numel_shape_x *
                (end - begin) %
                off_value_tensor_ub_size,
                move_times *
                off_value_tensor_ub_size,
                id_number)

        with self.tik_instance.for_range(0, self.numel_shape_x) as i:
            self.index_scalar.set_as(self.x_ub[i])
            with self.tik_instance.if_scope(tik.all(self.index_scalar < end, self.index_scalar >= begin)):
                with self.tik_instance.if_scope(self.numel_shape_x // self.off_value_each_block > 0):
                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                self.y_gm[self.index_scalar * self.numel_shape_x], 0, 1,
                                                (self.numel_shape_x - 1) // self.off_value_each_block + 1, 0, 0)
                    self.off_value_tensor_ub[i].set_as(self.on_value)
                    self.tik_instance.data_move(self.y_gm[self.index_scalar * self.numel_shape_x],
                                                self.off_value_tensor_ub, 0, 1,
                                                self.numel_shape_x // self.off_value_each_block, 0, 0)
                    self.align_to_32_last_block_first_dim(
                        self.numel_shape_x, (self.index_scalar - begin) * self.numel_shape_x, id_number)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope((self.index_scalar * self.numel_shape_x) >
                                                    (end * self.numel_shape_x - self.off_value_each_block)):
                        offset.set_as(self.off_value_each_block -
                                      (end *
                                       self.numel_shape_x -
                                       self.index_scalar *
                                       self.numel_shape_x))
                        self.tik_instance.data_move(self.off_value_tensor_ub,
                                                    self.y_gm[self.index_scalar * self.numel_shape_x - offset],
                                                    0, 1, 1, 0, 0)
                        self.off_value_tensor_ub[i +
                                                 offset].set_as(self.on_value)
                        self.tik_instance.data_move(self.y_gm[self.index_scalar * self.numel_shape_x - offset],
                                                    self.off_value_tensor_ub, 0, 1, 1, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(self.off_value_tensor_ub,
                                                    self.y_gm[self.index_scalar * self.numel_shape_x],
                                                    0, 1, 1, 0, 0)
                        self.off_value_tensor_ub[i].set_as(self.on_value)
                        self.tik_instance.data_move(self.y_gm[self.index_scalar * self.numel_shape_x],
                                                    self.off_value_tensor_ub, 0, 1, 1, 0, 0)

    # first axis with ub size is less than x and enough to off_value_tensor
    # some lines
    def one_hot_first_axis_third_mode(self, id_number):
        """
        the third calculate mode when the axis is -1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_index * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_index + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_index + begin)
        x_move_times = self.tik_instance.Scalar(
            dtype='int32', name='x_move_times')
        off_value_tensor_ub_size = self.tik_instance.Scalar(
            dtype='int32', name='off_value_tensor_ub_size')
        off_value_tensor_ub_size.set_as(
            self.per_part_unused_ub *
            self.off_value_tensor_part)
        x_move_times.set_as((self.numel_shape_x - 1) //
                            self.per_part_unused_ub + 1)
        self.vec_dump_off_value_tensor_ub(off_value_tensor_ub_size)

        move_times = self.tik_instance.Scalar(dtype='int32', name='move_times')
        move_times.set_as(
            (end -
             begin) *
            self.numel_shape_x //
            off_value_tensor_ub_size)
        with self.tik_instance.for_range(0, move_times) as k:
            self.tik_instance.data_move(self.y_gm[begin * self.numel_shape_x + k * off_value_tensor_ub_size],
                                        self.off_value_tensor_ub, 0, 1,
                                        off_value_tensor_ub_size // self.off_value_each_block, 0, 0)
        with self.tik_instance.if_scope(self.numel_shape_x * (
                end - begin) % off_value_tensor_ub_size // self.off_value_each_block > 0):
            self.tik_instance.data_move(self.y_gm[begin *
                                                  self.numel_shape_x +
                                                  move_times *
                                                  off_value_tensor_ub_size], self.off_value_tensor_ub, 0, 1,
                                        self.numel_shape_x * (end - begin) %
                                        off_value_tensor_ub_size //
                                        self.off_value_each_block, 0, 0)
            self.align_to_32_last_block_first_dim(
                self.numel_shape_x *
                (
                    end -
                    begin) %
                off_value_tensor_ub_size,
                move_times *
                off_value_tensor_ub_size,
                id_number)

        with self.tik_instance.for_range(0, x_move_times) as i:
            self.tik_instance.data_move(self.x_ub,
                                        self.x_gm[i * self.per_part_unused_ub],
                                        0,
                                        1,
                                        self.per_part_unused_ub // self.x_each_block,
                                        0,
                                        0)
            with self.tik_instance.if_scope(
                    tik.all(i == x_move_times - 1, self.numel_shape_x % self.per_part_unused_ub > 0)):
                with self.tik_instance.for_range(0, self.numel_shape_x % self.per_part_unused_ub) as j:
                    self.index_scalar.set_as(self.x_ub[j])
                    with self.tik_instance.if_scope(tik.all(self.index_scalar < end, self.index_scalar >= begin)):
                        self.tik_instance.data_move(self.off_value_tensor_ub,
                                                    self.y_gm[self.index_scalar * self.numel_shape_x], 0, 1,
                                                    (self.numel_shape_x - 1) // self.off_value_each_block + 1, 0, 0)
                        self.off_value_tensor_ub[i *
                                                 self.per_part_unused_ub +
                                                 j].set_as(self.on_value)
                        self.tik_instance.data_move(self.y_gm[self.index_scalar * self.numel_shape_x],
                                                    self.off_value_tensor_ub, 0, 1,
                                                    self.numel_shape_x // self.off_value_each_block, 0, 0)
                        self.align_to_32_last_block_first_dim(
                            self.numel_shape_x, (self.index_scalar - begin) * self.numel_shape_x, id_number)

            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, self.per_part_unused_ub) as j:
                    self.index_scalar.set_as(self.x_ub[j])
                    with self.tik_instance.if_scope(tik.all(self.index_scalar < end, self.index_scalar >= begin)):
                        self.tik_instance.data_move(self.off_value_tensor_ub,
                                                    self.y_gm[self.index_scalar * self.numel_shape_x], 0, 1,
                                                    (self.numel_shape_x - 1) // self.off_value_each_block + 1, 0, 0)
                        self.off_value_tensor_ub[i *
                                                 self.per_part_unused_ub +
                                                 j].set_as(self.on_value)
                        self.tik_instance.data_move(self.y_gm[self.index_scalar * self.numel_shape_x],
                                                    self.off_value_tensor_ub, 0, 1,
                                                    self.numel_shape_x // self.off_value_each_block, 0, 0)
                        self.align_to_32_last_block_first_dim(
                            self.numel_shape_x, (self.index_scalar - begin) * self.numel_shape_x, id_number)

    # first axis with ub size is less than x smaller than off_value_tensor one
    # line
    def one_hot_first_axis_fourth_mode(self, id_number):
        """
        the fourth calculate mode when the axis is -1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_index * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_index + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_index + begin)
        index_fill_on_value = self.tik_instance.Scalar(
            dtype='int32', name='index_fill_on_value')
        part_num = self.tik_instance.Scalar(dtype='int32', name='part_num')
        part_num.set_as(self.per_part_unused_ub * self.off_value_tensor_part)
        x_move_times = self.tik_instance.Scalar(
            dtype='int32', name='x_move_times')
        x_move_times.set_as((self.numel_shape_x - 1) //
                            self.per_part_unused_ub + 1)
        self.vec_dump_off_value_tensor_ub(part_num)
        offset = self.tik_instance.Scalar(dtype='int32', name='offset')
        burst_len = self.tik_instance.Scalar(dtype='int32', name='burst_len')

        move_times = self.tik_instance.Scalar(dtype='int32', name='move_times')
        move_times.set_as(((end - begin) * self.numel_shape_x) // part_num)
        with self.tik_instance.for_range(0, move_times) as k:
            self.tik_instance.data_move(self.y_gm[begin *
                                                  self.numel_shape_x +
                                                  k *
                                                  part_num], self.off_value_tensor_ub, 0, 1, part_num //
                                        self.off_value_each_block, 0, 0)
        with self.tik_instance.if_scope(self.numel_shape_x * (
                end - begin) % part_num // self.off_value_each_block > 0):
            self.tik_instance.data_move(self.y_gm[begin *
                                                  self.numel_shape_x +
                                                  move_times *
                                                  part_num], self.off_value_tensor_ub, 0, 1, self.numel_shape_x *
                                        (end -
                                         begin) %
                                        part_num //
                                        self.off_value_each_block, 0, 0)
            self.align_to_32_last_block_first_dim(
                self.numel_shape_x * (end - begin) % part_num, move_times * part_num, id_number)

        with self.tik_instance.for_range(0, x_move_times) as index:
            self.tik_instance.data_move(self.x_ub,
                                        self.x_gm[index * self.per_part_unused_ub],
                                        0,
                                        1,
                                        self.per_part_unused_ub // self.x_each_block,
                                        0,
                                        0)
            with self.tik_instance.if_scope(
                    tik.all(index == x_move_times - 1, self.numel_shape_x % self.per_part_unused_ub > 0)):
                with self.tik_instance.for_range(0, self.numel_shape_x % self.per_part_unused_ub) as k:
                    self.index_scalar.set_as(self.x_ub[k])
                    with self.tik_instance.if_scope(tik.all(self.index_scalar < end, self.index_scalar >= begin)):
                        with self.tik_instance.for_range(0, Constant.cal_num) as j:
                            with self.tik_instance.if_scope(
                                    tik.all(((self.numel_shape_x // self.per_part_unused_ub * self.per_part_unused_ub)
                                             + k) < part_num * (j + 1), ((self.numel_shape_x //
                                                                          self.per_part_unused_ub
                                                                          * self.per_part_unused_ub)
                                                                         + k) >= part_num * j)):
                                with self.tik_instance.if_scope(part_num * (j + 1) > self.numel_shape_x):
                                    offset.set_as(((self.numel_shape_x -
                                                    part_num *
                                                    j -
                                                    1) //
                                                   Constant.SCALAR_TENSOR_SIZE +
                                                   1) *
                                                  Constant.SCALAR_TENSOR_SIZE -
                                                  (self.numel_shape_x -
                                                   part_num *
                                                   j))
                                    burst_len.set_as((offset + (
                                        self.numel_shape_x - part_num * j)) // self.off_value_each_block)
                                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                                self.y_gm[self.index_scalar * self.numel_shape_x + j
                                                                          * part_num - offset], 0, 1,
                                                                burst_len, 0, 0)
                                    index_fill_on_value.set_as(k +
                                                               self.numel_shape_x //
                                                               self.per_part_unused_ub *
                                                               self.per_part_unused_ub -
                                                               (j *
                                                                part_num) +
                                                               offset)
                                    self.off_value_tensor_ub[index_fill_on_value].set_as(
                                        self.on_value)
                                    self.tik_instance.data_move(self.y_gm[self.index_scalar * self.numel_shape_x + j
                                                                          * part_num - offset],
                                                                self.off_value_tensor_ub,
                                                                0, 1, burst_len, 0, 0)
                                with self.tik_instance.else_scope():
                                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                                self.y_gm[self.index_scalar * self.numel_shape_x + j
                                                                          * part_num], 0, 1,
                                                                part_num // self.off_value_each_block, 0, 0)
                                    index_fill_on_value.set_as(k + self.numel_shape_x // self.per_part_unused_ub *
                                                               self.per_part_unused_ub - (j * part_num))
                                    self.off_value_tensor_ub[index_fill_on_value].set_as(
                                        self.on_value)
                                    self.tik_instance.data_move(self.y_gm[self.index_scalar *
                                                                          self.numel_shape_x +
                                                                          j *
                                                                          part_num], self.off_value_tensor_ub, 0, 1,
                                                                part_num // self.off_value_each_block, 0, 0)

            with self.tik_instance.else_scope():
                with self.tik_instance.for_range(0, self.per_part_unused_ub) as i:
                    self.index_scalar.set_as(self.x_ub[i])
                    with self.tik_instance.if_scope(tik.all(self.index_scalar < end, self.index_scalar >= begin)):
                        with self.tik_instance.for_range(0, Constant.cal_num) as j:
                            with self.tik_instance.if_scope(
                                    tik.all((i + index * self.per_part_unused_ub) < part_num * (j + 1),
                                            (i + index * self.per_part_unused_ub) >= part_num * j)):
                                with self.tik_instance.if_scope(part_num * (j + 1) > self.numel_shape_x):
                                    offset.set_as(((self.numel_shape_x -
                                                    part_num *
                                                    j -
                                                    1) //
                                                   Constant.SCALAR_TENSOR_SIZE +
                                                   1) *
                                                  Constant.SCALAR_TENSOR_SIZE -
                                                  (self.numel_shape_x -
                                                   part_num *
                                                   j))
                                    burst_len.set_as((offset + (
                                        self.numel_shape_x - part_num * j)) // self.off_value_each_block)
                                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                                self.y_gm[
                                                                    self.index_scalar * self.numel_shape_x + j
                                                                    * part_num - offset], 0, 1,
                                                                burst_len, 0, 0)
                                    index_fill_on_value.set_as(
                                        (i + index * self.per_part_unused_ub) - (j * part_num) + offset)
                                    self.off_value_tensor_ub[index_fill_on_value].set_as(
                                        self.on_value)
                                    self.tik_instance.data_move(
                                        self.y_gm[self.index_scalar * self.numel_shape_x + j
                                                  * part_num - offset], self.off_value_tensor_ub,
                                        0, 1, burst_len, 0, 0)
                                with self.tik_instance.else_scope():
                                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                                self.y_gm[self.index_scalar * self.numel_shape_x
                                                                          + j * part_num],
                                                                0,
                                                                1,
                                                                part_num // self.off_value_each_block,
                                                                0,
                                                                0)
                                    index_fill_on_value.set_as(
                                        (i + index * self.per_part_unused_ub) - (j * part_num))
                                    self.off_value_tensor_ub[index_fill_on_value].set_as(
                                        self.on_value)
                                    self.tik_instance.data_move(
                                        self.y_gm[self.index_scalar * self.numel_shape_x + j
                                                  * part_num], self.off_value_tensor_ub,
                                        0, 1, part_num // self.off_value_each_block, 0, 0)

    # middle axis with ub enough for all
    def one_hot_middle_axis_first_mode(self, id_number):
        """
        the first calculate mode when the axis is 0 < axis < len(x_shape) - 1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_numel * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_numel + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_numel + begin)
        move_numel = self.tik_instance.Scalar(dtype='int32', name='move_numel')
        move_numel.set_as(((((end - begin) * self.last_dim_x - 1) //
                            self.x_each_block + 1) * self.x_each_block))
        self.tik_instance.data_move(self.x_ub,
                                    self.x_gm[id_number * self.not_last_core_numel * self.last_dim_x],
                                    0,
                                    1,
                                    move_numel // self.x_each_block,
                                    0,
                                    0)

        off_value_ub_size = self.tik_instance.Scalar(
            dtype='int32', name='off_value_ub_size')
        with self.tik_instance.if_scope((move_numel * self.depth) <
                                        (self.per_part_unused_ub * self.off_value_tensor_part)):
            off_value_ub_size.set_as(move_numel * self.depth)
        with self.tik_instance.else_scope():
            off_value_ub_size.set_as(
                self.per_part_unused_ub *
                self.off_value_tensor_part)
        self.vec_dump_off_value_tensor_ub(off_value_ub_size)
        with self.tik_instance.for_range(0, (end - begin)) as i:
            with self.tik_instance.for_range(0, self.last_dim_x) as j:
                self.index_scalar.set_as(self.x_ub[i * self.last_dim_x + j])
                self.index_scalar.set_as(
                    i *
                    self.last_dim_x *
                    self.depth +
                    j +
                    self.last_dim_x *
                    self.index_scalar)
                self.off_value_tensor_ub[self.index_scalar].set_as(
                    self.on_value)
        with self.tik_instance.if_scope((end - begin) * self.depth * self.last_dim_x // self.off_value_each_block > 0):
            self.tik_instance.data_move(self.y_gm[begin *
                                                  self.depth *
                                                  self.last_dim_x], self.off_value_tensor_ub, 0, 1, (end -
                                                                                                     begin) *
                                        self.depth *
                                        self.last_dim_x //
                                        self.off_value_each_block, 0, 0)
            self.align_to_32_last_block((end -
                                         begin) *
                                        self.last_dim_x *
                                        self.depth, self.not_last_core_numel *
                                        (self.last_dim_x -
                                         1) *
                                        self.depth *
                                        id_number, id_number)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.y_gm[begin * self.depth * self.last_dim_x],
                                        self.off_value_tensor_ub, 0, 1, 1, 0, 0)

    # middle axis with ub size is more than x and enough to off_value_tensor
    # some lines
    def one_hot_middle_axis_second_mode(self, id_number):
        """
        the second calculate mode when the axis is 0 < axis < len(x_shape) - 1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_numel * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_numel + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_numel + begin)
        move_numel = self.tik_instance.Scalar(dtype='int32', name='move_numel')
        move_numel.set_as(((((end - begin) * self.last_dim_x - 1) //
                            self.x_each_block + 1) * self.x_each_block))
        self.tik_instance.data_move(self.x_ub,
                                    self.x_gm[id_number * self.not_last_core_numel * self.last_dim_x],
                                    0,
                                    1,
                                    move_numel // self.x_each_block,
                                    0,
                                    0)

        off_value_tensor_ub_size = self.tik_instance.Scalar(
            dtype='int32', name='off_value_tensor_ub_size')
        off_value_tensor_ub_size.set_as(
            self.per_part_unused_ub *
            self.off_value_tensor_part)
        self.vec_dump_off_value_tensor_ub(off_value_tensor_ub_size)

        move_times = self.tik_instance.Scalar(dtype='int32', name='move_times')
        move_times.set_as((self.depth * (end - begin) *
                           self.last_dim_x) // off_value_tensor_ub_size)
        with self.tik_instance.for_range(0, move_times) as k:
            self.tik_instance.data_move(self.y_gm[begin *
                                                  self.depth *
                                                  self.last_dim_x +
                                                  k *
                                                  off_value_tensor_ub_size], self.off_value_tensor_ub, 0, 1,
                                        off_value_tensor_ub_size // self.off_value_each_block, 0, 0)
        with self.tik_instance.if_scope(self.depth * self.last_dim_x * (end - begin)
                                        % off_value_tensor_ub_size // self.off_value_each_block > 0):
            self.tik_instance.data_move(
                self.y_gm[begin * self.depth * self.last_dim_x + move_times * off_value_tensor_ub_size],
                self.off_value_tensor_ub, 0, 1,
                self.last_dim_x * self.depth * (end - begin) % off_value_tensor_ub_size //
                self.off_value_each_block, 0, 0)
            self.align_to_32_last_block(self.last_dim_x *
                                        self.depth *
                                        (end -
                                         begin) %
                                        off_value_tensor_ub_size, self.not_last_core_numel *
                                        (self.last_dim_x -
                                         1) *
                                        self.depth *
                                        id_number +
                                        move_times *
                                        off_value_tensor_ub_size, id_number)

        offset = self.tik_instance.Scalar(dtype='int32', name='offset')
        with self.tik_instance.for_range(0, (end - begin)) as i:
            with self.tik_instance.for_range(0, self.last_dim_x) as j:
                self.index_scalar.set_as(self.x_ub[i * self.last_dim_x + j])
                with self.tik_instance.if_scope(self.last_dim_x // self.off_value_each_block > 0):
                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                self.y_gm[id_number * self.not_last_core_numel * self.last_dim_x *
                                                          self.depth + i * self.depth * self.last_dim_x
                                                          + self.index_scalar
                                                          * self.last_dim_x],
                                                0, 1,
                                                (self.last_dim_x - 1) // self.off_value_each_block + 1, 0, 0)
                    self.off_value_tensor_ub[j].set_as(self.on_value)
                    self.tik_instance.data_move(self.y_gm[id_number *
                                                          self.not_last_core_numel *
                                                          self.last_dim_x *
                                                          self.depth +
                                                          i *
                                                          self.depth *
                                                          self.last_dim_x +
                                                          self.index_scalar *
                                                          self.last_dim_x], self.off_value_tensor_ub, 0, 1,
                                                          self.last_dim_x // self.off_value_each_block, 0, 0)
                    self.align_to_32_last_block(
                        self.last_dim_x,
                        self.not_last_core_numel *
                        (
                            self.last_dim_x -
                            1) *
                        self.depth *
                        id_number +
                        i *
                        self.depth *
                        self.last_dim_x +
                        self.index_scalar *
                        self.last_dim_x,
                        id_number)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope((i * self.depth * self.last_dim_x + self.index_scalar
                                                     * self.last_dim_x) > ((end - begin) * self.last_dim_x *
                                                                           self.depth - self.off_value_each_block)):
                        offset.set_as(self.off_value_each_block -
                                      ((end -
                                        begin) *
                                       self.last_dim_x *
                                       self.depth -
                                       i *
                                       self.last_dim_x *
                                       self.depth -
                                       self.index_scalar *
                                       self.last_dim_x))
                        self.tik_instance.data_move(self.off_value_tensor_ub,
                                                    self.y_gm[id_number * self.not_last_core_numel * self.last_dim_x *
                                                              self.depth + i * self.depth * self.last_dim_x
                                                              + self.index_scalar
                                                              * self.last_dim_x - offset],
                                                    0, 1, 1, 0, 0)
                        self.off_value_tensor_ub[j +
                                                 offset].set_as(self.on_value)
                        self.tik_instance.data_move(self.y_gm[id_number *
                                                              self.not_last_core_numel *
                                                              self.last_dim_x *
                                                              self.depth +
                                                              i *
                                                              self.depth *
                                                              self.last_dim_x +
                                                              self.index_scalar *
                                                              self.last_dim_x -
                                                              offset], self.off_value_tensor_ub, 0, 1, 1, 0, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.data_move(self.off_value_tensor_ub,
                                                    self.y_gm[id_number * self.not_last_core_numel * self.last_dim_x *
                                                              self.depth + i * self.depth * self.last_dim_x
                                                              + self.index_scalar
                                                              * self.last_dim_x],
                                                    0, 1, 1, 0, 0)
                        self.off_value_tensor_ub[j].set_as(self.on_value)
                        self.tik_instance.data_move(self.y_gm[id_number *
                                                              self.not_last_core_numel *
                                                              self.last_dim_x *
                                                              self.depth +
                                                              i *
                                                              self.depth *
                                                              self.last_dim_x +
                                                              self.index_scalar *
                                                              self.last_dim_x], self.off_value_tensor_ub, 0, 1, 1, 0, 0)

    # middle axis with ub size is less than x and enough to off_value_tensor
    # some lines
    # 'pylint: disable =too-many-locals,too-many-statements
    def one_hot_middle_axis_third_mode(self, id_number):
        """
        the third calculate mode when the axis is 0 < axis < len(x_shape) - 1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_numel * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_numel + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_numel + begin)

        off_value_tensor_ub_size = self.tik_instance.Scalar(
            dtype='int32', name='off_value_tensor_ub_size')
        off_value_tensor_ub_size.set_as(
            self.per_part_unused_ub *
            self.off_value_tensor_part)
        self.vec_dump_off_value_tensor_ub(off_value_tensor_ub_size)
        move_times = self.tik_instance.Scalar(dtype='int32', name='move_times')
        move_times.set_as((self.depth * (end - begin) *
                           self.last_dim_x) // off_value_tensor_ub_size)
        with self.tik_instance.if_scope(move_times > 0):
            with self.tik_instance.for_range(0, move_times) as k:
                self.tik_instance.data_move(
                    self.y_gm[begin * self.depth * self.last_dim_x + k * off_value_tensor_ub_size],
                    self.off_value_tensor_ub, 0, 1,
                    off_value_tensor_ub_size // self.off_value_each_block, 0, 0)
        with self.tik_instance.if_scope(self.depth * self.last_dim_x * (
                end - begin) % off_value_tensor_ub_size // self.off_value_each_block > 0):
            self.tik_instance.data_move(
                self.y_gm[begin * self.depth * self.last_dim_x + move_times * off_value_tensor_ub_size],
                self.off_value_tensor_ub, 0, 1,
                self.last_dim_x * self.depth * (end - begin) % off_value_tensor_ub_size //
                self.off_value_each_block, 0, 0)
            self.align_to_32_last_block(self.last_dim_x *
                                        self.depth *
                                        (end -
                                         begin) %
                                        off_value_tensor_ub_size, self.not_last_core_numel *
                                        (self.last_dim_x -
                                         1) *
                                        self.depth *
                                        id_number +
                                        move_times *
                                        off_value_tensor_ub_size, id_number)

        x_move_times = self.tik_instance.Scalar(
            dtype='int32', name='x_move_times')
        offset = self.tik_instance.Scalar(dtype='int32', name='offset')

        with self.tik_instance.if_scope(self.per_part_unused_ub // self.last_dim_x > 0):
            x_ub_size_align = self.tik_instance.Scalar(
                dtype='int32', name='x_ub_size_align')
            x_ub_size_align.set_as(
                self.per_part_unused_ub // self.last_dim_x * self.last_dim_x)
            x_move_times.set_as(
                ((end - begin) * self.last_dim_x - 1) // x_ub_size_align + 1)

            with self.tik_instance.for_range(0, x_move_times) as i:
                self.tik_instance.data_move(self.x_ub,
                                            self.x_gm[begin * self.last_dim_x + x_ub_size_align * i],
                                            0,
                                            1,
                                            x_ub_size_align // self.x_each_block,
                                            0,
                                            0)
                with self.tik_instance.if_scope(x_ub_size_align % self.x_each_block > 0):
                    offset_begin = self.tik_instance.Scalar(
                        dtype='int32', name='offset_begin')
                    offset_begin.set_as(x_ub_size_align //
                                        self.x_each_block *
                                        self.x_each_block -
                                        (self.x_each_block -
                                         x_ub_size_align %
                                         self.x_each_block))
                    block_ub = self.tik_instance.Tensor(
                        self.dtype_x, (self.x_each_block,), name='block_ub', scope=tik.scope_ubuf)
                    self.tik_instance.data_move(block_ub, self.x_gm[id_number *
                                                                    self.not_last_core_numel *
                                                                    self.last_dim_x +
                                                                    x_ub_size_align *
                                                                    i +
                                                                    offset_begin], 0, 1, 1, 0, 0)
                    tmp = self.tik_instance.Scalar(dtype='int32', name='tmp')
                    with self.tik_instance.for_range(offset_begin, x_ub_size_align) as ind:
                        tmp.set_as(block_ub[ind - offset_begin])
                        self.x_ub[ind].set_as(tmp)

                with self.tik_instance.if_scope(
                        tik.all(i == x_move_times - 1, (end - begin) * self.last_dim_x % x_ub_size_align > 0)):
                    with self.tik_instance.for_range(0, (end - begin) * self.last_dim_x % x_ub_size_align
                                                     // self.last_dim_x) as j:
                        with self.tik_instance.for_range(0, self.last_dim_x) as k:
                            self.index_scalar.set_as(
                                self.x_ub[j * self.last_dim_x + k])
                            with self.tik_instance.if_scope(self.last_dim_x // self.off_value_each_block > 0):
                                self.tik_instance.data_move(self.off_value_tensor_ub,
                                                            self.y_gm[id_number * self.not_last_core_numel *
                                                                      self.last_dim_x *
                                                                      self.depth + self.depth * x_ub_size_align * i + j
                                                                      * self.last_dim_x * self.depth +
                                                                      self.index_scalar * self.last_dim_x], 0, 1,
                                                            (self.last_dim_x - 1) // self.off_value_each_block + 1,
                                                            0, 0)
                                self.off_value_tensor_ub[k].set_as(
                                    self.on_value)
                                self.tik_instance.data_move(self.y_gm[id_number *
                                                                      self.not_last_core_numel *
                                                                      self.last_dim_x *
                                                                      self.depth +
                                                                      self.depth *
                                                                      x_ub_size_align *
                                                                      i +
                                                                      j *
                                                                      self.last_dim_x *
                                                                      self.depth +
                                                                      self.index_scalar *
                                                                      self.last_dim_x], self.off_value_tensor_ub, 0, 1,
                                                            self.last_dim_x // self.off_value_each_block, 0, 0)
                                self.align_to_32_last_block(
                                    self.last_dim_x,
                                    self.not_last_core_numel *
                                    (
                                        self.last_dim_x -
                                        1) *
                                    self.depth *
                                    id_number +
                                    self.depth *
                                    x_ub_size_align *
                                    i +
                                    j *
                                    self.last_dim_x *
                                    self.depth +
                                    self.index_scalar *
                                    self.last_dim_x,
                                    id_number)

                            with self.tik_instance.else_scope():
                                with self.tik_instance.if_scope((self.depth * x_ub_size_align * i + j
                                                                 * self.last_dim_x * self.depth +
                                                                 self.index_scalar * self.last_dim_x) >
                                                                ((end - begin) * self.last_dim_x *
                                                                 self.depth - self.off_value_each_block)):
                                    offset.set_as(
                                        self.off_value_each_block - ((end - begin) * self.last_dim_x * self.depth -
                                                                     self.depth * x_ub_size_align * i - j
                                                                     * self.last_dim_x * self.depth -
                                                                     self.index_scalar * self.last_dim_x))
                                    self.tik_instance.data_move(
                                        self.off_value_tensor_ub,
                                        self.y_gm[
                                            id_number *
                                            self.not_last_core_numel *
                                            self.last_dim_x *
                                            self.depth +
                                            self.depth *
                                            x_ub_size_align *
                                            i +
                                            j *
                                            self.last_dim_x *
                                            self.depth +
                                            self.index_scalar *
                                            self.last_dim_x -
                                            offset],
                                        0,
                                        1,
                                        1,
                                        0,
                                        0)
                                    self.off_value_tensor_ub[k +
                                                             offset].set_as(self.on_value)
                                    self.tik_instance.data_move(
                                        self.y_gm[id_number * self.not_last_core_numel *
                                                  self.last_dim_x *
                                                  self.depth + self.depth * x_ub_size_align * i + j
                                                  * self.last_dim_x * self.depth +
                                                  self.index_scalar * self.last_dim_x - offset],
                                        self.off_value_tensor_ub, 0, 1, 1, 0, 0)
                                with self.tik_instance.else_scope():
                                    self.tik_instance.data_move(
                                        self.off_value_tensor_ub,
                                        self.y_gm[
                                            id_number *
                                            self.not_last_core_numel *
                                            self.last_dim_x *
                                            self.depth +
                                            self.depth *
                                            x_ub_size_align *
                                            i +
                                            j *
                                            self.last_dim_x *
                                            self.depth +
                                            self.index_scalar *
                                            self.last_dim_x],
                                        0,
                                        1,
                                        1,
                                        0,
                                        0)
                                    self.off_value_tensor_ub[k].set_as(
                                        self.on_value)
                                    self.tik_instance.data_move(
                                        self.y_gm[id_number * self.not_last_core_numel *
                                                  self.last_dim_x *
                                                  self.depth + self.depth * x_ub_size_align * i + j
                                                  * self.last_dim_x * self.depth +
                                                  self.index_scalar * self.last_dim_x],
                                        self.off_value_tensor_ub, 0, 1, 1, 0, 0)

                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, x_ub_size_align // self.last_dim_x) as j:
                        with self.tik_instance.for_range(0, self.last_dim_x) as k:
                            self.index_scalar.set_as(
                                self.x_ub[j * self.last_dim_x + k])
                            with self.tik_instance.if_scope(self.last_dim_x // self.off_value_each_block > 0):
                                self.tik_instance.data_move(self.off_value_tensor_ub,
                                                            self.y_gm[id_number * self.not_last_core_numel *
                                                                      self.last_dim_x *
                                                                      self.depth + self.depth * x_ub_size_align * i + j
                                                                      * self.last_dim_x * self.depth +
                                                                      self.index_scalar * self.last_dim_x], 0, 1,
                                                            (self.last_dim_x - 1) // self.off_value_each_block + 1,
                                                            0, 0)
                                self.off_value_tensor_ub[k].set_as(
                                    self.on_value)
                                self.tik_instance.data_move(self.y_gm[id_number *
                                                                      self.not_last_core_numel *
                                                                      self.last_dim_x *
                                                                      self.depth +
                                                                      self.depth *
                                                                      x_ub_size_align *
                                                                      i +
                                                                      j *
                                                                      self.last_dim_x *
                                                                      self.depth +
                                                                      self.index_scalar *
                                                                      self.last_dim_x], self.off_value_tensor_ub, 0, 1,
                                                            self.last_dim_x // self.off_value_each_block, 0, 0)
                                self.align_to_32_last_block(
                                    self.last_dim_x,
                                    self.not_last_core_numel *
                                    (
                                        self.last_dim_x -
                                        1) *
                                    self.depth *
                                    id_number +
                                    self.depth *
                                    x_ub_size_align *
                                    i +
                                    j *
                                    self.last_dim_x *
                                    self.depth +
                                    self.index_scalar *
                                    self.last_dim_x,
                                    id_number)

                            with self.tik_instance.else_scope():
                                with self.tik_instance.if_scope((self.depth * x_ub_size_align * i + j
                                                                 * self.last_dim_x * self.depth +
                                                                 self.index_scalar * self.last_dim_x) >
                                                                ((end - begin) * self.last_dim_x *
                                                                 self.depth - self.off_value_each_block)):
                                    offset.set_as(
                                        self.off_value_each_block - ((end - begin) * self.last_dim_x * self.depth -
                                                                     self.depth * x_ub_size_align * i - j
                                                                     * self.last_dim_x * self.depth -
                                                                     self.index_scalar * self.last_dim_x))
                                    self.tik_instance.data_move(
                                        self.off_value_tensor_ub,
                                        self.y_gm[
                                            id_number *
                                            self.not_last_core_numel *
                                            self.last_dim_x *
                                            self.depth +
                                            self.depth *
                                            x_ub_size_align *
                                            i +
                                            j *
                                            self.last_dim_x *
                                            self.depth +
                                            self.index_scalar *
                                            self.last_dim_x -
                                            offset],
                                        0,
                                        1,
                                        1,
                                        0,
                                        0)
                                    self.off_value_tensor_ub[k +
                                                             offset].set_as(self.on_value)
                                    self.tik_instance.data_move(
                                        self.y_gm[id_number * self.not_last_core_numel *
                                                  self.last_dim_x *
                                                  self.depth + self.depth * x_ub_size_align * i + j
                                                  * self.last_dim_x * self.depth +
                                                  self.index_scalar * self.last_dim_x - offset],
                                        self.off_value_tensor_ub, 0, 1, 1, 0, 0)

                                with self.tik_instance.else_scope():
                                    self.tik_instance.data_move(
                                        self.off_value_tensor_ub,
                                        self.y_gm[
                                            id_number *
                                            self.not_last_core_numel *
                                            self.last_dim_x *
                                            self.depth +
                                            self.depth *
                                            x_ub_size_align *
                                            i +
                                            j *
                                            self.last_dim_x *
                                            self.depth +
                                            self.index_scalar *
                                            self.last_dim_x],
                                        0,
                                        1,
                                        1,
                                        0,
                                        0)
                                    self.off_value_tensor_ub[k].set_as(
                                        self.on_value)
                                    self.tik_instance.data_move(
                                        self.y_gm[id_number * self.not_last_core_numel *
                                                  self.last_dim_x *
                                                  self.depth + self.depth * x_ub_size_align * i + j
                                                  * self.last_dim_x * self.depth +
                                                  self.index_scalar * self.last_dim_x],
                                        self.off_value_tensor_ub, 0, 1, 1, 0, 0)

        with self.tik_instance.else_scope():
            self.one_hot_middle_axis_third_mode_back(id_number, end, begin)

    # middle axis with ub size is less than x and enough to off_value_tensor
    # some lines
    def one_hot_middle_axis_third_mode_back(self, id_number, end, begin):
        """
        the back part of third calculate mode when the axis is 0 < axis < len(x_shape) - 1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now
        end:int
        the end index of cal
        begin:int
        the begin index of cal

        Returns
        -------
        None
        """
        x_last_dim_move_times = self.tik_instance.Scalar(
            dtype='int32', name='x_last_dim_move_times')
        x_last_dim_move_times.set_as(
            (self.last_dim_x - 1) // self.per_part_unused_ub + 1)

        with self.tik_instance.for_range(0, end - begin) as i:
            with self.tik_instance.for_range(0, x_last_dim_move_times) as j:
                self.tik_instance.data_move(self.x_ub, self.x_gm[id_number *
                                                                 (end -
                                                                  begin) *
                                                                 self.last_dim_x +
                                                                 i *
                                                                 self.last_dim_x +
                                                                 self.per_part_unused_ub *
                                                                 j], 0, 1, self.per_part_unused_ub //
                                            self.x_each_block, 0, 0)
                with self.tik_instance.if_scope(
                        tik.all(j == x_last_dim_move_times - 1, self.last_dim_x % self.per_part_unused_ub > 0)):
                    with self.tik_instance.for_range(self.last_dim_x // self.per_part_unused_ub *
                                                     self.per_part_unused_ub, self.last_dim_x) as index:
                        self.index_scalar.set_as(self.x_ub[index - self.last_dim_x // self.per_part_unused_ub *
                                                           self.per_part_unused_ub])
                        self.tik_instance.data_move(self.off_value_tensor_ub,
                                                    self.y_gm[id_number * self.not_last_core_numel *
                                                              self.last_dim_x *
                                                              self.depth + self.depth * self.last_dim_x * i +
                                                              self.index_scalar * self.last_dim_x], 0, 1,
                                                    (self.last_dim_x - 1) // self.off_value_each_block + 1, 0, 0)
                        self.off_value_tensor_ub[index].set_as(self.on_value)
                        self.tik_instance.data_move(self.y_gm[id_number *
                                                              self.not_last_core_numel *
                                                              self.last_dim_x *
                                                              self.depth +
                                                              self.depth *
                                                              self.last_dim_x *
                                                              i +
                                                              self.index_scalar *
                                                              self.last_dim_x], self.off_value_tensor_ub, 0, 1,
                                                    self.last_dim_x // self.off_value_each_block, 0, 0)
                        self.align_to_32_last_block(
                            self.last_dim_x,
                            self.not_last_core_numel *
                            (
                                self.last_dim_x -
                                1) *
                            self.depth *
                            id_number +
                            self.depth *
                            self.last_dim_x *
                            i +
                            self.index_scalar *
                            self.last_dim_x,
                            id_number)

                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.per_part_unused_ub) as k:
                        self.index_scalar.set_as(self.x_ub[k])
                        self.tik_instance.data_move(self.off_value_tensor_ub,
                                                    self.y_gm[id_number * self.not_last_core_numel *
                                                              self.last_dim_x *
                                                              self.depth + self.depth * self.last_dim_x * i +
                                                              self.index_scalar * self.last_dim_x], 0, 1,
                                                    (self.last_dim_x - 1) // self.off_value_each_block + 1, 0, 0)
                        self.off_value_tensor_ub[k + j *
                                                 self.per_part_unused_ub].set_as(self.on_value)
                        self.tik_instance.data_move(self.y_gm[id_number *
                                                              self.not_last_core_numel *
                                                              self.last_dim_x *
                                                              self.depth +
                                                              self.depth *
                                                              self.last_dim_x *
                                                              i +
                                                              self.index_scalar *
                                                              self.last_dim_x], self.off_value_tensor_ub, 0, 1,
                                                    self.last_dim_x // self.off_value_each_block, 0, 0)
                        self.align_to_32_last_block(
                            self.last_dim_x,
                            self.not_last_core_numel *
                            (
                                self.last_dim_x -
                                1) *
                            self.depth *
                            id_number +
                            self.depth *
                            self.last_dim_x *
                            i +
                            self.index_scalar *
                            self.last_dim_x,
                            id_number)

    # middle axis with ub size is less than x smaller than off_value_tensor
    # one line
    # 'pylint: disable =too-many-locals,too-many-statements
    def one_hot_middle_axis_fourth_mode(self, id_number):
        """
        the fourth calculate mode when the axis is 0 < axis < len(x_shape) - 1

        Parameters
        ----------
        id_number:int
        the index of the ai core used now

        Returns
        -------
        None
        """
        begin = self.tik_instance.Scalar(dtype='int32', name='begin')
        end = self.tik_instance.Scalar(dtype='int32', name='end')
        begin.set_as(self.not_last_core_numel * id_number)
        with self.tik_instance.if_scope(id_number == self.core_number - 1):
            end.set_as(self.last_core_numel + begin)
        with self.tik_instance.else_scope():
            end.set_as(self.not_last_core_numel + begin)

        index_fill_on_value = self.tik_instance.Scalar(
            dtype='int32', name='index_fill_on_value')
        part_num = self.tik_instance.Scalar(dtype='int32', name='part_num')
        part_num.set_as(self.per_part_unused_ub * self.off_value_tensor_part)
        x_move_times_last_dim = self.tik_instance.Scalar(
            dtype='int32', name='x_move_times_last_dim')
        off_value_tensor_ub_size = self.tik_instance.Scalar(
            dtype='int32', name='off_value_tensor_ub_size')
        off_value_tensor_ub_size.set_as(
            self.per_part_unused_ub *
            self.off_value_tensor_part)
        x_move_times_last_dim.set_as(
            (self.last_dim_x - 1) // self.per_part_unused_ub + 1)
        self.vec_dump_off_value_tensor_ub(part_num)
        offset = self.tik_instance.Scalar(dtype='int32', name='offset')
        burst_len = self.tik_instance.Scalar(dtype='int32', name='burst_len')

        move_times = self.tik_instance.Scalar(dtype='int32', name='move_times')
        move_times.set_as((self.depth * (end - begin) *
                           self.last_dim_x) // off_value_tensor_ub_size)
        with self.tik_instance.for_range(0, move_times) as k:
            self.tik_instance.data_move(self.y_gm[begin *
                                                  self.depth *
                                                  self.last_dim_x +
                                                  k *
                                                  off_value_tensor_ub_size], self.off_value_tensor_ub, 0, 1,
                                        off_value_tensor_ub_size //
                                        self.off_value_each_block, 0, 0)
        with self.tik_instance.if_scope(self.depth * (end - begin) * self.last_dim_x % off_value_tensor_ub_size > 0):
            self.tik_instance.data_move(
                self.y_gm[begin * self.depth * self.last_dim_x + move_times *
                          off_value_tensor_ub_size],
                self.off_value_tensor_ub, 0, 1,
                (self.depth * (end - begin) * self.last_dim_x % off_value_tensor_ub_size) //
                self.off_value_each_block, 0, 0)
            self.align_to_32_last_block(self.depth *
                                        (end -
                                         begin) *
                                        self.last_dim_x %
                                        off_value_tensor_ub_size, begin *
                                        self.depth *
                                        (self.last_dim_x -
                                         1) +
                                        move_times *
                                        off_value_tensor_ub_size, id_number)

        with self.tik_instance.for_range(0, (end - begin)) as i:
            with self.tik_instance.for_range(0, x_move_times_last_dim) as index:
                self.tik_instance.data_move(self.x_ub, self.x_gm[id_number *
                                                                 self.not_last_core_numel *
                                                                 self.last_dim_x +
                                                                 i *
                                                                 self.last_dim_x +
                                                                 index *
                                                                 self.per_part_unused_ub], 0, 1,
                                            self.per_part_unused_ub //
                                            self.x_each_block, 0, 0)
                with self.tik_instance.if_scope(
                        tik.all(index == x_move_times_last_dim - 1,
                                self.last_dim_x % self.per_part_unused_ub > 0)):
                    with self.tik_instance.for_range(0, self.last_dim_x % self.per_part_unused_ub) as k:
                        self.index_scalar.set_as(self.x_ub[k])
                        with self.tik_instance.for_range(0, Constant.cal_num) as j:
                            with self.tik_instance.if_scope(
                                    tik.all((k + (self.last_dim_x // self.per_part_unused_ub * self.per_part_unused_ub))
                                            < part_num * (j + 1),
                                            (k + (self.last_dim_x // self.per_part_unused_ub *
                                                  self.per_part_unused_ub)) >= part_num * j)):
                                with self.tik_instance.if_scope(part_num * (j + 1) > self.last_dim_x):
                                    offset.set_as(((self.last_dim_x -
                                                    part_num *
                                                    j -
                                                    1) //
                                                   self.off_value_each_block +
                                                   1) *
                                                  self.off_value_each_block -
                                                  (self.last_dim_x -
                                                   part_num *
                                                   j))
                                    burst_len.set_as((offset + (
                                        self.last_dim_x - part_num * j)) // self.off_value_each_block)
                                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                                self.y_gm[id_number * self.not_last_core_numel
                                                                          * self.last_dim_x * self.depth +
                                                                          i * self.depth * self.last_dim_x +
                                                                          self.index_scalar * self.last_dim_x + j
                                                                          * part_num - offset], 0, 1,
                                                                burst_len, 0, 0)
                                    index_fill_on_value.set_as(k +
                                                               self.last_dim_x //
                                                               self.per_part_unused_ub *
                                                               self.per_part_unused_ub -
                                                               (j *
                                                                part_num) +
                                                               offset)
                                    self.off_value_tensor_ub[index_fill_on_value].set_as(
                                        self.on_value)
                                    self.tik_instance.data_move(self.y_gm[id_number *
                                                                          self.not_last_core_numel *
                                                                          self.last_dim_x *
                                                                          self.depth +
                                                                          self.index_scalar *
                                                                          self.last_dim_x +
                                                                          i *
                                                                          self.depth *
                                                                          self.last_dim_x +
                                                                          j *
                                                                          part_num -
                                                                          offset], self.off_value_tensor_ub, 0, 1,
                                                                burst_len, 0, 0)
                                    self.align_to_32_last_block(offset +
                                                                (self.last_dim_x -
                                                                 part_num *
                                                                 j), self.not_last_core_numel *
                                                                (self.last_dim_x -
                                                                 1) *
                                                                self.depth *
                                                                id_number +
                                                                self.depth *
                                                                self.last_dim_x *
                                                                i +
                                                                j *
                                                                part_num +
                                                                self.index_scalar *
                                                                self.last_dim_x, id_number)

                                with self.tik_instance.else_scope():
                                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                                self.y_gm[id_number * self.not_last_core_numel
                                                                          * self.last_dim_x * self.depth +
                                                                          self.index_scalar * self.last_dim_x +
                                                                          i * self.depth * self.last_dim_x + j
                                                                          * part_num], 0, 1,
                                                                part_num // self.off_value_each_block, 0, 0)
                                    index_fill_on_value.set_as(k + self.last_dim_x // self.per_part_unused_ub *
                                                               self.per_part_unused_ub - (j * part_num))
                                    self.off_value_tensor_ub[self.index_scalar].set_as(
                                        self.on_value)
                                    self.tik_instance.data_move(
                                        self.y_gm[id_number * self.not_last_core_numel
                                                  * self.last_dim_x * self.depth +
                                                  self.index_scalar * self.last_dim_x +
                                                  i * self.depth * self.last_dim_x + j
                                                  * part_num], self.off_value_tensor_ub,
                                        0, 1, part_num // self.off_value_each_block, 0, 0)
                                    self.align_to_32_last_block(
                                        part_num,
                                        self.not_last_core_numel *
                                        (
                                            self.last_dim_x -
                                            1) *
                                        self.depth *
                                        id_number +
                                        self.depth *
                                        self.last_dim_x *
                                        i +
                                        j *
                                        part_num +
                                        self.index_scalar *
                                        self.last_dim_x,
                                        id_number)

                with self.tik_instance.else_scope():
                    with self.tik_instance.for_range(0, self.per_part_unused_ub) as cnt:
                        self.index_scalar.set_as(self.x_ub[cnt])
                        with self.tik_instance.for_range(0, Constant.cal_num) as j:
                            with self.tik_instance.if_scope(
                                    tik.all((cnt + index * self.per_part_unused_ub) < part_num * (j + 1),
                                            (cnt + index * self.per_part_unused_ub) >= part_num * j)):
                                with self.tik_instance.if_scope(part_num * (j + 1) > self.last_dim_x):
                                    offset.set_as(((self.last_dim_x -
                                                    part_num *
                                                    j -
                                                    1) //
                                                   self.off_value_each_block +
                                                   1) *
                                                  self.off_value_each_block -
                                                  (self.last_dim_x -
                                                   part_num *
                                                   j))
                                    burst_len.set_as((offset + (self.last_dim_x - part_num * j))
                                                     // self.off_value_each_block)
                                    self.tik_instance.data_move(self.off_value_tensor_ub,
                                                                self.y_gm[id_number * self.not_last_core_numel
                                                                          * self.last_dim_x * self.depth +
                                                                          self.index_scalar * self.last_dim_x +
                                                                          i * self.depth * self.last_dim_x + j
                                                                          * part_num - offset], 0, 1,
                                                                burst_len, 0, 0)
                                    index_fill_on_value.set_as(
                                        (cnt + index * self.per_part_unused_ub) - (j * part_num) + offset)
                                    self.off_value_tensor_ub[index_fill_on_value].set_as(
                                        self.on_value)
                                    self.tik_instance.data_move(
                                        self.y_gm[id_number * self.not_last_core_numel
                                                  * self.last_dim_x * self.depth +
                                                  self.index_scalar * self.last_dim_x +
                                                  i * self.depth * self.last_dim_x + j
                                                  * part_num - offset], self.off_value_tensor_ub,
                                        0, 1, burst_len, 0, 0)
                                    self.align_to_32_last_block(offset +
                                                                (self.last_dim_x -
                                                                 part_num *
                                                                 j), self.not_last_core_numel *
                                                                (self.last_dim_x -
                                                                 1) *
                                                                self.depth *
                                                                id_number +
                                                                self.depth *
                                                                self.last_dim_x *
                                                                i +
                                                                j *
                                                                part_num +
                                                                self.index_scalar *
                                                                self.last_dim_x -
                                                                offset, id_number)

                                with self.tik_instance.else_scope():
                                    self.tik_instance.data_move(
                                        self.off_value_tensor_ub,
                                        self.y_gm[
                                            id_number *
                                            self.not_last_core_numel *
                                            self.last_dim_x *
                                            self.depth +
                                            self.index_scalar *
                                            self.last_dim_x +
                                            i *
                                            self.depth *
                                            self.last_dim_x +
                                            j *
                                            part_num],
                                        0,
                                        1,
                                        part_num //
                                        self.off_value_each_block,
                                        0,
                                        0)
                                    index_fill_on_value.set_as(
                                        (cnt + index * self.per_part_unused_ub) - (j * part_num))
                                    self.off_value_tensor_ub[index_fill_on_value].set_as(
                                        self.on_value)
                                    self.tik_instance.data_move(
                                        self.y_gm[id_number * self.not_last_core_numel
                                                  * self.last_dim_x * self.depth +
                                                  self.index_scalar * self.last_dim_x +
                                                  i * self.depth * self.last_dim_x + j
                                                  * part_num], self.off_value_tensor_ub,
                                        0, 1, part_num // self.off_value_each_block, 0, 0)
                                    self.align_to_32_last_block(
                                        part_num,
                                        self.not_last_core_numel *
                                        (
                                            self.last_dim_x -
                                            1) *
                                        self.depth *
                                        id_number +
                                        self.depth *
                                        self.last_dim_x *
                                        i +
                                        j *
                                        part_num +
                                        self.index_scalar *
                                        self.last_dim_x,
                                        id_number)


def check_supported(x, depth, on_value, off_value, y, axis,
                    kernel_name="one_hot"):
    """
    dynamic is support, static and shape[0] is 2048, and axis is 0,
    onehot is support, else static not support, onehotd is support.
    x : dict
        dict with keys(range and dtype) of indices tensor
    depth: dict
        dict whith the scalar depth
    on_value : dict
        dict with the scalar on_value
    off_value: dict
        dict with the scalar off_value
    axis: int
        the axis which judge the mode of calculate
    y : dict
        dict with keys(range and dtype) of output
    kernel_name : str
        kernel name, default value is "one_hot"

    Returns
    -------
    True or False
    """
    x_shape = x.get("ori_shape")
    x_dtype = x.get("dtype").lower()
    depth_dtype = depth.get("dtype").lower()
    on_value_dtype = on_value.get("dtype").lower()
    off_value_dtype = off_value.get("dtype").lower()

    if x_dtype != "int32" or depth_dtype != "int32":
        reason = "x and y dtype is not int32, but is %s" % x_dtype
        return False, reason
    if on_value_dtype != off_value_dtype:
        reason = "on_value dtype is not the same as off_value dtype"
        return False, reason
    if on_value_dtype not in ("float16", "float32", "int32"):
        reason = "on_value not in (\"float16\", \"float32\", \"int32\"), but is %s" % on_value_dtype
        return False, reason
    # when static and x shape[0] is 2048 and axis is 0, one_hot is support
    shape_list = [(2048,)]
    if util_common.is_unknown([x, y]):
        return True, ""
    if x_shape in shape_list and axis == 0:
        reason = "when static and shape is 2048 and axis is 0"
        return True, reason
    # when static and the input0_shape ends wtih 1, the compilestatic process dose not support
    reason = "when static, x_shape[0] is not 2048 or axis is not 0, one_hot not support"
    return False, reason


def _check_param(x, depth, on_value, off_value):
    """
    check parameters, if one is invalid, then raise error

    Parameters
    ----------
    x : dict
        dict with keys(range and dtype) of indices tensor
    depth: dict
        dict whith the scalar depth
    on_value : dict
        dict with the scalar on_value
    off_value: dict
        dict with the scalar off_value
    axis: int
        the axis which judge the mode of calculate

    Returns
    -------
    None
    """
    x_dtype = x.get("dtype").lower()
    depth_dtype = depth.get("dtype").lower()
    on_value_dtype = on_value.get("dtype").lower()
    off_value_dtype = off_value.get("dtype").lower()
    para_check.check_dtype(x_dtype, ["int32",])
    para_check.check_dtype(depth_dtype, ["int32",])
    para_check.check_dtype(on_value_dtype, ["int32", "float32", "float16"])
    para_check.check_dtype(off_value_dtype, ["int32", "float32", "float16"])


# the register of OneHot op
# 'pylint: disable=unused-argument,too-many-arguments
@register_operator('OneHot')
def one_hot(x,
            depth,
            on_value,
            off_value,
            y,
            axis,
            kernel_name='one_hot'):
    """
    algorithm:one_hot
    Operation for one_hot

    Parameters
    ----------
    x : dict
        dict with keys(range and dtype) of indices tensor
    depth: dict
        dict whith the scalar depth
    on_value : dict
        dict with the scalar on_value
    off_value: dict
        dict with the scalar off_value
    axis: int
        the axis which judge the mode of calculate
    y : dict
        dict with keys(range and dtype) of output
    kernel_name : str
        kernel name, default value is "resize_bilinear_v2_grad"

    Returns
    -------
    None
    """
    _check_param(x, depth, on_value, off_value)
    one_hot_instance = OneHot(
        x, depth, on_value, off_value, axis, y, kernel_name)
    tik_instance = one_hot_instance.one_hot_compute_tiling()
    return tik_instance
