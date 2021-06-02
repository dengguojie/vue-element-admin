"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

as_strided
"""

from impl.util import util_common
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
RESERVED_UB = 4  # 4KB
MAX_INT64_VALUE = 2 ** 64 - 1
TILING_MAX_SIZE_GM = 2048  # 16KB

class AsStrided(object):
    """
    AsStrided
    """
    def __init__(self, tik_inst, x_dtype, tensor_list):
        data_in, size, stride, storage_offset, data_out, data_tiling = tensor_list
        self.tik_inst = tik_inst
        self.data_in = data_in
        self.data_out = data_out
        self.ub_size_b8 = self._get_ub_size_by_b8()
        self.ub_pattern = self.tik_inst.Tensor("uint32", (128,), tik.scope_ubuf, "ub_pattern")
        self.ub_input_b8 = self.tik_inst.Tensor("int8", (self.ub_size_b8,), tik.scope_ubuf, "ub_input_b8")
        self.ub_input = self.ub_input_b8.reinterpret_cast_to(x_dtype)

    def _get_ub_size_by_b8(self):
        return (UB_SIZE - RESERVED_UB * 1024)

    def compute_1(self):
        """
        compute_1 (100,240) -> (100, 24)
        """
        with self.tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
            # ---counter_mode---
            with self.tik_inst.if_scope(block_idx == 0):
                scalar_value = self.tik_inst.Scalar("int32")
                scalar_value.set_as(16843009)
                self.ub_pattern[0].set_as(scalar_value)
                self.ub_pattern[1].set_as(scalar_value)
                self.ub_pattern[2].set_as(scalar_value)
                self.ub_pattern[3].set_as(scalar_value)
                self.ub_pattern[4].set_as(scalar_value)
                self.ub_pattern[5].set_as(scalar_value)
                with self.tik_inst.for_range(0, 2) as k:
                    with self.tik_inst.for_range(0, 100) as i:
                        self.tik_inst.data_move_pad(self.ub_input, self.data_in[i * 240], 24, 4, 0, 4 * 9)
                        self.tik_inst.vreducev2(24 * 8,                     # mask
                                                self.ub_input[1024 * 10],   # dst
                                                self.ub_input,              # src0
                                                self.ub_pattern,            # src1_pattern
                                                1,                          # repeat_time
                                                1,                          # src0_blk_stride
                                                0,                          # src0_rep_stride
                                                0,                          # src1_rep_stride
                                                None,                       # rsvd_scalar
                                                "counter")                  # mask_mode
                        self.tik_inst.data_move(self.data_out[i * 24], self.ub_input[1024 * 10], 0, 1, 3, 0, 0)

        # ---normal_mode---
        #with self.tik_inst.if_scope(block_idx == 0):
        #    scalar_value = self.tik_inst.Scalar("uint32")
        #    zero_value = self.tik_inst.Scalar("uint32")
        #    scalar_value.set_as(2155905152)
        #    zero_value.set_as(0)
        #    self.ub_pattern[0].set_as(scalar_value)
        #    self.ub_pattern[1].set_as(scalar_value)
        #    self.ub_pattern[2].set_as(scalar_value)
        #    self.ub_pattern[3].set_as(scalar_value)
        #    with self.tik_inst.for_range(0, 100) as i:
        #        self.tik_inst.data_move_pad(self.ub_input, self.data_in[i * 240], 24, 4, 0, 4 * 9)
        #        self.tik_inst.vreducev2(None,                       # mask
        #                                self.ub_input[1024 * 10],   # dst
        #                                self.ub_input,              # src0
        #                                self.ub_pattern,            # src1_pattern
        #                                3,                          # repeat_time
        #                                1,                          # src0_blk_stride
        #                                8,                          # src0_rep_stride
        #                                0)                          # src1_rep_stride
        #        self.tik_inst.data_move(self.data_out[i * 24], self.ub_input[1024 * 10], 0, 1, 3, 0, 0)

    def compute_2(self):
        """
        compute_2 (100,240) -> (100, 120)
        """
        with self.tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
            with self.tik_inst.if_scope(block_idx == 0):
                scalar_value = self.tik_inst.Scalar("uint32")
                scalar_value.set_as(1431655765)
                self.ub_pattern[0].set_as(scalar_value)
                self.ub_pattern[1].set_as(scalar_value)
                self.ub_pattern[2].set_as(scalar_value)
                self.ub_pattern[3].set_as(scalar_value)
                self.ub_pattern[4].set_as(scalar_value)
                self.ub_pattern[5].set_as(scalar_value)
                self.ub_pattern[6].set_as(scalar_value)
                self.ub_pattern[7].set_as(scalar_value)
                with self.tik_inst.for_range(0, 100) as i:
                    self.tik_inst.data_move(self.ub_input, self.data_in[i * 240], 0, 1, 30, 0, 0)
                    self.tik_inst.vreducev2(240,                        # mask
                                            self.ub_input[1024 * 10],   # dst
                                            self.ub_input,              # src0
                                            self.ub_pattern,            # src1_pattern
                                            1,                          # repeat_time
                                            1,                          # src0_blk_stride
                                            0,                          # src0_rep_stride
                                            0,                          # src1_rep_stride
                                            None,
                                            "counter")
                    self.tik_inst.data_move(self.data_out[i * 120], self.ub_input[1024 * 10], 0, 1, 15, 0, 0)


@register_operator("AsStrided")
def as_strided(x, size, stride, storage_offset, y, kernel_name="as_strided"):
    """
    Generate contiguous memory with the given shape and strides

    Parameters
    ----------
    x : the input tensor
    size : the shape of output tensor
    stride: the stride of output tensor
    stroage_offset : the offset in the underlying storage of the output tensor
    y : the output tensor
    Returns
    -------
    compile info
    """
    tik_inst = tik.Tik()
    x_dtype = x.get("dtype").lower()
    size_dtype = size.get("dtype").lower()
    stride_dtype = stride.get("dtype").lower()
    y_shape = y.get("ori_shape")
    #storage_offset_dtype = storage_offset.get("dtype").lower()

    data_in  = tik_inst.Tensor(x_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "x")
    size = tik_inst.Tensor(size_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "size")
    stride = tik_inst.Tensor(stride_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "stride")
    storage_offset = tik_inst.Tensor(stride_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "storage_offset")
    data_out = tik_inst.Tensor(x_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "y")
    data_tiling = tik_inst.Tensor("int64", (TILING_MAX_SIZE_GM,), tik.scope_gm, "data_tiling")
    input_list = [data_in, size, stride, storage_offset]
    tensor_list = [data_in, size, stride, storage_offset, data_out, data_tiling]
    as_strided_instance = AsStrided(tik_inst, x_dtype, tensor_list)
    with tik_inst.if_scope(y_shape[1] == 24):
        as_strided_instance.compute_1()
    with tik_inst.if_scope(y_shape[1] == 120):
        as_strided_instance.compute_2()
    tbe_context.get_context().add_compile_info("vars", {"ub_size": UB_SIZE, "core_num": CORE_NUM, "dtype": x_dtype})
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=input_list,
                      outputs=[data_out],
                      flowtable=[data_tiling])
    return {"compile_info": tbe_context.get_context().get_compile_info()}

