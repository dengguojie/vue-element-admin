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

c1hwc0_2_chw
"""


from te import tik
from te import platform as tbe_platform


def _ceil_div(value_x, value_y):
    """
    do ceil division
    """
    return (value_x + value_y - 1) // value_y


def _floor_trunc(value_x, value_y):
    """
    do floor truncate
    """
    return value_x // value_y * value_y


def _get_dtype_factor(dtype):
    """
    get dtype length in byte
    """

    if dtype.lower() == "float16":
        dtype_factor = 2
    elif dtype.lower() in ("float32", "int32"):
        dtype_factor = 4
    else:
        dtype_factor = 1

    return dtype_factor


# 'pylint: disable=too-many-locals,too-many-statements
def _multi_core_on_hw(tik_inst, data_in, data_out, shape_in, shape_out):
    """
    do c1hwc0 to chw transfer by multiple core on axis hw
    """

    axis_n, axis_c1, axis_h, axis_w, axis_c0 = shape_in
    axis_c = shape_out[1]
    axis_hw = axis_h * axis_w
    hw_size = axis_hw
    dtype_len = _get_dtype_factor(data_in.dtype)
    vnc_len = 16
    block_byte_size = 32
    ele_count_per_block = block_byte_size // dtype_len
    # save 8kb to avoid repeat time of vnchwconv is larger than 255
    mod_ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
    if mod_ub_size > 248 * 1024:
        mod_ub_size = 248 * 1024
    need_ub_size = _floor_trunc(mod_ub_size // 2 // dtype_len, ele_count_per_block)

    # each core process certain hw'c lines
    core_num = _ceil_div(hw_size, _ceil_div(hw_size,
                         tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)))
    # to make sure every core process hw is block align except last core
    sub_hw_per_loop = _floor_trunc(need_ub_size // vnc_len // axis_c0, ele_count_per_block) * vnc_len
    per_core_hw_cnt = _floor_trunc(_ceil_div(hw_size, core_num), ele_count_per_block)
    last_core_hw_cnt = hw_size - per_core_hw_cnt * (core_num - 1)

    # alloc input and output ub
    in_ub = tik_inst.Tensor(data_in.dtype, (need_ub_size,), name="in_ub", scope=tik.scope_ubuf)
    out_ub = tik_inst.Tensor(data_in.dtype, (need_ub_size,), name="out_ub", scope=tik.scope_ubuf)

    with tik_inst.for_range(0, core_num, block_num=core_num) as block_idx:

        # 'pylint: disable=too-many-locals,too-many-statements
        def _hw_transfer_process(axis_n_index, axis_c1_index, sub_hw_len):
            """
            process of hw transfer
            """
            # 'pylint: disable=not-use-list-comprehension
            def _inner_hw_transfer(sub_hw_lp_idx, inner_hw_len):
                """
                inner hw transfer process
                """

                # move data to ubuf
                inner_hw_block_cnt = inner_hw_len // ele_count_per_block
                inner_hw_block_block_align = _ceil_div(inner_hw_len, ele_count_per_block)
                inner_hw_left = inner_hw_len % ele_count_per_block
                back_len = inner_hw_left - ele_count_per_block

                in_offset = (axis_n_index * axis_c1 * axis_hw + axis_c1_index * axis_hw +
                             sub_hw_lp_idx * sub_hw_per_loop + block_idx * per_core_hw_cnt) * axis_c0
                if inner_hw_block_cnt:
                    tik_inst.data_move(in_ub, data_in[in_offset],
                                       0, 1, inner_hw_block_cnt * axis_c0, 0, 0)
                if inner_hw_left:
                    # to padding hw to block align
                    tik_inst.data_move(in_ub[inner_hw_block_cnt * ele_count_per_block * axis_c0],
                                       data_in[in_offset +
                                               (inner_hw_block_cnt * ele_count_per_block + back_len) * axis_c0],
                                       0, 1, axis_c0, 0, 0)

                # do hwc0 to c0hw transfer
                vnc_idx_list = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
                src_addr_list = [in_ub[vnc_len * i] for i in vnc_idx_list]
                dst_addr_list = [out_ub[sub_hw_per_loop * i] for i in vnc_idx_list]
                repeat_cnt = _ceil_div(inner_hw_block_block_align * axis_c0, vnc_len)
                src_stride = 0 if repeat_cnt == 1 else 16
                dst_stride = 0 if repeat_cnt == 1 else 1
                tik_inst.vnchwconv(False, False,
                                   dst_addr_list, src_addr_list,
                                   repeat_cnt, dst_stride, src_stride)

                # move data to gm
                out_offset = (axis_n_index * axis_c * axis_hw + axis_c1_index * axis_c0 * axis_hw +
                              sub_hw_lp_idx * sub_hw_per_loop + block_idx * per_core_hw_cnt)
                with tik_inst.new_stmt_scope():
                    c0_lp_cnt = tik_inst.Scalar()
                    with tik_inst.if_scope(tik.any(axis_c1_index != axis_c1 - 1, axis_c % axis_c0 == 0)):
                        c0_lp_cnt.set_as(axis_c0)
                    with tik_inst.else_scope():
                        c0_lp_cnt.set_as(axis_c % axis_c0)

                    if inner_hw_len % ele_count_per_block > 0:
                        if inner_hw_block_cnt > 0:
                            with tik_inst.for_range(0, c0_lp_cnt) as c0_idx:
                                tik_inst.data_move(data_out[out_offset + c0_idx * hw_size],
                                                   out_ub[c0_idx * sub_hw_per_loop], 0, 1,
                                                   inner_hw_len // ele_count_per_block, 0, 0)
                        with tik_inst.for_range(0, c0_lp_cnt) as c0_idx1:
                            tik_inst.data_move(data_out[out_offset + c0_idx1 * hw_size + back_len +
                                                        inner_hw_len // ele_count_per_block * ele_count_per_block],
                                               out_ub[c0_idx1 * sub_hw_per_loop +
                                                      inner_hw_len // ele_count_per_block * ele_count_per_block],
                                               0, 1, 1, 0, 0)

                    else:
                        with tik_inst.for_range(0, c0_lp_cnt) as c0_idx:
                            tik_inst.data_move(data_out[out_offset + c0_idx * hw_size],
                                               out_ub[c0_idx * sub_hw_per_loop], 0, 1,
                                               _ceil_div(inner_hw_len, ele_count_per_block), 0, 0)

            sub_hw_lp_cnt = sub_hw_len // sub_hw_per_loop
            sub_hw_left = sub_hw_len % sub_hw_per_loop

            with tik_inst.for_range(0, sub_hw_lp_cnt) as sub_hw_lp_idx:
                _inner_hw_transfer(sub_hw_lp_idx, sub_hw_per_loop)
            if sub_hw_left:
                _inner_hw_transfer(sub_hw_lp_cnt, sub_hw_left)

        with tik_inst.for_range(0, axis_n) as axis_n_idx:
            with tik_inst.for_range(0, axis_c1) as axis_c1_idx:
                with tik_inst.if_scope(block_idx == core_num - 1):
                    _hw_transfer_process(axis_n_idx, axis_c1_idx, last_core_hw_cnt)
                with tik_inst.else_scope():
                    _hw_transfer_process(axis_n_idx, axis_c1_idx, per_core_hw_cnt)


def c1hwc0_2_chw_compute(tik_inst, data_in, data_out):
    """
    do hwc to chw transfer
    """
    _multi_core_on_hw(tik_inst, data_in, data_out, shape_in.shape, shape_out.shape)


def c1hwc0_2_chw(in_shape, dst_shape, in_dtype, kernel_name="c1hwc0_2_chw"):
    """
    used to transfer c1hwc0 to chw
    """

    # initial Tik
    tik_inst = tik.Tik()
    # define input and output tensors
    data_in = tik_inst.Tensor(in_dtype, in_shape, tik.scope_gm, "data_in")
    data_out = tik_inst.Tensor(in_dtype, dst_shape, tik.scope_gm, "data_out")

    # do transfer
    c1hwc0_2_chw_compute(tik_inst, data_in, data_out)

    # build cce
    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[data_in], outputs=[data_out])
