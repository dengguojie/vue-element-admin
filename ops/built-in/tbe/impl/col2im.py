"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Col2im
"""

from te import tik
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check
from impl import constant_util as constant

# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@fusion_manager.register("Col2im")
def col2im_compute(
        x_gm, output_size_gm, y_gm, kernel_size, dilation, padding, stride, tik_instance
    ):
    """
    do Col2im compute
    Parameters:
    ----------------
    x : input tensor x
    output_size : input tensor output_size
    y : output tensor  y
    kernel_size : value of kernel_size, data type int[2]
    dilation : value of dilation, data type int[2]
    padding : value of padding, data type int[2]
    stride : value of stride, data type int[2]
    kernel_name : cce kernel name, default value is "Col2im"
    ----------------
    """
    output_batch, output_c1, output_h, output_w, output_c0 = y_gm.shape
    input_batch, input_c1, input_w, input_h, input_c0 = x_gm.shape

    input_dtype = x_gm.dtype
    output_dtype = y_gm.dtype
    if input_dtype == "float32":
        dtype_byte_num = constant.DATA_SIZE_FOUR
        mask = constant.MASK64
    else:
        dtype_byte_num = constant.DATA_SIZE_TWO
        mask = constant.MASK128

    kernel_h, kernel_w = kernel_size
    kernel_num = kernel_h * kernel_w

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    ho = (output_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    wo = (output_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    with tik_instance.for_range(
            0, output_batch * output_c1, block_num = output_batch * output_c1
        ) as nc:
        n = nc // output_c1
        ci = nc % output_c1
        
        with tik_instance.new_stmt_scope():
            zeors_ub = tik_instance.Tensor(
                output_dtype, (output_w, output_c0), tik.scope_ubuf, "zeors_ub"
            )
            dup_rpt = (output_w * output_c0) // mask
            dup_rmd = (output_w * output_c0) % mask
            
            if (dup_rpt):
                tik_instance.vec_dup(mask, zeors_ub, 0, dup_rpt, constant.REPEAT_STRIDE_EIGHT)
            if (dup_rmd):
                tik_instance.vec_dup(dup_rmd, zeors_ub[dup_rpt * mask], 0, constant.REPEAT_TIME_ONCE, constant.REPEAT_STRIDE_EIGHT)
            
            with tik_instance.for_range(0, output_h) as height_id:
                tik_instance.data_move(
                    y_gm[n, ci, height_id, 0, 0], zeors_ub, constant.SID, 1, 
                    (output_w * output_c0 * dtype_byte_num) // constant.BLOCK_SIZE, 
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )
        output_ub = tik_instance.Tensor(
            output_dtype, (output_w, output_c0), tik.scope_ubuf, "output_ub"
        )

        input_ub = tik_instance.Tensor(
            input_dtype, (wo, input_c0), tik.scope_ubuf, "input_ub"
        )        
        with tik_instance.for_range(0, kernel_num) as mask_id:
            width = mask_id % kernel_w
            height = mask_id // kernel_w
            with tik_instance.for_range(0, ho) as h:
                # don't support to padding at current version
                output_offset_h = height * dilation_h + h * stride_h 
                

                tik_instance.data_move(
                    input_ub, x_gm[n, ci, mask_id, h * wo, 0], 
                    constant.SID, constant.DEFAULT_NBURST, 
                    (wo * input_c0 * dtype_byte_num) // constant.BLOCK_SIZE, 
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )
                tik_instance.data_move(
                    output_ub, y_gm[n, ci, output_offset_h, 0, 0], 
                    constant.SID, constant.DEFAULT_NBURST, 
                    (output_w * output_c0 * dtype_byte_num) // constant.BLOCK_SIZE, 
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )
                
                add_rpt = (wo * input_c0) // mask
                add_rmd = (wo * input_c0) % mask
                
                kernel_w_offset = width * constant.SIZE_SIXTEEN

                if (add_rpt):
                    tik_instance.vadd(
                        mask, output_ub[kernel_w_offset], output_ub[kernel_w_offset], input_ub,
                        add_rpt, constant.BLOCK_STRIDE_ONE, constant.BLOCK_STRIDE_ONE, constant.BLOCK_STRIDE_ONE,
                        constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT
                    )

                if (add_rmd):
                    add_rpt_offset = add_rpt * mask
                    output_ub_offset = kernel_w_offset + add_rpt_offset
                    tik_instance.vadd(
                        add_rmd, output_ub[output_ub_offset], output_ub[output_ub_offset], input_ub[add_rpt_offset],
                        constant.REPEAT_TIME_ONCE, constant.BLOCK_STRIDE_ONE, constant.BLOCK_STRIDE_ONE, constant.BLOCK_STRIDE_ONE,
                        constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT, constant.REPEAT_STRIDE_EIGHT
                    )



                tik_instance.data_move(
                    y_gm[n, ci, output_offset_h, 0, 0], output_ub, 
                    constant.SID, constant.DEFAULT_NBURST, 
                    (output_w * output_c0 * dtype_byte_num) // constant.BLOCK_SIZE, 
                    constant.STRIDE_ZERO, constant.STRIDE_ZERO
                )


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, 
    para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT, 
    para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT, 
    para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME
)
def col2im(
        x, output_size, y, kernel_size, dilation, padding, stride, kernel_name="Col2im"
    ):
    """
    do col2im operation on x, result is y, and y's height/width is value of output_size
    Parameters:
    ----------
    x : dict of x, include shape and dtype, dtype support float32
    output_size : dict of output_size, include shape and dtype, dtype support int32
    y : dict of y, include shape and dtype, dtype support float32
    kernel_size : value of kernel_size, data type int[2]
    dilation : value of dilation, data type int[2]
    padding : value of padding, data type int[2]
    stride : value of stride, data type int[2]
    kernel_name : cce kernel name, default value is "Col2im"
    -------
    """
    tik_instance = tik.Tik()

    y_gm = tik_instance.Tensor(y["dtype"], y["shape"], tik.scope_gm, "y_gm")
    x_gm = tik_instance.Tensor(x["dtype"], x["shape"], tik.scope_gm, "x_gm")
    output_size_gm = tik_instance.Tensor(output_size["dtype"], output_size["shape"], tik.scope_gm, "output_size_gm")

    col2im_compute(x_gm, output_size_gm, y_gm, kernel_size, dilation, padding, stride, tik_instance)

    tik_instance.BuildCCE(
        kernel_name = kernel_name,
        inputs = [x_gm, output_size_gm],
        outputs = [y_gm]
    )

    return tik_instance  

