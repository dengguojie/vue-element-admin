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

DTYPE_BYTE_NUM = 4  # only fp32

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

    kernel_h, kernel_w = kernel_size
    kernel_num = kernel_h * kernel_w

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    ho = (output_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    wo = (output_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    wo_16 = 16
    wo_num = (wo + wo_16 - 1) // wo_16

    with tik_instance.for_range(
            0, output_batch * output_c1, block_num = output_batch * output_c1
        ) as nc:
        n = nc // output_c1
        ci = nc % output_c1
        
        output_ub = tik_instance.Tensor(
            "float32", (output_w, output_c0), tik.scope_ubuf, "output_ub"
        )
        rpt = (output_w * output_c0) // 64  # fewer than 255
        rmd = (output_w * output_c0) % 64

        tik_instance.vec_dup(64, output_ub, 0, rpt, 8)
        if (rmd):
            tik_instance.vec_dup(rmd, output_ub[rpt * 64], 0, 1, 8)

        tik_instance.data_move(
            y_gm[n, ci, 0, 0, 0], output_ub, 0, output_h, 
            (output_w * output_c0 * DTYPE_BYTE_NUM) // 32, 0, 0
        )

        input_ub = tik_instance.Tensor(
            "float32", (wo_num * wo_16, input_c0), tik.scope_ubuf, "input_ub"
        )
        tik_instance.vec_dup(64, input_ub, 0, (wo_num * wo_16 * input_c0) // 64, 8)
        
        with tik_instance.for_range(0, kernel_num) as mask_id:
            width = mask_id % kernel_w
            height = mask_id // kernel_w
            with tik_instance.for_range(0, ho) as h:
                # don't support to padding at current version
                output_offset_h = height * dilation_h + h * stride_h 
                
                tik_instance.data_move(
                    input_ub, x_gm[n, ci, mask_id, h * wo, 0], 
                    0, 1, (wo * input_c0 * DTYPE_BYTE_NUM) // 32, 0, 0
                )

                tik_instance.data_move(
                    output_ub, y_gm[n, ci, output_offset_h, 0, 0], 
                    0, 1, (output_w * output_c0 * DTYPE_BYTE_NUM) // 32, 0, 0
                )
                
                tik_instance.col2img(
                    output_ub, input_ub, (0, 0, 0, 0), output_h, output_w, width, 
                    0, 0, 0, stride_w, stride_h, kernel_w, kernel_h, 
                    dilation_w, dilation_h, wo_num
                )

                tik_instance.data_move(
                    y_gm[n, ci, output_offset_h, 0, 0], output_ub, 
                    0, 1, (output_w * output_c0 * DTYPE_BYTE_NUM) // 32, 0, 0
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
    tik_dprofile = tik.Dprofile("v100", "cloud")
    tik_instance = tik.Tik(tik_dprofile)

    output_shape = y["shape"]
    output_batch, output_c1, output_h, output_w, output_c0 = output_shape

    input_shape = x["shape"]
    input_batch, input_c1, input_w, input_h, input_c0 = input_shape

    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    ho = (output_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    wo = (output_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    assert output_batch == input_batch, "output_batch should equal to input_batch."
    assert output_c1 == input_c1, "output_c1 should equal to input_c1."
    assert output_c0 == 16, "output_c0 should equal to 0."
    assert input_c0 == 16, "input_c0 should equal to 0."

    assert input_h == ho * wo, "input_h should equal to ho*wo."
    assert input_w == kernel_h * kernel_w, "input_w should equal to kernel_h*kernel_w."

    y_gm = tik_instance.Tensor("float32", output_shape, tik.scope_gm, "y_gm")
    x_gm = tik_instance.Tensor("float32", input_shape, tik.scope_gm, "x_gm")
    output_size_gm = tik_instance.Tensor("int32", output_size["shape"], tik.scope_gm, "output_size_gm")

    col2im_compute(x_gm, output_size_gm, y_gm, kernel_size, dilation, padding, stride, tik_instance)

    tik_instance.BuildCCE(
        kernel_name = kernel_name,
        inputs = [x_gm, output_size_gm],
        outputs = [y_gm]
    )

    return tik_instance  

