#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPoolGradGradWithArgmax", None, None)

def case_gen(inputShape, ksize, strides, dtype="float16", pad="VALID", format="NHWC"):
    fmap_n, fmap_h, fmap_w, fmap_c = inputShape
    inputShape = fmap_n, (fmap_c + 15) // 16, fmap_h, fmap_w, 16
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    shape_max_pool_h, pad_top, pad_bottom = \
        _get_windowed_output_size(fmap_h, kernel_h, stride_h, pad)
    shape_max_pool_w, pad_left, pad_right = \
        _get_windowed_output_size(fmap_w, kernel_w, stride_w, pad)
    # shape_max_pool = (fmap_n, shape_max_pool_h, shape_max_pool_w, fmap_c)
    shape_max_pool = (fmap_n, (fmap_c + 15) // 16, shape_max_pool_h, shape_max_pool_w, 16)

    outputShape = shape_max_pool
    argmax_shape = (fmap_n, (fmap_c + 15) // 16, kernel_h*kernel_w, (shape_max_pool[2]*shape_max_pool[3] + 31) // 16 * 16, 16 // 16)
    return inputShape, argmax_shape, outputShape, dtype, ksize, pad, strides, format
def _get_windowed_output_size(input_size, kernel_size,
                              stride, padding_type,
                              dilation_rate=1):
    if stride <= 0:
        raise RuntimeError("Stride must be > 0, but got", stride)

    if dilation_rate < 1:
        raise RuntimeError("Dilation rate must be >= 1, but got", dilation_rate)

    effective_kernel_size = (kernel_size - 1) * dilation_rate + 1  # 3
    if padding_type == "VALID":
        output_size = (input_size - effective_kernel_size + stride) // stride
        padding_before = 0
        padding_after = 0
    elif padding_type == "SAME":
        output_size = (input_size + stride - 1) // stride
        padding_needed = (output_size - 1)*stride + effective_kernel_size - input_size
        # padding_needed = max(0, (output_size - 1) * stride + effective_kernel_size - input_size)
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before
    else:
        raise RuntimeError("Unsupported padding type", padding_type)

    return output_size, padding_before, padding_after
inputShape, argmax_shape, outputShape, dtype, ksize, pad, strides, format = case_gen(inputShape=(1, 768, 768, 16), ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), pad="SAME")

x = {"shape": inputShape, "ori_shape": inputShape, "format": format, "dtype": "float16", "ori_format": format}
grad = {"shape": inputShape, "ori_shape": inputShape, "format": format, "dtype": "float16", "ori_format": format}
argmax = {"shape": argmax_shape, "ori_shape": argmax_shape, "format": format, "dtype": "uint16", "ori_format": format}
y = {"shape": outputShape, "ori_shape": outputShape, "format": format, "dtype": "float16", "ori_format": format}

case1 = {"params":[x, grad, argmax, y, ksize, strides, pad],
         "case_name": "max_pool_grad_grad_with_argmax_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)