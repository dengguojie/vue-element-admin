# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe
from tbe.common.testing.testing import debug

warnings.filterwarnings("ignore")
ut_case = OpUT("pooling2d_cpu", "dsl_cpu.test_pooling2d_cpu_impl")


def _gen_golden_data(shape_in, shape_window, stride, pooling_mode, input1_dtype, output_dtype, ctx,
                     padding_mode, pad=(0, 0, 0, 0), dilation=(1, 1)):
    batch_size, c1, h, w, _ = shape_in
    windows_h, windows_w = shape_window
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_top, pad_bottom, pad_left, pad_right = pad

    if pooling_mode in ["AVG", "MAX"]:
        if padding_mode == "SAME":
            outshape_h = (h + stride_h - 1) // stride_h
            outshape_w = (w + stride_w - 1) // stride_w
            pad_rows = (outshape_h - 1) * stride_h + ((windows_h - 1) * dilation_h + 1) - h
            pad_cols = (outshape_w - 1) * stride_w + ((windows_w - 1) * dilation_w + 1) - w
            if pad_rows % 2 == 0:
                pad_top = pad_bottom = pad_rows // 2
            else:
                pad_top = pad_rows // 2
                pad_bottom = pad_rows - pad_top
            if pad_cols % 2 == 0:
                pad_left = pad_right = pad_cols // 2
            else:
                pad_left = pad_cols // 2
                pad_right = pad_cols - pad_left

            pad_top = 0 if pad_top < 0 else pad_top
            pad_bottom = 0 if pad_bottom < 0 else pad_bottom
            pad_left = 0 if pad_left < 0 else pad_left
            pad_right = 0 if pad_right < 0 else pad_right
        if padding_mode == "VALID":
            outshape_h = (h - windows_h + 1 + (stride_h - 1)) // stride_h
            outshape_w = (w - windows_w + 1 + (stride_w - 1)) // stride_w
    if pooling_mode in ["GAP", "GMP"]:
        outshape_h = outshape_w = 1

    input_shape_nchw = (batch_size, c1 * 16, h, w)
    res_outshape_nchw = (batch_size, c1 * 16, outshape_h, outshape_w)

    np.set_printoptions(suppress=True)
    tensor_in_nchw = np.array(np.random.uniform(low=0.0, high=0.9, size=input_shape_nchw).astype(input1_dtype))
    res_out_nchw = np.array(np.zeros(res_outshape_nchw, dtype=output_dtype))
    tensor_in_nc1hwc0 = tvm.nd.array(tensor_in_nchw.reshape(batch_size, c1, 16, h, w).transpose(0, 1, 3, 4, 2), ctx)

    if pooling_mode in ["AVG", "GAP"]:
        if padding_mode == "VALID":
            valid_h = (outshape_h - 1) * stride_h + windows_h
            valid_w = (outshape_w - 1) * stride_w + windows_w
            valid_shape = (batch_size, c1 * 16, valid_h, valid_w)
            tensor_in_with_pad = np.array(np.zeros(valid_shape)).astype(np.float16)
            tensor_in_with_pad[:, :, :valid_h, :valid_w] = tensor_in_nchw[:, :, :valid_h, :valid_w]
            for i in range(outshape_h):
                for j in range(outshape_w):
                    tensor_in_mask_with_window = tensor_in_with_pad[
                                                 :, :, i * stride_h: i * stride_h + windows_h,
                                                 j * stride_w: j * stride_w + windows_w]
                    res_out_nchw[:, :, i, j] = \
                        1.0 * np.sum(tensor_in_mask_with_window, axis=(2, 3)) / (windows_h * windows_w)
        if padding_mode == "SAME":
            same_shape = (batch_size, c1 * 16, h + pad_top + pad_bottom, w + pad_left + pad_right)
            tensor_in_with_pad = np.array(np.zeros(same_shape)).astype(np.float16)
            tensor_in_with_pad[:, :, pad_top: h + pad_top, pad_left: w + pad_left] = tensor_in_nchw

            for i in range(outshape_h):
                for j in range(outshape_w):
                    tensor_in_mask_with_window = tensor_in_with_pad[
                                                 :, :, i * stride_h: i * stride_h + windows_h,
                                                 j * stride_w: j * stride_w + windows_w]
                    res_out_nchw[:, :, i, j] = np.sum(tensor_in_mask_with_window, axis=(2, 3))

            avg_mean_factor = []
            for i in range(outshape_h):
                for j in range(outshape_w):
                    area = windows_w * windows_h
                    mean_value = 1.0 / float(area)
                    avg_mean_factor.append(mean_value)
            avg_mean_factor = np.array(avg_mean_factor).reshape(outshape_h, outshape_w)

            for i in range(outshape_h):
                for j in range(outshape_w):
                    res_out_nchw[:, :, i, j] = res_out_nchw[:, :, i, j] * avg_mean_factor[i, j]

    if pooling_mode in ["MAX", "GMP"]:
        if padding_mode == "VALID":
            valid_h = (outshape_h - 1) * stride_h + windows_h
            valid_w = (outshape_w - 1) * stride_w + windows_w
            valid_shape = (batch_size, c1 * 16, valid_h, valid_w)
            tensor_in_with_pad = np.array(np.zeros(valid_shape))
            tensor_in_with_pad[:, :, :valid_h, :valid_w] = tensor_in_nchw[:, :, :valid_h, :valid_w]
        if padding_mode == "SAME":
            same_shape = (batch_size, c1 * 16, h + pad_top + pad_bottom, w + pad_left + pad_right)
            tensor_in_with_pad = np.array(np.zeros(same_shape)) * (-65504)
            tensor_in_with_pad[:, :, pad_top: h + pad_top, pad_left: w + pad_left] = tensor_in_nchw

        for i in range(outshape_h):
            for j in range(outshape_w):
                tensor_in_mask_with_window = tensor_in_with_pad[
                                             :, :, i * stride_h: i * stride_h + windows_h,
                                             j * stride_w: j * stride_w + windows_w]
                res_out_nchw[:, :, i, j] = np.max(tensor_in_mask_with_window, axis=(2, 3))
    return tensor_in_nc1hwc0, res_out_nchw, outshape_h, outshape_w


def test_pooling2d_cpu_api_avg(_):
    """
    for pooling2d api
    @return: Ture && false
    """
    shape = (1, 1, 112, 112, 16)
    windows = (1, 1)
    stride = (2, 2)
    input1 = tvm.placeholder(shape, name="input1", dtype="float16")
    output = tbe.pooling2d(input1, windows, stride, "AVG", "SAME")
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a, golden_data, out_h, out_w = \
        _gen_golden_data(shape, windows, stride, "AVG", input1.dtype, output.dtype, ctx, "SAME")
    b = tvm.nd.array(np.zeros((1, 1, out_h * out_w, 16), dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(
            b.asnumpy().reshape(1, 1, out_h, out_w, 16).transpose(0, 4, 2, 3, 1).reshape(1, 16, out_h, out_w),
            golden_data)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_pooling2d_cpu_api_gap(_):
    """
    for pooling2d api
    @return: Ture && false
    """
    shape = (1, 1, 3, 3, 16)
    windows = (3, 3)
    stride = (2, 2)
    input1 = tvm.placeholder(shape, name="input1", dtype="float16")
    output = tbe.pooling2d(input1, windows, stride, "GAP", "VALID")
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a, golden_data, out_h, out_w = \
        _gen_golden_data(shape, windows, stride, "GAP", input1.dtype, output.dtype, ctx, "VALID")
    b = tvm.nd.array(np.zeros((1, 1, out_h * out_w, 16), dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(
            b.asnumpy().reshape(1, 1, out_h, out_w, 16).transpose(0, 4, 2, 3, 1).reshape(1, 16, out_h, out_w),
            golden_data, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_pooling2d_cpu_api_gmp(_):
    """
    for pooling2d api
    @return: Ture && false
    """
    shape = (1, 1, 10, 10, 16)
    windows = (10, 10)
    stride = (2, 2)
    input1 = tvm.placeholder(shape, name="input1", dtype="float16")
    output = tbe.pooling2d(input1, windows, stride, "GMP", "VALID")
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a, golden_data, out_h, out_w = \
        _gen_golden_data(shape, windows, stride, "GMP", input1.dtype, output.dtype, ctx, "VALID")
    b = tvm.nd.array(np.zeros((1, 1, out_h * out_w, 16), dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(
            b.asnumpy().reshape(1, 1, out_h, out_w, 16).transpose(0, 4, 2, 3, 1).reshape(1, 16, out_h, out_w),
            golden_data, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_pooling2d_cpu_api_max(_):
    """
    for pooling2d api
    @return: Ture && false
    """
    with debug():
        shape = (1, 1, 15, 15, 16)
        windows = (3, 3)
        stride = (1, 1)
        input1 = tvm.placeholder(shape, name="input1", dtype="float16")
        output = tbe.pooling2d(input1, windows, stride, "MAX", "VALID")
        sch = tvm.create_schedule(output.op)
        func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a, golden_data, out_h, out_w = \
            _gen_golden_data(shape, windows, stride, "MAX", input1.dtype, output.dtype, ctx, "VALID")
        b = tvm.nd.array(np.zeros((1, 1, out_h, out_w, 16), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func(a, b)
        # 3.verify the correctness of output
        try:
            tvm.testing.assert_allclose(
                b.asnumpy().reshape(1, 1, out_h, out_w, 16).transpose(0, 4, 2, 3, 1).reshape(1, 16, out_h, out_w),
                golden_data)
        except AssertionError as e:
            print(e)
            return False
        return True


def test_pooling2d_cpu_api_gmp_windows_lt_nine(_):
    """
    for pooling2d api
    @return: Ture && false
    """
    with debug():
        shape = (1, 1, 8, 8, 16)
        windows = (8, 8)
        stride = (1, 1)
        input1 = tvm.placeholder(shape, name="input1", dtype="float16")
        output = tbe.pooling2d(input1, windows, stride, "GMP", "VALID")
        sch = tvm.create_schedule(output.op)
        func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a, golden_data, out_h, out_w = \
            _gen_golden_data(shape, windows, stride, "GMP", input1.dtype, output.dtype, ctx, "VALID")
        b = tvm.nd.array(np.zeros((1, 1, out_h, out_w, 16), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func(a, b)
        # 3.verify the correctness of output
        try:
            tvm.testing.assert_allclose(
                b.asnumpy().reshape(1, 1, out_h, out_w, 16).transpose(0, 4, 2, 3, 1).reshape(1, 16, out_h, out_w),
                golden_data)
        except AssertionError as e:
            print(e)
            return False
        return True


test_func_list = [
    # test_pooling2d_cpu_api_avg,
    test_pooling2d_cpu_api_gap,
    test_pooling2d_cpu_api_gmp,
    test_pooling2d_cpu_api_max,
    test_pooling2d_cpu_api_gmp_windows_lt_nine,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
