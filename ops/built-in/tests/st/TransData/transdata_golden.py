import numpy as np
from functools import reduce


def _pad_len(x, y):
    res = (y - x % y) % y
    return res


def _ceil_div(x, y):
    res = (x + y - 1) // y
    return res


def calc_expect_func_nd_2_nz(src, dst, src_format, dst_format):
     in_shape = src.get("shape")
     if -1 in in_shape:
         in_shape = src.get("typical_shape")
     input_tensor = src.get("value")
     dst_shape = dst.get("shape")
     in_len = len(in_shape)

     if in_len == 1:
         axis_h, axis_n, axis_c = 1, 1, in_shape[0]
     elif in_len == 2:
         axis_h, axis_n, axis_c == 1, in_shape[0], in_shape[1]
     else:
         axis_h, axis_n, axis_c = reduce(lambda x, y: x * y, in_shape[:-2]), in_shape[-2], in_shape[-1]
     axis_c0 = dst_shape[-1]
     axis_ni = 16
     axis_c1 = _ceil_div(axis_c, axis_c0)
     axis_no = _ceil_div(axis_n, axis_ni)
     c_pad = _pad_len(axis_c, axis_c0)
     n_pad = _pad_len(axis_n, axis_ni)

     tmp_input_tensor = np.pad(input_tensor, ((0, 0), (0, n_pad), (0, c_pad)), mode="constant", constant_values=(0, 0))
     tmp_input_tensor = tmp_input_tensor.reshape(axis_h, axis_no, axis_ni, axis_c1, axis_c0)
     output_tensor = np.transpose(tmp_input_tensor, axes=(0, 3, 1, 2, 4))

     return output_tensor


def calc_expect_func_nchw_2_zn(src, dst, src_format, dst_format):
     in_shape = src.get("shape")
     if -1 in in_shape:
         in_shape = src.get("typical_shape")
     input_tensor = src.get("value")
     dst_shape = dst.get("shape")
     in_len = len(in_shape)

     axis_n, axis_c, axis_h, axis_w = in_shape
     axis_c0 = dst_shape[-1]
     axis_ni = 16
     axis_c1 = _ceil_div(axis_c, axis_c0)
     axis_no = _ceil_div(axis_n, axis_ni)
     c_pad = _pad_len(axis_c, axis_c0)
     n_pad = _pad_len(axis_n, axis_ni)

     tmp_input_tensor = np.pad(input_tensor, ((0, n_pad), (0, c_pad), (0, 0), (0, 0)), mode="constant", constant_values=(0, 0))
     tmp_input_tensor = tmp_input_tensor.reshape(axis_no, axis_ni, axis_c1, axis_c0, axis_h, axis_w)
     output_tensor = np.transpose(tmp_input_tensor, axes=(2, 4, 5, 0, 1, 3))
     output_tensor = output_tensor.reshape(axis_c1*axis_h*axis_w, axis_no, axis_ni, axis_c0)

     return output_tensor


def calc_expect_func_ndhwc_2_z3d(src, dst, src_format, dst_format):
     in_shape = src.get("shape")
     if -1 in in_shape:
         in_shape = src.get("typical_shape")
     input_tensor = src.get("value")
     dst_shape = dst.get("shape")
     in_len = len(in_shape)

     axis_n, axis_d, axis_h, axis_w, axis_c = in_shape
     axis_c0 = dst_shape[-1]
     axis_ni = 16
     axis_c1 = _ceil_div(axis_c, axis_c0)
     axis_no = _ceil_div(axis_n, axis_ni)
     c_pad = _pad_len(axis_c, axis_c0)
     n_pad = _pad_len(axis_n, axis_ni)

     tmp_input_tensor = np.pad(input_tensor, ((0, n_pad), (0, 0), (0, 0), (0, 0), (0, c_pad)), mode="constant", constant_values=(0, 0))
     tmp_input_tensor = tmp_input_tensor.reshape(axis_no, axis_ni, axis_d, axis_h, axis_w, axis_c1, axis_c0)
     output_tensor = np.transpose(tmp_input_tensor, axes=(2, 5, 3, 4, 0, 1, 6))

     return output_tensor


def calc_expect_func_z3d_2_ndhwc(src, dst, src_format, dst_format):
    in_shape = src.get("shape")
    if -1 in in_shape:
         in_shape = src.get("typical_shape")
    input_tensor = src.get("value")
    dst_shape = dst.get("shape")

    axis_n, axis_d, axis_h, axis_w, axis_c = dst_shape
    axis_dc1hw, axis_no, axis_ni, axis_c0 = in_shape
    axis_c1 = axis_dc1hw // (axis_d * axis_h * axis_w)

    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = input_tensor.reshape(axis_d, axis_c1, axis_h, axis_w, axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(4, 5, 0, 2, 3, 1, 6))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_no*axis_ni, axis_d, axis_h, axis_w, axis_c1*axis_c0)
    output_tensor = tmp_input_tensor[:n_pad, :, :, :, :c_pad]

    return output_tensor


def calc_expect_func_zn_2_nchw(src, dst, src_format, dst_format):
    in_shape = src.get("shape")
    if -1 in in_shape:
         in_shape = src.get("typical_shape")
    input_tensor = src.get("value")
    dst_shape = dst.get("shape")

    axis_n, axis_c, axis_h, axis_w = dst_shape
    axis_c1hw, axis_no, axis_ni, axis_c0 = in_shape
    axis_c1 = axis_c1hw // (axis_h * axis_w)

    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = input_tensor.reshape(axis_c1, axis_h, axis_w, axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(3, 4, 0, 5, 1, 2))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_no*axis_ni, axis_c1*axis_c0, axis_h, axis_w)
    output_tensor = tmp_input_tensor[:n_pad, :c_pad, :, :]

    return output_tensor


def calc_expect_func_nc1hwc0_2_nchw(src, dst, src_format, dst_format):
    in_shape = src.get("shape")
    if -1 in in_shape:
         in_shape = src.get("typical_shape")
    input_tensor = src.get("value")
    dst_shape = dst.get("shape")

    axis_n, axis_c1, axis_h, axis_w, axis_c0 = in_shape
    axis_c = dst_shape[1]

    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    tmp_input_tensor = input_tensor.reshape(axis_n, axis_c1, axis_h, axis_w, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 1, 4, 2, 3))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_n, axis_c1*axis_c0, axis_h, axis_w)
    output_tensor = tmp_input_tensor[:, :c_pad, :, :]

    return output_tensor


def calc_expect_func_nc1hwc0_2_nhwc(src, dst, src_format, dst_format):
    in_shape = src.get("shape")
    if -1 in in_shape:
         in_shape = src.get("typical_shape")
    input_tensor = src.get("value")
    dst_shape = dst.get("shape")

    axis_n, axis_c1, axis_h, axis_w, axis_c0 = in_shape
    axis_c = dst_shape[3]

    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    tmp_input_tensor = input_tensor.reshape(axis_n, axis_c1, axis_h, axis_w, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_n, axis_h, axis_w, axis_c1*axis_c0)
    output_tensor = tmp_input_tensor[:, :, :, :c_pad]

    return output_tensor


def calc_expect_func_nz_2_nd(src, dst, src_format, dst_format):
    in_shape = src.get("shape")
    if -1 in in_shape:
         in_shape = src.get("typical_shape")
    input_tensor = src.get("value")
    dst_shape = dst.get("shape")

    in_len = len(in_shape)
    if in_len == 4:
        axis_h = 1
        axis_c1, axis_no, axis_ni, axis_c0 = in_shape
    else:
        axis_h = reduce(lambda x, y: x * y, in_shape[:-4])
        axis_c1, axis_no, axis_ni, axis_c0 = in_shape[-4:]

    dst_len = len(dst_shape)
    if dst_len == 1:
        axis_n, axis_c = 1, dst_shape[0]
    else:
        axis_n, axis_c = dst_shape[-2:]

    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    tmp_input_tensor = input_tensor.reshape(axis_h, axis_c1, axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_h, axis_no*axis_ni, axis_c1*axis_c0)
    output_tensor = tmp_input_tensor[:, :n_pad, :c_pad]

    return output_tensor


def calc_expect_func_hwcn_2_zng(src, dst, src_format, dst_format, groups):
     in_shape = src.get("shape")
     input_tensor = src.get("value")
     dst_shape = dst.get("shape")

     axis_h, axis_w, axis_c, axis_n = in_shape
     axis_c = axis_c * groups
     axis_c0 = dst_shape[-1]
     axis_n = 1
     axis_ni = 16
     axis_c1 = _ceil_div(axis_c, axis_c0)
     axis_no = _ceil_div(axis_n, axis_ni)
     c_pad = _pad_len(axis_c, axis_c0)
     n_pad = _pad_len(axis_n, axis_ni)

     tmp_input_tensor = input_tensor.reshape(axis_h, axis_w, axis_c, axis_n)
     tmp_input_tensor = np.pad(tmp_input_tensor, ((0, 0), (0, 0), (0, c_pad), (0, n_pad)), mode="constant", constant_values=(0, 0))
     tmp_input_tensor = tmp_input_tensor.reshape(axis_h, axis_w, axis_c1, axis_c0, axis_no, axis_ni)
     output_tensor = np.transpose(tmp_input_tensor, axes=(2, 0, 1, 4, 5, 3))

     return output_tensor


def calc_expect_func_zng_2_hwcn(src, dst, src_format, dst_format, groups):
     in_shape = src.get("shape")
     input_tensor = src.get("value")
     dst_shape = dst.get("shape")

     axis_h, axis_w, axis_c, axis_n = dst_shape
     axis_c = axis_c * groups
     axis_n = 1
     axis_c0 = dst_shape[-1]
     axis_ni = 16
     axis_c1 = _ceil_div(axis_c, axis_c0)
     axis_no = _ceil_div(axis_n, axis_ni)
     c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0
     n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni

     tmp_input_tensor = np.transpose(input_tensor, axes=(1, 2, 0, 5, 3, 4))
     tmp_input_tensor = tmp_input_tensor.reshape(axis_h, axis_w, axis_c1*axis_c0, axis_no*axis_ni)
     output_tensor = tmp_input_tensor[:, :, :c_pad, :n_pad]

     return output_tensor

