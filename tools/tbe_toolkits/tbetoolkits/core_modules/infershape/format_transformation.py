# -*- coding: UTF-8 -*-
"""
High Performance Transform Data with NumPy
Tips:
  - Chinese comments are not recommended.
  - The unified naming style is highly recommended.
  - Use "python type hints" if possible
  - Recommend to use methods in the standard library instead of reinventing-the-wheel.
"""
# Standard Packages
import copy
from typing import Tuple
from typing import List
from math import gcd
# Third-Party Packages
import numpy

BLOCK_SIZE = 16


def lcm(a, b):
    return a * b // gcd(a, b)


def align_factor(dtype: str = "float16"):
    return 16
    # bits = int(re.findall(r"\d+", dtype)[0])
    # return 256 // bits

def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]

def _ceildiv(a, b):
    return (a + b - 1) // b


def _align(a, b):
    return _ceildiv(a, b) * b


def _calculate_group(cin, cout, groups):
    mag_factor0 = lcm(cin // groups, BLOCK_SIZE) // (cin // groups)
    mag_factor1 = lcm(cout // groups, BLOCK_SIZE) // (cout // groups)
    mag_factor = min(lcm(mag_factor0, mag_factor1), groups)

    cin_g = _align(mag_factor * (cin // groups), BLOCK_SIZE)
    cout_g = _align(mag_factor * (cout // groups), BLOCK_SIZE)

    group_dict = {
        "real_g": _ceildiv(groups, mag_factor),
        "mag_factor": mag_factor,
        "cin_g": cin_g,
        "cin1_g": cin_g // BLOCK_SIZE,
        "cout_g": cout_g,
        "cout1_g": cout_g // BLOCK_SIZE,
        "groups": groups,
        "cin_ori": cin // groups,
        "cout_ori": cout // groups
    }
    print('cin:%d, cout:%d, groups:%d, group_dict:' % (cin, cout, groups),
          group_dict)
    return group_dict


def nd_shape2fhd_shape(shape,
                       nd_format: str = "NCHW",
                       dtype: str = "float16") -> Tuple:
    C0 = align_factor(dtype)
    if nd_format == "NCHW":
        C1 = _ceildiv(shape[1], C0)
        return shape[0], C1, shape[2], shape[3], C0
    if nd_format == "NHWC":
        C1 = _ceildiv(shape[3], C0)
        return shape[0], C1, shape[1], shape[2], C0


def nd_shape2nz_shape(shape: List or Tuple) -> Tuple:
    M, N = shape[-2:]
    M0 = N0 = align_factor()
    M1 = _ceildiv(M, M0)
    N1 = _ceildiv(N, N0)
    return tuple(shape[:-2] + (N1, M1, M0, N0))


def fhd2nd(data, nd_shape, nd_format: str = "NCHW"):
    pad = data.shape[4]
    if nd_format == "NCHW":
        pad = 1 + (nd_shape[1] - 1) % pad
    if nd_format == "NHWC":
        pad = 1 + (nd_shape[3] - 1) % pad
    main_block = data[:, :data.shape[1] - 1, :, :, :]  # main block
    tail_block = data[:, data.shape[1] - 1, :, :, :pad]  # tail block
    # NC1HWC0 -> NHWC1C0
    main_block = main_block.transpose((0, 2, 3, 1, 4))
    # NHWC1C0 -> NHWC
    main_block = main_block.reshape(main_block.shape[:3] + (-1,))
    # concatenate
    nhwc = numpy.concatenate((main_block, tail_block), axis=-1)
    if nd_format == "NCHW":
        return nhwc.transpose((0, 3, 1, 2))
    return nhwc


def nd2fhd(data, nd_format="NCHW"):
    fhd_shape = nd_shape2fhd_shape(data.shape, nd_format=nd_format)
    if nd_format == "NCHW":
        data = numpy.transpose(data, axes=(0, 2, 3, 1))  # pivot NCHW to NHWC
    C0 = align_factor(str(data.dtype))
    C1C0 = _ceildiv(data.shape[-1], C0) * C0
    if C1C0 > data.shape[3]:
        zero_block = numpy.zeros(data.shape[:3] + (C1C0 - data.shape[3],))
        fhd = numpy.concatenate((data, zero_block), axis=-1).reshape(
            (data.shape[0], data.shape[1], data.shape[2], -1, C0))
    else:
        fhd = data.reshape((data.shape[0], data.shape[1], data.shape[2], -1, C0))
    fhd = numpy.transpose(fhd, axes=(0, 3, 1, 2, 4))
    return fhd


def nz2nd(data, nd_shape):
    """
    Convert FRACTAL_NZ format to ND format
    (A0, A1, A2, ..., An, N1, M1, M0, N0) -> (A, N1, M1, M0, N0) -> (A, M1, M0, N1, N0)
    (A, M1, M0, N1, N0) -> (A, [0, M1-2], M0, [0, N1-2], N0) == (A, (M1-1), M0, (N1-1), N0)
                                                             -> (A, (M1-1) *M0, (N1-1) *N0)
                        -> (A, [M1-1], [0, pad_m-1], [0, N1-2], N0) == (A, 1, pad_m, (N1-1), N0)
                                                                    -> (A, pad_m, (N1-1) *N0)
                        -> (A, [0, M1-2], M0, [N1-1], [0, pad_n-1]) == (A, (M1-1), M0, 1, pad_n)
                                                                    -> (A, (M1-1) *M0, pad_n)
                        -> (A, [M1-1], [0, pad_m-1], [N1-1], [0, pad_n-1]) == (A, 1, pad_m, 1, pad_n)
                                                                           -> (A, pad_m, pad_n)
    M = (M1-1) *M0 + pad_m
    N = (N1-1) *N0 + pad_n
    (A, (M1-1) *M0, (N1-1) *N0) + (A, pad_m, (N1-1) *N0) -> (A, (M1-1) *M0 + pad_m, (N1-1) *N0) -> (A, M, (N1-1) *N0)
    (A, (M1-1) *M0, pad_n) + (A, pad_m, pad_n) -> (A, (M1-1) *M0 + pad_m, pad_n) -> (A, M, pad_n)
    (A, M, (N1-1) *N0) + (A, M, pad_n) -> (A, M, (N1-1) *N0 + pad_n) -> (A, M, N)
    (A, M, N) -> (A0, A1, A2, ..., An, M, N)
    """
    ori_nz_shape = data.shape
    ori_nd_shape = copy.deepcopy(nd_shape)
    if len(data.shape) == 4:
        data = numpy.reshape(data, (1,) + data.shape)
    nd_shape = (1,) + tuple(nd_shape)
    data_shape = data.shape
    m, n = nd_shape[-2:]
    pad_m = 1 + (m - 1) % align_factor()
    pad_n = 1 + (n - 1) % align_factor()
    N1, M1 = data_shape[-4:-2]
    M0 = N0 = align_factor()
    # (A0, A1, A2, ... , An, N1, M1, M0, N0) -> (A, N1, M1, M0, N0) -> (A, M1, M0, N1, N0)
    data = numpy.reshape(data, (numpy.prod(data_shape[:-4]),) + data_shape[-4:]).transpose((0, 2, 3, 1, 4))
    main_block = data[:, :M1 - 1, :, :N1 - 1, :]  # main block
    part_1 = data[:, M1 - 1, :pad_m, :N1 - 1, :]  # part 1
    part_2 = data[:, :M1 - 1, :, N1 - 1, :pad_n]  # part 2
    tail_block = data[:, M1 - 1, :pad_m, N1 - 1, :pad_n]  # tail_block
    # Reshape
    A = data.shape[0]
    main_block = numpy.reshape(main_block, (A, (M1 - 1) * M0, (N1 - 1) * N0))
    part_1 = numpy.reshape(part_1, (A, pad_m, (N1 - 1) * N0))
    part_2 = numpy.reshape(part_2, (A, (M1 - 1) * M0, pad_n))
    tail_block = numpy.reshape(tail_block, (A, pad_m, pad_n))
    # Concatenate
    main_concat_part1 = numpy.concatenate((main_block, part_1), axis=1)  # (A, M, (N1-1) *N0)
    part_2_concat_tail = numpy.concatenate((part_2, tail_block), axis=1)  # (A, M, pad_n)
    nd = numpy.concatenate((main_concat_part1, part_2_concat_tail), axis=-1)
    # Reshape
    nd = numpy.reshape(nd, data_shape[:-4]+(m, n))
    nd = numpy.reshape(nd, ori_nd_shape)

    return nd


def nd2nz(data):
    """
    Convert ND format to FRACTAL_NZ format
    (A0, A1, A2, ..., An, M, N) -> (A, M, N)
    (A, M, N) -> (A, M, (N1-1) *N0 + pad_n) -> (A, M, (N1-1) *N0) -> (A, (M1-1) *M0, (N1-1) *N0)
                                                                  -> (A, pad_m, (N1-1) *N0)
                                            -> (A, M, pad_n)      -> (A, (M1-1) *M0, pad_n)
                                                                  -> (A, pad_m, pad_n)
    (A, (M1-1) *M0, pad_n)      -> (A, (M1-1), M0, 1, pad_n)   -> (A, [0, M1-2], M0, [N1-1], [0, pad_n-1])
                                                               => (A, [0, M1-2], M0, [N1-1], N0)
    (A, (M1-1) *M0, (N1-1) *N0) -> (A, (M1-1), M0, (N1-1), N0) -> (A, [0, M1-2], M0, [0, N1-2], N0)
                                                              ==> (A, [0, M1-2], M0, N1, N0)
    (A, pad_m, pad_n)           -> (A, 1, pad_m, 1, pad_n)     -> (A, [M1-1], [0, pad_m-1], [N1-1], [0, pad_n-1])
                                                               => (A, [M1-1], [0, pad_m-1], [N1-1], N0)
    (A, pad_m, (N1-1) *N0)      -> (A, 1, pad_m, (N1-1), N0)   -> (A, [M1-1], [0, pad_m-1], [0, N1-2], N0)
                                                              ==> (A, [M1-1], [0, pad_m-1], N1, N0)
    (A, [M1-1], [0, pad_m-1], N1, N0) => (A, [M1-1], M0, N1, N0)
    (A, [0, M1-2], M0, N1, N0)        == (A, [0, M1-2], M0, N1, N0)
                                     ==> (A, M1, M0, N1, N0)
    """
    ori_nd_shape = data.shape
    if len(ori_nd_shape) <= 2:
        data = numpy.reshape(data, (1,) * (3 - len(ori_nd_shape)) + ori_nd_shape)
    data_shape = data.shape
    nz_shape = nd_shape2nz_shape(data_shape)
    A = numpy.prod(data_shape[:-2])
    M, N = data_shape[-2:]
    pad_m = 1 + (M - 1) % align_factor()
    pad_n = 1 + (N - 1) % align_factor()
    N1, M1 = nz_shape[-4:-2]
    M0 = N0 = align_factor()
    # (A0, A1, A2, ..., An, M, N) -> (A, M, N)
    data = numpy.reshape(data, (A, M, N))
    # (A, M, N) -> (A, M, (N1-1) *N0 + pad_n) -> (A, M, (N1-1) *N0) -> (A, (M1-1) *M0, (N1-1) *N0)
    #                                                               -> (A, pad_m, (N1-1) *N0)
    main_concat_part1 = data[:, :, :(N1 - 1) * N0]  # -> (A, M, (N1-1) *N0)
    main_block = main_concat_part1[:, :(M1 - 1) * M0, :]  # (A, (M1-1) *M0, (N1-1) *N0)
    part_1 = main_concat_part1[:, (M1 - 1) * M0:, :]  # (A, pad_m, (N1-1) *N0)
    # (A, M, N) -> (A, M, (N1-1) *N0 + pad_n) -> (A, M, pad_n) -> (A, (M1-1) *M0, pad_n)
    #                                                          -> (A, pad_m, pad_n)
    part_2_concat_tail = data[:, :, (N1 - 1) * N0:]  # -> (A, M, pad_n)
    part_2 = part_2_concat_tail[:, :(M1 - 1) * M0, :]  # (A, (M1-1) *M0, pad_n)
    tail_block = part_2_concat_tail[:, (M1 - 1) * M0:, :]  # (A, pad_m, pad_n)

    # (A, (M1-1) *M0, (N1-1) *N0) -> (A, (M1-1), M0, (N1-1), N0) -> (A, [0, M1-2], M0, [0, N1-2], N0)
    main_block = numpy.reshape(main_block, (A, (M1 - 1), M0, (N1 - 1), N0))  # (A, (M1-1), M0, (N1-1), N0)
    # (A, pad_m, (N1-1) *N0)      -> (A, 1, pad_m, (N1-1), N0)   -> (A, [M1-1], [0, pad_m-1], [0, N1-2], N0)
    part_1 = numpy.reshape(part_1, (A, 1, pad_m, (N1 - 1), N0))  # (A, 1, pad_m, (N1-1), N0)
    # (A, (M1-1) *M0, pad_n)      -> (A, (M1-1), M0, 1, pad_n)   -> (A, [0, M1-2], M0, [N1-1], [0, pad_n-1])
    part_2 = numpy.reshape(part_2, (A, (M1 - 1), M0, 1, pad_n))  # (A, (M1-1), M0, 1, pad_n)
    # (A, pad_m, pad_n)           -> (A, 1, pad_m, 1, pad_n)     -> (A, [M1-1], [0, pad_m-1], [N1-1], [0, pad_n-1])
    tail_block = numpy.reshape(tail_block, (A, 1, pad_m, 1, pad_n))  # (A, 1, pad_m, 1, pad_n)

    # (A, [0, M1-2], M0, [N1-1], [0, pad_n-1]) => (A, [0, M1-2], M0, [N1-1], N0)
    part_2_pad = numpy.concatenate((part_2, numpy.zeros((A, M1-1, M0, 1, N0-pad_n))), axis=-1)
    # (A, [0, M1-2], M0, [0, N1-2], N0) + (A, [0, M1-2], M0, [N1-1], N0) ==> (A, [0, M1-2], M0, N1, N0)
    main_block_concat_part_2_pad = numpy.concatenate((main_block, part_2_pad), axis=-2)
    # (A, [M1-1], [0, pad_m-1], [N1-1], [0, pad_n-1]) => (A, [M1-1], [0, pad_m-1], [N1-1], N0)
    tail_block_pad = numpy.concatenate((tail_block, numpy.zeros((A, 1, pad_m, 1, N0-pad_n))), axis=-1)
    # (A, [M1-1], [0, pad_m-1], [0, N1-2], N0) + (A, [M1-1], [0, pad_m-1], [N1-1], N0)
    #                                            ==> (A, [M1-1], [0, pad_m-1], N1, N0)
    part_1_concat_tail_block_pad = numpy.concatenate((part_1, tail_block_pad), axis=-2)

    # (A, [M1-1], [0, pad_m-1], N1, N0) => (A, [M1-1], M0, N1, N0)
    part_1_concat_tail_block_pad_pad \
        = numpy.concatenate((part_1_concat_tail_block_pad, numpy.zeros((A, 1, M0-pad_m, N1, N0))), axis=2)
    # (A, [0, M1-2], M0, N1, N0) + (A, [M1-1], M0, N1, N0) ==> (A, M1, M0, N1, N0)
    nz = numpy.concatenate((main_block_concat_part_2_pad, part_1_concat_tail_block_pad_pad), axis=1)
    # (A, M1, M0, N1, N0) -> (A, N1, M1, M0, N0) -> (A0, A1, ..., An, N1, M1, M0, N0)
    nz = numpy.transpose(nz, (0, 3, 1, 2, 4)).reshape(data_shape[:-2]+(N1, M1, M0, N0))
    if len(ori_nd_shape) <= 2:
        nz = numpy.reshape(nz, nz.shape[-4:])

    return nz


def to_fractal_z(filter_data, filter_format, groups):
    shape_filter = filter_data.shape
    # data_format:NCHW or NHWC
    filter_n = shape_filter[filter_format.index("N")]
    filter_c = shape_filter[filter_format.index("C")]
    filter_h = shape_filter[filter_format.index("H")]
    filter_w = shape_filter[filter_format.index("W")]
    c_in = filter_c * groups
    c_out = filter_n
    group_dict = _calculate_group(c_in, c_out, groups)
    G = group_dict["real_g"]
    ci_ori = group_dict["cin_ori"]
    co_ori = group_dict["cout_ori"]
    cin1_g = group_dict["cin1_g"]
    cou1_g = group_dict["cout1_g"]
    E = group_dict["mag_factor"]
    # Initialization
    out_filter = numpy.zeros(
        [G * cou1_g * 16, cin1_g * 16, filter_h,
         filter_w]).astype(filter_data.dtype)
    filter_data = filter_data.transpose(filter_format.index("N"),
                                        filter_format.index("C"),
                                        filter_format.index("H"),
                                        filter_format.index("W"))
    for m in range(groups):
        for k in range(co_ori):
            for l in range(0, ci_ori):
                i = m // E
                j = m % E
                out_filter[i * E * co_ori + j * co_ori + k, j * ci_ori + l, :, :] = \
                    filter_data[i * E * co_ori + j * co_ori + k, l, :, :]
    # nchw->FRACTAL_Z
    out_filter = out_filter.reshape((G, cou1_g * 16, cin1_g, BLOCK_SIZE, filter_h, filter_w)
                                    ).transpose(0, 2, 4, 5, 1, 3)

    return out_filter


def to_fractal_z_3d(filter_data, filter_format, groups):
    shape_filter = filter_data.shape
    # data_format: 'NCDHW' or 'NDHWC'
    w_batch = shape_filter[filter_format.index("N")]
    filter_c = shape_filter[filter_format.index("C")]
    w_h = shape_filter[filter_format.index("H")]
    w_w = shape_filter[filter_format.index("W")]
    w_d = shape_filter[filter_format.index("D")]
    fmap_c = filter_c * groups
    out_c = w_batch
    group_dict = _calculate_group(fmap_c, out_c, groups)
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    mag_factor = group_dict["mag_factor"]
    weight_group = numpy.zeros(
        (real_g, w_d, cin1_g, w_h, w_w, cout_g, BLOCK_SIZE),
        dtype=filter_data.dtype)
    filter_data = filter_data.transpose(filter_format.index("N"),
                                        filter_format.index("C"),
                                        filter_format.index("D"),
                                        filter_format.index("H"),
                                        filter_format.index("W"))
    for g in range(groups):
        for ci in range(filter_c):
            for co in range(w_batch // groups):
                try:
                    e = g % mag_factor
                    dst_cin = e * filter_c + ci
                    dst_cout = e * (w_batch // groups) + co
                    src_cout = g * (w_batch // groups) + co
                    weight_group[g // mag_factor, :, dst_cin // BLOCK_SIZE, :, :, dst_cout, dst_cin % BLOCK_SIZE] = \
                        filter_data[src_cout, ci, :, :, :]
                except:
                    e = g % mag_factor
                    dst_cin = e * filter_c + ci
                    dst_cout = e * (w_batch // groups) + co
                    src_cout = g * (w_batch // groups) + co
                    print(
                        "================================== Error Detected ======================================="
                    )
                    print("weight_group shape:", weight_group.shape)
                    print("Weight Shape : ", filter_data.shape)
                    print("C0:", co)
                    print("e : ", e)
                    print("dst_cin :", dst_cin)
                    print("dst_cout : ", dst_cout)
                    print("src_cout and Ci", src_cout, "", ci)
                    print("mag_factor : ", mag_factor)
                    raise
    return weight_group


def to_NC1HWC0(data, ori_format):
    ori_shape = data.shape
    c_ori = ori_shape[ori_format.index("C")]
    c1 = _ceildiv(c_ori, BLOCK_SIZE)
    N = ori_shape[ori_format.index("N")]
    H = ori_shape[ori_format.index("H")]
    W = ori_shape[ori_format.index("W")]
    data = data.transpose(ori_format.index("N"), ori_format.index("C"),
                          ori_format.index("H"), ori_format.index("W"))

    num_2_padding_in_Cin = c1 * BLOCK_SIZE - c_ori
    zero_padding_array = numpy.zeros((N, num_2_padding_in_Cin, H, W))
    data = numpy.concatenate((data, zero_padding_array), axis=1)
    data = data.reshape((N, c1, BLOCK_SIZE, H, W)).transpose(0, 1, 3, 4, 2)
    return data


def to_NDC1HWC0(data, ori_format):
    ori_shape = data.shape
    c_ori = ori_shape[ori_format.index("C")]
    c1 = _ceildiv(c_ori, BLOCK_SIZE)
    N = ori_shape[ori_format.index("N")]
    D = ori_shape[ori_format.index("D")]
    H = ori_shape[ori_format.index("H")]
    W = ori_shape[ori_format.index("W")]
    data = data.transpose(ori_format.index("N"), ori_format.index("C"),
                          ori_format.index("D"), ori_format.index("H"),
                          ori_format.index("W"))

    num_2_padding_in_Cin = c1 * BLOCK_SIZE - c_ori
    zero_padding_array = numpy.zeros((N, num_2_padding_in_Cin, D, H, W))
    data = numpy.concatenate((data, zero_padding_array), axis=1)
    data = data.reshape((N, c1, BLOCK_SIZE, D, H, W)).transpose(0, 3, 1, 4, 5, 2)
    return data


def nd_to_fractal_nz(data, ori_format):
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0,0),) * batch_num
    if data.dtype == 'int8' :
        m0, n0 = 16, 32
        m1 = _ceildiv(m_ori, m0)
        n1 = _ceildiv(n_ori, n0)
        
    else:
        m0, n0 = 16, 16
        m1 = _ceildiv(m_ori, m0)
        n1 = _ceildiv(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = numpy.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


def nd_to_fractal_z(data, ori_format):
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0,0),) * batch_num
    if data.dtype == 'int8' :
        m0, n0 = 32, 16
        m1 = _ceildiv(m_ori, m0)
        n1 = _ceildiv(n_ori, n0)   
    else:
        m0, n0 = 16, 16
        m1 = _ceildiv(m_ori, m0)
        n1 = _ceildiv(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = numpy.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [0, 2, 3, 1])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data

