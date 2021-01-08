# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
conv3d compute
"""
import copy
import te.platform as tbe_platform
from te.domain.tiling.get_tiling import get_tiling
from te.utils.error_manager import error_manager_util
from te.utils import shape_util
from te.lang.cce.te_compute import cube_util
from te.lang.cce.te_compute import util as te_util
from te.lang.base.operation_impl import get_te_var
from te import tvm

_OP_TAG = "conv3d_"
_TENSOR_MAP = {}
_DIM_MAP = {}
_NAME_INDEX = [0]

# filterD must be in [1,255]
_FILTER_DHW_MIN = 1
_FILTER_DHW_MAX = 255
# pad must be in [0,255]
_PAD_MIN = 0
_PAD_MAX = 255
# stride must be in [1,63]
_STRIDE_MIN = 1
_STRIDE_MAX = 63

# fmap H and W must be in [1, 4096]
_FMAP_HW_MIN = 1
_FMAP_HW_MAX = 4096
# dilations must be in [1,255]
_DILATION_MIN = 1
_DILATION_MAX = 255


class Conv3DParam(object):
    """
    ConvParam
    """

    def __init__(self):
        pass

    def _get_tensor_map(self):
        """
         get the tensor_map in convparam
        """
        return self._TENSOR_MAP

    _TENSOR_MAP = {}
    dim_map = {}
    tiling = None
    tiling_query_param = {}
    var_map = {}
    dynamic_mode = None
    tiling_info_dict = {}


def _cube_3d_compute(fmap,
                     weight,
                     mad_dtype,
                     res_dtype,
                     pads,
                     stride_dhw,
                     dilation_dhw,
                     shape_filter_ncdhw,
                     cyclebuffer_flag,
                     group_dict,
                     bias=None):
    """
    conv

    Parameters
    ----------
    fmap : tvm.tensor, Feature Map

    weight: tvm.tensor, Filter

    mad_dtype : the compute data type

    res_dtype : the result data type

    pads: the padding shape
        [head, tail, top, bottom, left, right]

    stride_dhw: the stride value
        [stride_d, stride_h, stride_w]

    dilation_dhw: the dilation value
        [dilation_d, dilation_h, dilation_w]

    shape_filter_ncdhw: the filter shape

    bias: the tag for bias or not

    tiling: default none, tiling

    group_dict: the information needed for group convolution

    -------
    Returns

    wrapped_tensor
    """
    in_dtype = fmap.dtype
    w_dtype = weight.dtype

    _TENSOR_MAP["fmap"] = fmap
    _TENSOR_MAP["filter"] = weight

    fmap_shape = shape_util.shape_to_list(fmap.shape)
    batch_size = fmap_shape[0]
    fmap_d = fmap_shape[1]
    fmap_c1 = fmap_shape[2]
    fmap_h = fmap_shape[3]
    fmap_w = fmap_shape[4]
    fmap_c0 = fmap_shape[5]
    filter_cout, _, filter_d, filter_h, filter_w = shape_filter_ncdhw
    pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right = pads
    stride_d, stride_h, stride_w = stride_dhw
    dilation_d, dilation_h, dilation_w = dilation_dhw

    _TENSOR_MAP["filter_d"] = filter_d
    if Conv3DParam.dynamic_mode == "dynamic_dhw":
        height_out = Conv3DParam.var_map.get("h_out")
        width_out = Conv3DParam.var_map.get("w_out")
        d_out = Conv3DParam.var_map.get("d_out")
    else:
        height_out = (fmap_h + pad_top + pad_bottom - ((filter_h - 1) * dilation_h + 1)) // stride_h + 1
        width_out = (fmap_w + pad_left + pad_right - ((filter_w - 1) * dilation_w + 1)) // stride_w + 1
        d_out = (fmap_d + pad_head + pad_tail - ((filter_d - 1) * dilation_d + 1)) // stride_d + 1

    config = tbe_platform.CUBE_MKN[in_dtype]
    block_size_k = config['mac'][1]
    block_size_m = config['mac'][0]
    opti_h_flag = filter_h == 1 and stride_h > 1
    _TENSOR_MAP["opti_h_flag"] = opti_h_flag
    _TENSOR_MAP["d_out"] = d_out

    _TENSOR_MAP["group_dict"] = group_dict
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    cin_ori = group_dict["cin_ori"]
    cout_ori = group_dict["cout_ori"]
    fmap_fuse_shape = (batch_size * d_out, filter_d * fmap_c1, fmap_h, fmap_w,
                       fmap_c0)
    fuse_fmap_tensor = _get_fuse_fmap_tensor(fmap_fuse_shape,
                                             fmap,
                                             d_out,
                                             filter_d,
                                             stride_d,
                                             stride_h,
                                             pad_head,
                                             opti_h_flag,
                                             cyclebuffer_flag,
                                             tag=_OP_TAG)

    _TENSOR_MAP["fmap_do_tensor"] = fuse_fmap_tensor

    # im2col
    # small-z-big-Z
    howo_mad = (height_out * width_out + block_size_m -
                1) // block_size_m * block_size_m
    pad_hw = pads[2:]

    if Conv3DParam.dynamic_mode:
        # new data layout (N, C1, H, W, C0) -> (N, loop_m, loop_k, cube_m, cube_k)
        howo_mad_1 = (height_out * width_out + block_size_m -
                    1) // block_size_m
        fmap_im2col_fractal_shape = (real_g, fmap_fuse_shape[0],
                    howo_mad_1, filter_d * cin1_g * filter_h * filter_w,
                    block_size_m, block_size_k)
        stride_hw = stride_dhw[1:]
        if opti_h_flag:
            stride_hw[0] = 1
        im2col_para = (fuse_fmap_tensor, filter_h, filter_w, pad_hw, stride_hw,
                       width_out, 1, cin_ori)
        fmap_im2col_fractal_res = cube_util.im2col_fractal_v2(
                    fmap_im2col_fractal_shape, im2col_para)
        _TENSOR_MAP["fmap_im2col_fractal_res"] = fmap_im2col_fractal_res
    else:
        # set_fmatrix
        # new data layout (N,C1,H,W,C0) -> (N,HoWo,C1,Hk,Wk,C0)
        fmap_im2col_row_major_shape = (fmap_fuse_shape[0], height_out * width_out,
                                       fmap_fuse_shape[1], filter_h, filter_w,
                                       fmap_c0)
        stride_hw = [stride_h, stride_w]
        dilation_hw = [dilation_h, dilation_w]
        fmap_im2col_row_major_res = cube_util.im2col_row_major(fmap_im2col_row_major_shape,
                                                               fuse_fmap_tensor,
                                                               filter_w,
                                                               pad_hw,
                                                               stride_hw,
                                                               fmap.dtype,
                                                               opti_h_flag,
                                                               tag=_OP_TAG,
                                                               dilation=dilation_hw)
        _TENSOR_MAP["fmap_im2col_row_major_res"] = fmap_im2col_row_major_res

        # im2col
        # small-z-big-Z
        # new data layout (N,HoWo,C1,Hk,Wk,C0) -> (N,loop_m,loop_k,cube_m,cube_k)
        fmap_im2col_fractal_shape = (real_g, fmap_fuse_shape[0],
                                     howo_mad // block_size_m,
                                     filter_d * cin1_g * filter_h * filter_w,
                                     block_size_m, block_size_k)

        fmap_im2col_fractal_res = cube_util.im2col_fractal_3d(
            fmap_im2col_fractal_shape,
            fmap_im2col_row_major_res,
            fmap_c1,
            d_out,
            filter_d,
            stride_d,
            cin1_g,
            cyclebuffer_flag,
            tag=_OP_TAG)
        _TENSOR_MAP["fmap_im2col_fractal_res"] = fmap_im2col_fractal_res

    config = tbe_platform.CUBE_MKN[res_dtype]

    l0a_load2d_flag = _TENSOR_MAP["l0a_load2d_flag"]

    mad_shape = (real_g, fmap_fuse_shape[0],
                 (cout_g + config['mac'][2] - 1) // (config['mac'][2]),
                 howo_mad, config['mac'][2])

    config = tbe_platform.CUBE_MKN[w_dtype]

    if l0a_load2d_flag:
        c_col = _mad_by_load2d(mad_shape, fmap, weight, config, mad_dtype,
                               pads, stride_d, d_out, group_dict)
    else:
        c_col = _mad(mad_shape, fmap_im2col_fractal_res, weight, config,
                     mad_dtype, pads, stride_d, d_out, fmap_d, real_g)

    _TENSOR_MAP["c_col"] = c_col

    conv_shape = (fmap_fuse_shape[0],
                  (cout_ori + config['mac'][2] - 1) // (config['mac'][2]),
                  height_out * width_out, config['mac'][2])

    _DIM_MAP["out_img_shape"] = conv_shape
    cout1_g = cout_g // config['mac'][2]

    attrs_dict = {'true_shape': conv_shape,
                  'sqrt': False,
                  'res_dtype': res_dtype,
                  'kernel_h': filter_h,
                  'kernel_w': filter_w,
                  'kernel_d': filter_d,
                  'padding': pads[2:],
                  'stride': stride_dhw[1:],
                  'dilation': dilation_dhw[1:]}

    conv_aligned_shape = (fmap_fuse_shape[0],
                  (cout_ori + config['mac'][2] - 1) // (config['mac'][2]),
                  howo_mad, config['mac'][2])
    c_ub = tvm.compute(conv_aligned_shape,
                        lambda n, i, j, k: c_col(i // cout1_g, n,
                                                 i % cout1_g, j, k).astype(
                            res_dtype),
                        name='C_UB',
                        tag=_OP_TAG + "C_UB",
                        attrs=attrs_dict)

    _TENSOR_MAP["c_ub"] = c_ub
    dim_map1 = _im2col_dim(shape_util.shape_to_list(fuse_fmap_tensor.shape),
                           shape_filter_ncdhw, list(pads), list(stride_dhw),
                           list(dilation_dhw), config)
    dim_map_copy = _DIM_MAP.copy()
    dim_map_copy.update(dim_map1)

    Conv3DParam._TENSOR_MAP = _TENSOR_MAP
    Conv3DParam.dim_map = dim_map_copy
    Conv3DParam.tiling = None
    res = c_ub

    if isinstance(bias, tvm.tensor.Tensor):
        res = _bias_add(c_ub, bias, attrs_dict)

    return res


def _get_fuse_fmap_tensor(fmap_fuse_shape, fmap, d_out, kernel_d, stride_d,
                          stride_h, pad_head, opti_h_flag,
                          cyclebuffer_flag, tag):
    """
    calculate expand tensor
    Parameters

    ----------
    fmap_fuse_shape : the shape of new tensor

    fmap : the input feature

    d_out : the D dimension of out shape

    stride_d : the D dimension of strides

    stride_h : the H dimension of strides

    pad_head : the pad head of pads

    tiling : the tiling of Conv3D

    opti_h_flag : the flag for optimizing on h dimension

    cyclebuffer_flag : the flag for cyclebuffer

    tag : the tensor tag

    Returns
    -------
    new tensor
    """
    _, fmap_d, fmap_c1, _, _, _ = fmap.shape
    # multi core
    d_dim = tvm.var(name='d_dim', dtype='int')
    _TENSOR_MAP["d_dim"] = d_dim
    opti_h_factor = 1
    if opti_h_flag:
        fmap_fuse_shape = list(fmap_fuse_shape)
        fmap_fuse_shape[2] = (fmap_fuse_shape[2] - 1) // stride_h + 1
        opti_h_factor = stride_h

    def __get_fuse_tensor_indices(indices):
        """
        return the indices of the fuse_fmap
        """
        n_index, dc_index, h_index, w_index, c0_index = indices

        batch_index = n_index // d_out
        d_index = n_index % d_out * stride_d + (dc_index // fmap_c1 + \
                  n_index % d_out * (kernel_d - stride_d) * cyclebuffer_flag) % \
                  kernel_d - pad_head
        c1_index = dc_index % fmap_c1

        return tvm.select(tvm.all(d_index >= 0, d_index < fmap_d,
                                  tvm.any(tvm.floordiv(cyclebuffer_flag, 1) == 0,
                                          d_index + pad_head > (n_index % d_out - 1) * stride_d + kernel_d - 1,
                                          tvm.floormod(n_index, tvm.floordiv(d_out, d_dim)) == 0)),
                                  fmap(batch_index,
                                       d_index,
                                       c1_index,
                                       h_index * opti_h_factor,
                                       w_index,
                                       c0_index))

    return tvm.compute(fmap_fuse_shape,
                       lambda *indices: __get_fuse_tensor_indices(indices),
                       name="fuse_fmap_tensor",
                       tag=tag + "fuse_fmap_tensor")

def _mad_by_load2d(mad_shape, fmap, weight, config, mad_dtype, pads, stride_d,
                   d_out, group_dict):
    """
    calculate mad
    Parameters

    ----------
    mad_shape : the shape of new tensor

    fmap : the input feature

    weight : the input filter

    config : the MKN config

    mad_dtype : the compute dtype of mad

    pads : the pad of Conv3D

    stride_d : the stride on d dimension

    d_out : the output shape on d dimension

    filter_d : the filter on d dimension

    Returns
    -------
    new tensor
    """
    fmap_shape = shape_util.shape_to_list(fmap.shape)
    batch_size = fmap_shape[0]
    fmap_d = fmap_shape[1]
    fmap_c1 = fmap_shape[2]
    fmap_h = fmap_shape[3]
    fmap_w = fmap_shape[4]
    fmap_c0 = fmap_shape[5]
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cin_ori = group_dict["cin_ori"]

    shape_al1_load2d = (batch_size * fmap_d, fmap_c1, fmap_h * fmap_w, fmap_c0)
    al1_load2d = tvm.compute(
        shape_al1_load2d,
        lambda n, c1, m, c0: fmap(n // fmap_d, n % fmap_d, c1, m // fmap_w, m %
                                  fmap_w, c0),
        name=_OP_TAG + "al1_load2d")
    _TENSOR_MAP["al1_load2d"] = al1_load2d

    hw_dim = te_util.int_ceil_div(fmap_h * fmap_w,
                                  tbe_platform.CUBE_MKN[fmap.dtype]["mac"][0])

    shape_al0_load2d = (real_g, batch_size * fmap_d, hw_dim, cin1_g,
                        tbe_platform.CUBE_MKN[fmap.dtype]["mac"][0],
                        fmap_c0)

    al0_load2d = tvm.compute(
        shape_al0_load2d,
        lambda g, n, m1, c1, m0, c0:
        al1_load2d(n, g * cin1_g + c1,
                   m0 + tbe_platform.CUBE_MKN[fmap.dtype]["mac"][0] * m1,
                   c0),
        name=_OP_TAG + "al0_load2d")

    _TENSOR_MAP["al0_load2d"] = al0_load2d

    c_col = _mad(mad_shape, al0_load2d, weight, config, mad_dtype, pads,
                 stride_d, d_out, fmap_d, real_g)
    return c_col


def _get_load2d_flag(stride, pads, shape_filter_ncdhw):
    """
    calculate use load2d or not
    Parameters

    ----------
    stride : the input strides

    pads : the input pads

    shape_filter_ncdhw : the shape of filter

    Returns
    -------
    True or False
    """
    l0a_load2d_flag = False
    _, _, filter_d, filter_h, filter_w = shape_filter_ncdhw

    if (list(pads) == [0, 0, 0, 0, 0, 0]
        and list(stride) == [1, 1, 1]
        and [filter_d, filter_h, filter_w] == [1, 1, 1]):
        l0a_load2d_flag = True
    return l0a_load2d_flag


def _im2col_dim(shape_fmap, shape_filter_ncdhw, pads, stride_dhw, dilation_dhw, config):
    """
    calculate shape
    Parameters

    ----------
    shape_fmap : shape of feature

    shape_filter_ncdhw : shape of filter

    pads : the padding shape

    stride_dhw : the stride value

    config : the MKN infor

    Returns
    -------
    img_shape, fmap_matrix_dim
    """
    mac_dim = config['mac']

    batch, fmap_c1, fmap_h, fmap_w, fmap_c0 = shape_fmap
    filter_cout, _, _, filter_h, filter_w = shape_filter_ncdhw
    _, _, pad_top, pad_bottom, pad_left, pad_right = pads
    _, dilation_h, dilation_w = dilation_dhw

    out_h = (fmap_h + pad_top + pad_bottom - ((filter_h - 1) * dilation_h + 1)) // stride_dhw[1] + 1
    out_w = (fmap_w + pad_left + pad_right - ((filter_w - 1) * dilation_w + 1)) // stride_dhw[2] + 1

    fmap_valid_dim = (batch, out_h * out_w,
                      fmap_c1 * filter_h * filter_w * fmap_c0)

    fmap_matrix_dim = (fmap_valid_dim[0],
                       ((fmap_valid_dim[-2] + mac_dim[0] - 1) // mac_dim[0]),
                       ((fmap_valid_dim[-1] + mac_dim[1] - 1) // mac_dim[1]),
                       mac_dim[0], mac_dim[1])

    filter_valid_dim = (fmap_valid_dim[-1], filter_cout)

    filter_matrix_dim = ((filter_valid_dim[-2] + mac_dim[1] - 1) // mac_dim[1],
                         (filter_valid_dim[-1] + mac_dim[2] - 1) // mac_dim[2],
                         mac_dim[2], mac_dim[1])

    return {
        "img_shape": shape_fmap,
        "fmap_matrix_dim": fmap_matrix_dim,
        "filter_matrix_dim": filter_matrix_dim,
        "shape_filter_ncdhw": shape_filter_ncdhw
    }


def _mad(mad_shape, fmap, weight, config, mad_dtype, pads, stride_d, d_out,
         fmap_d, real_g=1):
    """
    calculate mad result tensor
    Parameters
    ----------
    mad_shape : shape of mad result

    fmap : feature map

    weight : filter

    config: the config of cube

    mad_dtype: dtype of mad output

    pads: input pad

    stride_d: stride for d channel

    d_out: output d channel

    fmap_d: input fmap d channel

    filter_d: input filter d channel

    Returns
    -------
    mad result tensor
    """
    block_size = config['mac'][1]
    block_size_m = config['mac'][0]
    pad_head = pads[0]

    shape_w = list(i.value for i in weight.shape)
    ckk = shape_w[0] // real_g

    axis_k1 = tvm.reduce_axis((0, ckk), name='k1')
    axis_k0 = tvm.reduce_axis((0, block_size), name='k0')

    if mad_dtype in ["float16", "int32"]:
        mode = 'f162f16'
    else:
        mode = 'f162f32'
    c_col = tvm.compute(
        mad_shape,
        lambda g, n, index_j1, i, index_j0: tvm.sum(
            (fmap[g, n, i // block_size_m, axis_k1, i % block_size_m, axis_k0] *
             weight[g * ckk + axis_k1, index_j1, index_j0, axis_k0]).astype(mad_dtype),
            axis=[axis_k1, axis_k0]),
        name='mad1',
        tag=_OP_TAG + "c_col",
        attrs={
            'mode': mode,
            'pad_head': pad_head,
            'fmap_d': fmap_d,
            'stride_d': stride_d,
            'd_out': d_out
        })
    return c_col


def _bias_add(in_tensor0, in_tensor1, attrs={}):
    """
    calculate conv res + bias in UB
    Parameters
    ----------
    in_tensor0: cnv res tensor

    in_tensor1: bias vector

    Returns
    -------
    in_tensor0+in_tensor1 tensor
    """
    dim_map = {}
    dim_map["out_img_shape"] = shape_util.shape_to_list(in_tensor0.shape)
    _NAME_INDEX[0] += 1

    with tvm.tag_scope('conv_vector_bias_add'):
        c_add_vector = tvm.compute(
            dim_map["out_img_shape"],
            lambda *indice: in_tensor0(*indice)
                            + in_tensor1(indice[1]
                            * tbe_platform.CUBE_MKN[in_tensor0.dtype]['mac'][2]
                            +indice[3]),
            name='bias_add_vector' + "_cc_" + str(_NAME_INDEX[0]),
            attrs=attrs)

    return c_add_vector


def _loc_bias_add(c_col, bias, conv_shape, group_dict, w_dtype=None):
    """
    calculate conv res + bias in l0c
    Parameters
    ----------
    c_col: fmap & filter matrix multi res tensor

    bias: bias vector

    Returns
    -------
    c_col_bias tensor
    """
    block_size = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]

    cout_g = group_dict["cout_g"]
    cout1_g = cout_g // block_size


    bias_ub_brc_tensor = tvm.compute(conv_shape,
                                     lambda i, j, k, l: bias(
                                         j * block_size + l),
                                     name=_OP_TAG + 'bias_ub_brc')
    _TENSOR_MAP["bias_ub_brc"] = bias_ub_brc_tensor

    bias_l0c = tvm.compute(conv_shape,
                           lambda *indices: bias(*indices),
                           name=_OP_TAG + 'bias_l0c')
    _TENSOR_MAP["bias_l0c"] = bias_l0c

    c_col_bias = tvm.compute(conv_shape,
                             lambda i, j, k, l: c_col(j // cout1_g, i,
                                                      j % cout1_g, k, l) + \
                                                bias_l0c(i, j, k, l),
                             name=_OP_TAG + 'c_col_bias')
    _TENSOR_MAP["c_col_bias"] = c_col_bias

    return c_col_bias


def _remove_pad(res, res_remove_pad_shape):
    """
    remove pad
    Parameters
    ----------
    res: input tensor

    res_remove_pad_shape: true shape

    Returns
    -------
    res_remove_pad tensor
    """
    _NAME_INDEX[0] += 1
    with tvm.tag_scope('conv_vector_remove_pad'):
        res_tensor = tvm.compute(res_remove_pad_shape,
                                 lambda *indice: res(*indice),
                                 name='remove_pad' + "_cc_" +
                                      str(_NAME_INDEX[0]))
    return res_tensor


def _handle_res_c(res, conv_shape):
    """
    res_c
    Parameters
    ----------
    res: input tensor

    conv_shape: res_c true shape

    Returns
    -------
    res_c tensor
    """
    res_c = tvm.compute(conv_shape,
                        lambda batch, cout1, howo, cout0:
                        res(batch, cout1, howo, cout0),
                        name='C',
                        tag=_OP_TAG + "C")

    _TENSOR_MAP["C"] = res_c
    return res_c


def _get_dynamic_para():
    """
    Get dynamic para.
    """
    if Conv3DParam.dynamic_mode == "dynamic_batch":
        fmap_range = [get_te_var("batch_n").get_bound()]
    elif Conv3DParam.dynamic_mode == "dynamic_dhw":
        fmap_range = [get_te_var("fmap_d").get_bound(),
                      get_te_var("fmap_h").get_bound(),
                      get_te_var("fmap_w").get_bound()]
    else:
        return None
    dynamic_para = {
        "dynamic_mode": Conv3DParam.dynamic_mode,
        "fmap_range": fmap_range
    }
    return dynamic_para


def _get_dynamic_mode(data):
    """
    Return dynamic mode.
    """
    if isinstance(data.shape[0], tvm.expr.Var):
        return "dynamic_batch"
    if isinstance(data.shape[1], tvm.expr.Var) and \
            isinstance(data.shape[3], tvm.expr.Var) \
            and isinstance(data.shape[4], tvm.expr.Var):
        return "dynamic_dhw"
    return None


def _get_var_map():
    """
    Get dynamic mode for Conv3DParam.
    """
    if Conv3DParam.dynamic_mode == "dynamic_batch":
        return {"batch_n" : get_te_var("batch_n").get_tvm_var()}
    if Conv3DParam.dynamic_mode == "dynamic_dhw":
        return {v : get_te_var(v).get_tvm_var()
                for v in ("fmap_h", "fmap_w", "fmap_d", "h_out", "w_out", "d_out")}
    return None


def _check_conv3d_shape(shape_fm, shape_filter, pads, stride_dhw, dilation_dhw,
                        fmp_dtype, w_dtype):
    """
    algorithm: check the input params of conv3d

    Parameters
    ----------
    shape_fm: the shape of feature, format is 'NCDHW'.
        a list/tuple of 'int' that has length `== 5`

    shape_filter: the shape of filter, format is 'NCDHW'.
        a list of 'int' that has length `== 5`

    pads: tuple/list of 6 integers
        [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]

    stride_dhw: A list of `ints` that has length `== 3`

    dilation_dhw: A list of `ints` that has length `== 3`

    fmp_dtype: the dtype of feature

    w_dtype: the dtype of filter

    Returns
    -------
    None
    """

    _, _, fmap_d, fmap_h, fmap_w = shape_fm
    _, _, filter_d, filter_h, filter_w = shape_filter
    pad_d = [pads[0], pads[1]]
    _check_d_dimension(fmap_d, filter_d, pad_d, stride_dhw[0], dilation_dhw[0])

    pad_h = [pads[2], pads[3]]
    _check_h_dimension(fmap_h, filter_h, pad_h, stride_dhw[1], dilation_dhw[1])

    pad_w = [pads[4], pads[5]]
    _check_w_dimension(fmap_w, filter_w, pad_w, stride_dhw[2], dilation_dhw[2])

    # C dimension should align 16
    block_size_k = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][1]
    block_size_m = tbe_platform.CUBE_MKN[fmp_dtype]['mac'][0]

    # calculated by h_i and w_i
    dilation_d, dilation_h, dilation_w = dilation_dhw
    filter_dilated_d = (filter_d - 1) * dilation_d + 1
    filter_dilated_h = (filter_h - 1) * dilation_h + 1
    filter_dilated_w = (filter_w - 1) * dilation_w + 1

    h_out = (fmap_h + (pad_h[0] + pad_h[1]) - filter_dilated_h) // stride_dhw[1] + 1
    w_out = (fmap_w + (pad_w[0] + pad_w[1]) - filter_dilated_w) // stride_dhw[2] + 1
    d_out = (fmap_d + (pad_d[0] + pad_d[1]) - filter_dilated_d) // stride_dhw[0] + 1

    load2d_pass_flag = ((filter_dilated_d == 1) and (filter_dilated_h == 1) and (filter_dilated_w == 1) and
                        (list(pads) == [0, 0, 0, 0, 0, 0]) and
                        (list(stride_dhw) == [1, 1, 1]))

    #  Chip Design demand only h_dimesion constraint
    only_fhkh_pass_flag = ((1 <= filter_dilated_h <= 11) and
                           (stride_dhw[1] == 1) and
                           (h_out == 1))

    if load2d_pass_flag or only_fhkh_pass_flag:
        pass
    else:
        if w_out < 2 and h_out != 1:
            # Chip Design demand w_out must >=2 when h_out != 1
            dict_args = {
                'errCode': 'E62006',
                'error_desc': 'Chip Design demand w_out must >=2 when h_out != 1'
            }
            raise RuntimeError(dict_args,
                               error_manager_util.get_error_message(dict_args))

    # check for not bigger than L1
    l1_buffer_size = tbe_platform.get_soc_spec("L1_SIZE")
    m_bit_ratio = {"float16": 2, "int8": 1}
    point_per_w = ((fmap_w - filter_dilated_w) +
                   pad_w[0] + pad_w[1]) // stride_dhw[2] + 1
    w_in = block_size_m // point_per_w + 2
    tmp = ((w_in - 1) * stride_dhw[1] + filter_dilated_h) * fmap_w
    max_feature_map_l1 = block_size_k * tmp * m_bit_ratio[w_dtype]

    if max_feature_map_l1 > l1_buffer_size:
        dict_args = {
            'errCode': 'E60026',
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _check_d_dimension(fmap_d, filter_d, pad_d, stride_d, dilation_d):
    filter_dilated_d = (filter_d - 1) * dilation_d + 1
    if filter_d < _FILTER_DHW_MIN or filter_d > _FILTER_DHW_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'weight',
            'dim': 'D',
            'range': '[{}, {}]'.format(_FILTER_DHW_MIN, _FILTER_DHW_MAX),
            'actual_value': str(filter_d)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if (fmap_d + pad_d[0] + pad_d[1]) < filter_dilated_d:
        dict_args = {
            'errCode': 'E60012',
            'depth_of_x': str(fmap_d + pad_d[0] + pad_d[1]),
            'depth_of_filter': str(filter_dilated_d),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_d[0] < _PAD_MIN or pad_d[1] < _PAD_MIN or pad_d[0] > _PAD_MAX or pad_d[1] > _PAD_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'pad',
            'dim': 'D',
            'range': '[{}, {}]'.format(_PAD_MIN, _PAD_MAX),
            'actual_value': 'pad_d[0] = {}, pad_d[1] = {}'.format(pad_d[0],
                                                                  pad_d[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_d[0] >= filter_dilated_d or pad_d[1] >= filter_dilated_d:
        dict_args = {
            'errCode': 'E60013',
            'depth_of_pad': 'pad_d[0] = {}, pad_d[1] = {}'.format(pad_d[0],
                                                                  pad_d[1]),
            'depth_of_filter': str(filter_dilated_d)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if stride_d < _STRIDE_MIN or stride_d > _STRIDE_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'stride',
            'dim': 'D',
            'range': '[{}, {}]'.format(_STRIDE_MIN, _STRIDE_MAX),
            'actual_value': str(stride_d),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if dilation_d < _DILATION_MIN or dilation_d > _DILATION_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'dilation',
            'dim': 'D',
            'range': '[{}, {}]'.format(_DILATION_MIN, _DILATION_MAX),
            'actual_value': str(dilation_d),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _check_h_dimension(fmap_h, filter_h, pad_h, stride_h, dilation_h):
    filter_dilated_h = (filter_h - 1) * dilation_h + 1
    if fmap_h < _FMAP_HW_MIN or fmap_h > _FMAP_HW_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'input',
            'dim': 'H',
            'range': '[{}, {}]'.format(_FMAP_HW_MIN, _FMAP_HW_MAX),
            'actual_value': str(fmap_h)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if filter_h < _FILTER_DHW_MIN or filter_h > _FILTER_DHW_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'filter',
            'dim': 'H',
            'range': '[{}, {}]'.format(_FILTER_DHW_MIN, _FILTER_DHW_MAX),
            'actual_value': str(filter_h)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_h[0] < _PAD_MIN or pad_h[1] < _PAD_MIN or pad_h[0] > _PAD_MAX or pad_h[1] > _PAD_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'pad',
            'dim': 'H',
            'range': '[{}, {}]'.format(_PAD_MIN, _PAD_MAX),
            'actual_value': 'pad_h[0] = {}, pad_h[1] = {}'.format(pad_h[0],
                                                                  pad_h[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if (fmap_h + pad_h[0] + pad_h[1]) < filter_dilated_h:
        # Chip Design demand, Load3D
        dict_args = {
            'errCode': 'E60014',
            'h_of_x': str(fmap_h + pad_h[0] + pad_h[1]),
            'h_of_filter': str(filter_dilated_h)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if stride_h < _STRIDE_MIN or stride_h > _STRIDE_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'stride',
            'dim': 'H',
            'range': '[{}, {}]'.format(_STRIDE_MIN, _STRIDE_MAX),
            'actual_value': 'stride_h = {}'.format(stride_h)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_h[0] >= filter_dilated_h or pad_h[1] >= filter_dilated_h:
        dict_args = {
            'errCode': 'E60016',
            'h_of_filter': str(filter_dilated_h),
            'h_of_pad': '[pad_h[0]={}, pad_h[1]={}]'.format(pad_h[0], pad_h[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if dilation_h < _DILATION_MIN or dilation_h > _DILATION_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'dilation',
            'dim': 'H',
            'range': '[{}, {}]'.format(_DILATION_MIN, _DILATION_MAX),
            'actual_value': str(dilation_h),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def _check_w_dimension(fmap_w, filter_w, pad_w, stride_w, dilation_w):
    filter_dilated_w = (filter_w - 1) * dilation_w + 1
    if fmap_w < _FMAP_HW_MIN or fmap_w > _FMAP_HW_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'input',
            'dim': 'W',
            'range': '[{}, {}]'.format(_FMAP_HW_MIN, _FMAP_HW_MAX),
            'actual_value': str(fmap_w)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if filter_w < _FILTER_DHW_MIN or filter_w > _FILTER_DHW_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'filter',
            'dim': 'W',
            'range': '[{}, {}]'.format(_FILTER_DHW_MIN, _FILTER_DHW_MAX),
            'actual_value': str(filter_w)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_w[0] < _PAD_MIN or pad_w[1] < _PAD_MIN or pad_w[0] > _PAD_MAX or pad_w[1] > _PAD_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'pad',
            'dim': 'W',
            'range': '[{}, {}]'.format(_PAD_MIN, _PAD_MAX),
            'actual_value': 'pad_w[0] = {}, pad_w[1] = {}'
                            .format(pad_w[0], pad_w[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if filter_dilated_w > (fmap_w + pad_w[0] + pad_w[1]):
        # Chip Design demand, Load3D
        dict_args = {
            'errCode': 'E60015',
            'w_of_x': str(fmap_w + pad_w[0] + pad_w[1]),
            'w_of_filter': str(filter_dilated_w)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if stride_w < _STRIDE_MIN or stride_w > _STRIDE_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'stride',
            'dim': 'W',
            'range': '[{}, {}]'.format(_STRIDE_MIN, _STRIDE_MAX),
            'actual_value': str(stride_w)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_w[0] >= filter_dilated_w or pad_w[1] >= filter_dilated_w:
        dict_args = {
            'errCode': 'E60017',
            'w_of_filter': str(filter_dilated_w),
            'w_of_pad': '[pad_w[0]={}, pad_w[1]={}]'.format(pad_w[0], pad_w[1])
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if dilation_w < _DILATION_MIN or dilation_w > _DILATION_MAX:
        dict_args = {
            'errCode': 'E62003',
            'param_name': 'dilation',
            'dim': 'W',
            'range': '[{}, {}]'.format(_DILATION_MIN, _DILATION_MAX),
            'actual_value': str(dilation_w),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


@tvm.target.generic_func
def conv3d(x, filter, filter_size, para_dict):
    """
    conv

    Parameters
    ----------
    x: feature map

    weight: filter

    filter_size : filter_size

    para_dict: dict of params

    Returns
    -------
    tensor : res
    """
    in_dtype = x.dtype
    w_dtype = filter.dtype
    Conv3DParam.dynamic_mode = _get_dynamic_mode(x)
    Conv3DParam.var_map = _get_var_map()

    bias_tensor = para_dict["bias_tensor"]
    bias_flag = (bias_tensor is not None)

    group_dict = para_dict["group_dict"]
    pads = para_dict["pads"]
    pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right = pads
    pad_d = [pad_head, pad_tail]
    pad_w = [pad_left, pad_right]
    pad_h = [pad_top, pad_bottom]

    stride_dhw = para_dict["strides"]
    stride_d, stride_h, stride_w = stride_dhw
    if Conv3DParam.dynamic_mode:
        dilation_dhw = para_dict["dilation_dhw"]
    else:
        dilation_dhw = para_dict["dilations"]
    
    shape_filter_ncdhw = filter_size
    filter_n, filter_c, filter_d, filter_h, filter_w = shape_filter_ncdhw

    mad_dtype = para_dict["mad_dtype"]
    res_dtype = para_dict["res_dtype"]

    block_size_k = tbe_platform.CUBE_MKN[w_dtype]['mac'][1]
    filter_c1 = (filter_c + block_size_k - 1) // block_size_k

    # for tiling
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    fmap_shape_ndc1hwc0 = shape_util.shape_to_list(x.shape)
    fmap_n, fmap_d, fmap_c1, fmap_h, fmap_w, fmap_c0 = fmap_shape_ndc1hwc0
    fmap_shape_ndc1hwc0 = [fmap_n, fmap_d, cin1_g, fmap_h, fmap_w, fmap_c0]
    shape_w_ndc1hwc0 = [cout_g, filter_d, cin1_g, filter_h, filter_w,
                        block_size_k]

    # Parameter check
    shape_fmap_ncdhw = [fmap_n, fmap_c1 * fmap_c0, fmap_d, fmap_h, fmap_w]
    if not Conv3DParam.dynamic_mode:
        _check_conv3d_shape(shape_fmap_ncdhw, shape_filter_ncdhw, pads,
                            stride_dhw, dilation_dhw, in_dtype, w_dtype)

    dilation_d, dilation_h, dilation_w = dilation_dhw
    Conv3DParam.tiling_query_param = {
        "fmap_shape_ndc1hwc0": fmap_shape_ndc1hwc0,
        "shape_w_ndc1hwc0": shape_w_ndc1hwc0,
        "in_dtype": in_dtype,
        "w_dtype": w_dtype,
        "res_dtype": res_dtype,
        "mad_dtype": mad_dtype,
        "padw": pad_w,
        "padh": pad_h,
        "padd": pad_d,
        "strideh": stride_h,
        "stridew": stride_w,
        "strided": stride_d,
        "dilationd": dilation_d,
        "dilationh": dilation_h,
        "dilationw": dilation_w,
        "bias_flag": bias_flag,
        "default_tiling": False,
        "group": group_dict["real_g"]
    }

    _TENSOR_MAP["kernel_name"] = para_dict["kernel_name"]
    l0a_load2d_flag = _get_load2d_flag(stride_dhw, pads, shape_filter_ncdhw)
    if not Conv3DParam.dynamic_mode:
        cyclebuffer_flag = tvm.var(name='cyclebuffer_flag', dtype='int')
    else:
        cyclebuffer_flag = 0
        Conv3DParam.tiling_info_dict = {
            "op_type": "convolution_3d",
            "a_shape": fmap_shape_ndc1hwc0,
            "b_shape": shape_w_ndc1hwc0,
            "a_dtype": in_dtype,
            "b_dtype": w_dtype,
            "c_dtype": res_dtype,
            "mad_dtype": mad_dtype,
            "pad": [pad_d[0], pad_d[1], pad_h[0], pad_h[1], pad_w[0], pad_w[1]],
            "stride": [stride_d, stride_h, stride_w],
            "bias_flag": bias_flag,
            "fused_double_operand_num": 0,
            "kernel_name": para_dict["kernel_name"],
            "dynamic_shape_flag": True,
            "dilation": dilation_dhw
        }
    _TENSOR_MAP["l0a_load2d_flag"] = l0a_load2d_flag
    _TENSOR_MAP["cycle_flag_info"] = cyclebuffer_flag
    dsl_flag = para_dict.get("dsl_flag")
    _TENSOR_MAP["dsl_flag"] = dsl_flag

    conv_res = _cube_3d_compute(x,
                                filter,
                                mad_dtype,
                                res_dtype,
                                pads,
                                stride_dhw,
                                dilation_dhw,
                                shape_filter_ncdhw,
                                cyclebuffer_flag,
                                group_dict,
                                bias=bias_tensor)
    res = conv_res
    res_remove_pad_shape = list(res.shape)
    # UB fusion
    if Conv3DParam.dynamic_mode == "dynamic_dhw":
        res_remove_pad_shape[2] = conv_res.op.attrs['true_shape'][2]
    else:
        res_remove_pad_shape[2] = conv_res.op.attrs['true_shape'][2].value
    if dsl_flag:
        c_ub_remove_pad = _handle_res_c(res, res_remove_pad_shape)
        return c_ub_remove_pad


    # Remove H-aligned data in the output shape
    res_remove_pad = _remove_pad(res, res_remove_pad_shape)

    return res_remove_pad
