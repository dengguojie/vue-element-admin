# Copyright 2019 Huawei Technologies Co., Ltd
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
util_conv3d
"""

import warnings

from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import tbe_platform


DYNAMIC_DIM_VAL = -1
_BLOCK_SIZE = 16
FORMAT_NDCHW_DIM = 5
MAX_N_FUZZ_BUILD = 2**31 - 1
MAX_DHW_FUZZ_BUILD = 4096
DX_INPUT_INDEX = 2
DYNAMIC_RANK_FLAG = -2
STRIDE_LENGTH = 5
DILATION_LENGTH = 5
PADS_LENGTH = 6
FILTER_FORMAT_WHITE_LIST = ["NCDHW", "NDHWC", "DHWCN"]

def transform_shape_with_exception(src_format, to_format, ori_shape,
                                   format_white_list, attr_name):
    """
    algorithm: transform_shape_with_exception

    Parameters:
    -------------
    src_format: format of ori_shape

    to_format: format of dst_shape

    ori_shape: origin shape

    format_white_list: white list of format

    attr_name: name of attr

    Returns
    ---------
    new shape
    """
    res = transform_shape_with_format(src_format, to_format, ori_shape,
                                      format_white_list)
    if res is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': attr_name,
            'expected_format_list': ",".join(format_white_list),
            'format': src_format if src_format not in format_white_list else to_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    return res


def transform_shape_with_format(src_format, to_format, ori_shape, format_white_list):
    """
    algorithm: transform_shape_with_format

    Parameters:
    -------------
    src_format: format of ori_shape

    to_format: format of dst_shape

    ori_shape: origin shape

    format_white_list: white list of format

    Returns
    ---------
    new shape
    """
    # input format is not expected
    if ((src_format not in format_white_list) or
        (to_format not in format_white_list)):
        return None
    # need not to transform
    if src_format == to_format:
        return list(ori_shape)
    res_shape = [1 for _ in range(len(to_format))]
    for i, _ in enumerate(to_format):
        for j, _ in enumerate(src_format):
            if to_format[i] == src_format[j]:
                res_shape[i] = ori_shape[j]
                break
    return res_shape


def check_bias(bias, res_dtype):
    """
    algorithm: Check the input params of bias

    Parameters
    ----------

    bias: A dict with keys(shape and dtype) or None
    input bias tensor

    res_dtype: The dtype of output

    Returns
    -------
    None
    """
    bias_dtype = bias.get("dtype").lower()
    if bias_dtype != res_dtype:
        dict_args = {
            'errCode': 'E65002',
            'param_1': 'bias_dtype',
            'param_2': 'res_dtype'
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))


def generalize_input_keep_rank(param_dict: dict) -> dict:
    """
    algorithm: generalize_input_keep_rank

    Parameters:
    -------------
    param_dict: dict of params

    Returns
    ---------
    None
    """
    ori_shape, ori_format = param_dict["ori_shape"], param_dict["ori_format"]
    idx_n = ori_format.find('N')
    idx_d = ori_format.find('D')
    idx_h = ori_format.find('H')
    idx_w = ori_format.find('W')
    idx_tup = (idx_n, idx_d, idx_h, idx_w)
    ori_shape = list(ori_shape)
    for idx in idx_tup:
        ori_shape[idx] = DYNAMIC_DIM_VAL
    param_dict["ori_shape"] = tuple(ori_shape)


def check_l1_limitation_dx(w_value: int, stride_d: int, filter_h_dilation: int,
                           filter_d_dilation: int, block_size_k: int) -> None:
    """
    algorithm: check_l1_limitation_dx

    Parameters:
    -------------
    w_value: w value

    stride_d: stride of d dim

    filter_h_dilation: h dim of filter after dilation

    filter_d_dilation: d dim of filter after dilation

    block_size_k: block size of k dim

    Returns
    ---------
    None
    """
    if w_value > _BLOCK_SIZE:
        h_value_max = filter_h_dilation
    else:
        h_value_max = filter_h_dilation + _BLOCK_SIZE // w_value
    if w_value % _BLOCK_SIZE != 0 and _BLOCK_SIZE % w_value != 0:
        h_value_max += 1
    a_l1_size = h_value_max * w_value * ((filter_d_dilation - 2) // stride_d + 2) * block_size_k * 2
    b_l1_size = _BLOCK_SIZE * block_size_k * 2
    l1_size = tbe_platform.get_soc_spec("L1_SIZE")
    if (a_l1_size + b_l1_size) > l1_size:
        dict_args = {
            'errCode': 'E60026'
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))


def _check_fuzz_shape_range_illegal(check_num, min_num, max_num):
    return not check_num or check_num < min_num or check_num > max_num


def check_fuzz_dynamic_mode(tensor: dict) -> bool:
    """
    check graph mode
    """
    # check graph mode or single mode in fuzzy compile
    if (list(tensor.get("ori_shape")) == [DYNAMIC_RANK_FLAG] or DYNAMIC_DIM_VAL in tensor.get("ori_shape") and
        "ori_range" in tensor.keys()):
        return True
    return False


def check_para_fuzz_compile_3d(x, y, weight, dilations, strides, pads,
                               is_dynamic_fuzz_mode, op_type, x_index=DX_INPUT_INDEX):
    """
    check fuzz compile parameters
    """
    # unknow_rank inputs ori_shape is [-2], others' shape length is 5
    unknow_rank = (list(x.get("ori_shape")) == [DYNAMIC_RANK_FLAG])
    if unknow_rank:
        warnings.warn("{} not support unknow_rank".format(op_type))
        return [{"result": "UNSUPPORTED"}]
    # check strides, pads, dilations length
    if len(strides) != STRIDE_LENGTH:
        warnings.warn("{} strides should be 5d list".format(op_type))
        return [{"result": "UNSUPPORTED"}]
    if len(dilations) != DILATION_LENGTH:
        warnings.warn("{} dilations should be 5d list".format(op_type))
        return [{"result": "UNSUPPORTED"}]
    if len(pads) != PADS_LENGTH:
        warnings.warn("{} pads should be 6d list".format(op_type))
        return [{"result": "UNSUPPORTED"}]
    # check weight format
    if len(weight.get("ori_shape")) != FORMAT_NDCHW_DIM or weight.get("ori_format") not in FILTER_FORMAT_WHITE_LIST:
        warnings.warn("{} weight shape or format illegal".format(op_type))
        return [{"result": "UNSUPPORTED"}]
    # check input and output
    have_range = {"inputs": x, "outputs": y}
    support_format = ["NDHWC", "NCDHW"]
    for name, tensor in have_range.items():
        if tensor.get("ori_format") not in support_format:
            warnings.warn("{} invalid {} ori_format {}, only support {}".format(op_type,
                            name, str(tensor.get("ori_format")), str(support_format)))
            return [{"result": "UNSUPPORTED"}]
        # only change shape NDHW dim to -1, range is already set at infershape
        ori_shape_valid = (isinstance(tensor.get("ori_shape"), (list, tuple)) and
                           len(tensor["ori_shape"]) == FORMAT_NDCHW_DIM)
        if not ori_shape_valid:
            warnings.warn("{}, invalid {} ori_shape {}, only support {}d".format(op_type,
                            name, str(tensor.get("ori_shape")), str(FORMAT_NDCHW_DIM)))
            return [{"result": "UNSUPPORTED"}]
        tensor_format = tensor.get("ori_format")
        n_pos, d_pos, h_pos, w_pos = \
            tensor_format.find("N"), tensor_format.find("D"), tensor_format.find("H"), tensor_format.find("W")
        # check dilation_dhw
        dilations_illegal = (dilations[d_pos] != 1 or dilations[h_pos] != 1 or dilations[w_pos] != 1)
        if dilations_illegal:
            warnings.warn("{} dilations shape d,h,w only support 1.".format(op_type))
            return [{"result": "UNSUPPORTED"}]
        # graph mode check range
        if is_dynamic_fuzz_mode:
            ori_range = tensor.get("ori_range")
            if len(ori_range) != FORMAT_NDCHW_DIM:
                warnings.warn("{} ori_range length illegal".format(op_type))
                return [{"result": "UNSUPPORTED"}]
            dynamic_lower_range_illegal = \
                any([_check_fuzz_shape_range_illegal(dim_range[0], 1, MAX_DHW_FUZZ_BUILD) for dim_range in
                [ori_range[d_pos], ori_range[h_pos], ori_range[w_pos]]]) or \
                _check_fuzz_shape_range_illegal(ori_range[n_pos][0], 1, MAX_N_FUZZ_BUILD)
            if dynamic_lower_range_illegal:
                return [{"result": "UNSUPPORTED", "reason": {"param_index": [x_index], "type": ["lower_limit"]}}]
            dynamic_upper_range_illegal = \
                any([_check_fuzz_shape_range_illegal(dim_range[1], 1, MAX_DHW_FUZZ_BUILD) for dim_range in
                [ori_range[d_pos], ori_range[h_pos], ori_range[w_pos]]]) or \
                _check_fuzz_shape_range_illegal(ori_range[n_pos][1], 1, MAX_N_FUZZ_BUILD)
            if dynamic_upper_range_illegal:
                return [{"result": "UNSUPPORTED", "reason": {"param_index": [x_index], "type": ["upper_limit"]}}]
        # single mode check shape
        ori_shape = tensor.get("ori_shape")
        static_shape_illegal = not is_dynamic_fuzz_mode and \
            (any([_check_fuzz_shape_range_illegal(dim_shape, 1, MAX_DHW_FUZZ_BUILD) for dim_shape in
            [ori_shape[d_pos], ori_shape[h_pos], ori_shape[w_pos]]]) or \
            _check_fuzz_shape_range_illegal(ori_shape[n_pos], 1, MAX_N_FUZZ_BUILD))
        if static_shape_illegal:
            return [{"result": "UNSUPPORTED"}]
    return []


def get_range(fmap):
    """
    get range from ori_range for fuzz build
    """
    fmap_ori_format = fmap.get("ori_format")
    fmap_format = fmap.get("format")
    fmap_ori_range = fmap.get("ori_range")
    if not fmap_format:
        fmap_format = fmap_ori_format
        fmap["format"] = fmap_ori_format
        fmap["range"] = fmap_ori_range
    else:
        pos_n = fmap_ori_format.find('N')
        pos_c = fmap_ori_format.find('C')
        pos_d = fmap_ori_format.find('D')
        pos_h = fmap_ori_format.find('H')
        pos_w = fmap_ori_format.find('W')
        if fmap_format == "NDHWC":
            fmap["range"] = [fmap_ori_range[pos_n], fmap_ori_range[pos_d], fmap_ori_range[pos_h],
                             fmap_ori_range[pos_w], fmap_ori_range[pos_c]]
        else:
            fmap["range"] = [fmap_ori_range[pos_n], fmap_ori_range[pos_c], fmap_ori_range[pos_d],
                             fmap_ori_range[pos_h], fmap_ori_range[pos_w]]


def correct_pads(fmap, out_backprop, weight, strides, pads, is_dynamic_fuzz_mode):
    """
    in fuzz build mode pads might be default to [0, 0, 0, 0, 0, 0],
    set pads to [-1, -1, -1, -1, -1, -1] while padding is not valid.
    """
    if is_dynamic_fuzz_mode:
        return pads
    out_backprop_shape = out_backprop.get("ori_shape")
    out_backprop_format = out_backprop.get("ori_format")
    out_backprop_d = out_backprop_shape[out_backprop_format.find('D')]
    out_backprop_h = out_backprop_shape[out_backprop_format.find('H')]
    out_backprop_w = out_backprop_shape[out_backprop_format.find('W')]
    weight_shape = weight.get("ori_shape")
    weight_format = weight.get("ori_format")
    weight_d = weight_shape[weight_format.find('D')]
    weight_h = weight_shape[weight_format.find('H')]
    weight_w = weight_shape[weight_format.find('W')]
    fmap_shape = fmap.get("ori_shape")
    fmap_format = fmap.get("ori_format")
    pos_d = fmap_format.find('D')
    pos_h = fmap_format.find('H')
    pos_w = fmap_format.find('W')
    fmap_d = fmap_shape[pos_d]
    fmap_h = fmap_shape[pos_h]
    fmap_w = fmap_shape[pos_w]
    stride_d = strides[pos_d]
    stride_h = strides[pos_h]
    stride_w = strides[pos_w]
    need_corret_pads = (all(i == 0 for i in pads) and (out_backprop_d != (fmap_d - weight_d + 1) // stride_d or
                        out_backprop_h != (fmap_h - weight_h + 1) // stride_h or
                        out_backprop_w != (fmap_w - weight_w + 1) // stride_w))
    if need_corret_pads:
        return [-1, -1, -1, -1, -1, -1]
    return pads
