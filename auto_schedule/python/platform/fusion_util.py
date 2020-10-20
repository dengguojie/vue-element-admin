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
fusion template
"""
# pylint: disable=import-error, ungrouped-imports
import os
import sys
import stat
import inspect
import json
import traceback
import importlib
import functools

from te.utils import check_para
from te.platform import cce_policy
from te.platform import operation as op
from te.platform.fusion_manager import fusion_manager
from te.platform.fusion_manager import op_build_cfg_en
from te.platform.fusion_manager import reset_fusion_build_cfg
from te.platform.fusion_manager import get_fusion_build_cfg

from te.lang.cce.te_compute.common import cast_to as _cast_to
from te.lang.cce.te_schedule.cce_schedule_declarations import OpPatterns
from te.lang.cce.te_schedule.cce_schedule import cce_build_code
from te.lang.dynamic.schedule.auto_schedule import build as _dynamic_build

from te.tvm import api as tvm
from te.utils.cce import auto_schedule
from te.tvm.target import cce
from te.tvm import tensor as _tensor


def trans_shape_fullycompress(data_node, node):
    """
    tansform fullyconnectioncompress op shape, input tensor order is
    x, w, compress_index, b, and so on
    :param data_node:
    :param node:
    """
    # trans x shape
    if data_node["name"] == node["input_desc"][0]["name"]:
        shape = data_node["shape"]
        if len(shape) == 5:
            data_node["shape"] = [shape[0], shape[1] *
                                  shape[2]*shape[3]*shape[4]]

    if data_node["name"] == node["input_desc"][3]["name"]:
        shape = data_node["shape"]
        if len(shape) == 5:
            data_node["shape"] = [shape[1]*shape[4]]


def trans_shape(data_node, op_list):
    """
    tansform op shape if necessary
    :param data_node:
    :param op_list:
    """
    try:
        for node in op_list:
            if node["type"] == "FullyConnection":
                trans_fully_connection(data_node, node)
            elif node["type"] == "FullyConnectionCompress":
                trans_shape_fullycompress(data_node, node)
            elif node["type"] == "DepthwiseConv2D":
                trans_depthwise_conv2d(data_node, node)
            elif node["type"] == "MatMulV2":
                trans_matmul_bias_shape(data_node, node)
            elif node["type"] == "BNInferenceD":
                trans_bninference_shape(data_node, node)
            elif node["type"] == "AvgPool":
                trans_avgpool_shape(data_node, node)
            elif node["type"] == "Mul":
                trans_mul_shape(data_node, node, op_list)
            elif node["type"] == "AscendRequantS16":
                trans_arequant_s16(data_node, node)
            elif node["type"] == "BatchMatMul":
                trans_batch_matmul_shape(data_node, node)
            else:
                continue
    except Exception:           # 'pylint: disable=broad-except
        pass


def trans_shape_by_pattern(data_node, op_list):
    """trans shape"""
    try:
        for node in op_list:
            if node.get('pattern') == 'ElemWise':
                trans_elemwise_shape(data_node, node, op_list)
    except Exception:           # 'pylint: disable=broad-except
        pass


def trans_elemwise_shape(data_node, node, op_list):
    """
    broadcast elemwise input shape if necessary
    """
    if data_node['shape'] != [] and data_node['shape'] != [1]:
        return

    for operator in op_list:
        if operator.get('pattern') in ('Convolution', 'Conv2d_backprop_input'):
            # no need to broadcast in Conv+elemwise fusion
            return

    data_node_name = data_node['name']
    max_input_dim = 0

    input_names = [x['name'] for x in node['input_desc']]
    if data_node_name not in input_names:
        return

    input_shapes = [x['shape'] for x in node['input_desc']]
    for shape in input_shapes:
        max_input_dim = max(len(shape), max_input_dim)

    data_node['shape'] = [1] * max(1, max_input_dim)


def trans_depthwise_conv2d(data_node, node):
    """
    tranform shape for depthwise_conv2d
    """
    if data_node["name"] == node["input_desc"][0]["name"]:
        shape = data_node["shape"]
        if len(shape) == 5:
            data_node["shape"] = [shape[0], shape[1], 1,
                                  shape[2], shape[3], shape[4]]
    if data_node["name"] == node["input_desc"][1]["name"]:
        shape = data_node["shape"]
        if len(shape) == 6:
            data_node["shape"] = [shape[0], shape[1], shape[2],
                                  shape[4], shape[5]]


def trans_arequant_s16(data_node, node):
    """
    transform AscendRequantS16 shape
    """
    if data_node["name"] == node["input_desc"][0]["name"]:
        shape = data_node["shape"]
        if len(shape) == 5:
            data_node["shape"] = [shape[0], shape[1],
                                  shape[2] * shape[3], shape[4]]
    if data_node["name"] == node["input_desc"][2]["name"]:
        shape = data_node["shape"]
        if len(shape) == 5:
            data_node["shape"] = [shape[0], shape[1],
                                  shape[2] * shape[3], shape[4]]


def trans_batch_matmul_shape(data_node, node):
    """
    tansform batch_matmul op shape if necessary
    :param data_node:
    :param node:
    """
    if data_node["name"] == node["input_desc"][0]["name"]:
        shape = data_node["shape"]
        if len(shape) > 5:
            data_node["shape"] = [functools.reduce(
                lambda x, y: x * y, shape[:-4])] + shape[-4:]
        return
    if data_node["name"] == node["input_desc"][1]["name"]:
        shape = data_node["shape"]
        if len(shape) > 5:
            data_node["shape"] = [functools.reduce(
                lambda x, y: x * y, shape[:-4])] + shape[-4:]
        return
    if data_node["name"] == node["input_desc"][2]["name"]:
        shape = data_node["shape"]
        if len(shape) > 5:
            data_node["shape"] = [functools.reduce(
                lambda x, y: x * y, shape[:-4])] + shape[-4:]
        return


def trans_mul_shape(data_node, node, op_list):
    """
    tansform mul shape
    :param node_name:
    :param op_list:
    """
    if data_node["name"] == node["input_desc"][1]["name"]:
        shape = data_node["shape"]
        for node_list in op_list:
            if node["input_desc"][0]["name"] == \
                node_list["output_desc"][0]["name"] and \
                (node_list["type"] == "AvgPool" or
                 node_list["type"] == "AscendDequant"):
                shape = data_node["shape"]
                data_node["shape"] = [shape[0], shape[1], 1,
                                      shape[2] * shape[3], shape[4]]


def trans_avgpool_shape(data_node, node):
    """
    tansform avgpool shape
    :param data_node:
    :param node:
    """
    all_out = []
    if data_node["name"] == node["input_desc"][1]["name"]:
        shape = data_node["shape"]
        for arg in node['prebuild_outs_attrs']['list_args']:
            all_out.append(arg)
        ksize = all_out[1]
        format_attr = all_out[4]
        if format_attr == "NHWC":
            ksize_hw = ksize[1]
        else:
            ksize_hw = ksize[2]

        if len(shape) == 4 and data_node["data_type"] != "int8":
            data_node["shape"] = [shape[0] // (ksize_hw * ksize_hw),
                                  ksize_hw, ksize_hw, shape[2], shape[3]]

        if len(shape) == 6 and data_node["data_type"] == "int8":
            shape = data_node["shape"]
            data_node["shape"] = [shape[0],
                                  ksize_hw, ksize_hw, 32, 32]

    if data_node["name"] == node["input_desc"][2]["name"]:
        ori_shape = data_node["ori_shape"]
        data_node["shape"] = ori_shape


def trans_fully_connection(data_node, node):
    """trans fully connection"""
    if data_node["name"] == node["input_desc"][0]["name"]:
        shape = data_node["shape"]
        if len(shape) == 5:
            data_node["shape"] = [shape[0], shape[1] *
                                  shape[2]*shape[3]*shape[4]]

    if data_node["name"] == node["input_desc"][2]["name"]:
        shape = data_node["shape"]
        if len(shape) == 5:
            data_node["shape"] = [shape[1]*shape[4]]


def trans_bninference_shape(data_node, node):
    """trans bn inference shape"""
    if data_node["name"] == node["input_desc"][1]["name"] \
            or data_node["name"] == node["input_desc"][2]["name"]:
        shape = data_node["shape"]
        if data_node["format"] == "NHWC":
            shape = [1, 1, 1] + shape
        elif data_node["format"] == "NCHW":
            shape = [1] + shape + [1, 1]
        data_node["shape"] = shape


def trans_matmul_bias_shape(data_node, node):
    """
    tansform matmul bias op shape if necessary
    :param data_node:
    :param node:
    """
    if data_node["name"] == node["input_desc"][2]["name"]:
        shape = data_node["shape"]
        if len(shape) == 1:
            data_node["shape"] = _reshape_bias_shape(shape)
            return


def _reshape_bias_shape(shape_bias):
    """
    tansform matmul bias op shape
    :param shape_bias:
    """
    bias_length = shape_bias[0]
    if bias_length % 16 != 0:
        bias_length = (bias_length // 16) * 16 + 16
        shape_bias = []
        shape_bias.append(bias_length)

    return shape_bias


def aipp_format_change(op_node, op_list):
    """
    specific case for conv2d_compress placeholder
    :param data_node:
    :param op_list:
    """
    for op_operator in op_list:
        try:
            if op_operator["type"] != "Aipp":
                continue

            aipp_input = op_operator["input_desc"]
            if op_node['name'] != aipp_input[0]['name']:
                continue

            desc = op_node["output_desc"][0]

            for op_f, op_s in (('format', 'shape'),
                               ('ori_format', 'ori_shape')):
                if desc.get(op_f) == "NHWC":
                    desc[op_f] = "NCHW"
                    desc[op_s] = [desc[op_s][0], desc[op_s][3],
                                  desc[op_s][1], desc[op_s][2]]
        except Exception:       # 'pylint: disable=broad-except
            continue


def compress_node(op_node, op_list):
    """
    specific case for conv2d_compress placeholder
    :param data_node:
    :param op_list:
    """
    for op_operator in op_list:
        if op_operator["type"] != "Conv2DCompress" and \
                op_operator["type"] != "FullyConnectionCompress":
            continue

        compress_index_input = op_operator["input_desc"]
        try:
            if op_node["name"] != compress_index_input[2]["name"]:
                continue
        except Exception:       # 'pylint: disable=broad-except
            continue

        compress_index_shape = tvm.var("compress_index_shape", dtype="int32")
        compress_index = tvm.placeholder((compress_index_shape,),
                                         name='compress_index', dtype="int8")
        return compress_index

    return None


def create_placeholder_tensor(op_node, tensor_list, input_list,
                              op_list, params_count):
    """
    create placeholder tensor, get input tensor list

    Parameters
    ----------
    op_node : input fusion op node

    tensor_list : tensor list

    input_list : input tensor list

    op_list : op list

    params_count : param loop count

    Returns
    -------
    None
    """

    desc = op_node["output_desc"][0]
    aipp_format_change(op_node, op_list)

    if desc["shape"] == "NULL":
        tensor_list[desc["name"]] = None
    else:
        sformat = desc.get("format", "")
        ori_shape = desc.get("ori_shape", [])
        ori_format = desc.get("ori_format", "")
        addr_type = desc.get("addr_type", 0)
        valid_shape = desc.get("valid_shape", [])
        slice_offset = desc.get("slice_offset", [])
        l1_fusion_type = desc.get("L1_fusion_type", -1)
        l1_addr_flag = desc.get("L1_addr_flag", -1)
        l1_addr_offset = desc.get("L1_addr_offset", -1)
        l1_valid_size = desc.get("L1_valid_size", -1)

        para_name = "params_%s" % str(params_count[0])
        trans_shape(desc, op_list)
        trans_shape_by_pattern(desc, op_list)

        out = compress_node(op_node, op_list)
        if out is not None:
            tensor_list[desc["name"]] = out
        else:
            attr = {
                "format": sformat,
                "ori_shape": ori_shape,
                "ori_format": ori_format,
                "addr_type": addr_type,
                "valid_shape": valid_shape,
                "slice_offset": slice_offset,
                "L1_fusion_type":  l1_fusion_type,
                "L1_addr_flag": l1_addr_flag,
                "L1_addr_offset": l1_addr_offset,
                "L1_valid_size": l1_valid_size}

            tensor_list[desc["name"]] = \
                tvm.placeholder(desc["shape"], desc["data_type"],
                                para_name, attrs=attr)
        params_count[0] = params_count[0] + 1

    input_list.append(tensor_list[desc["name"]])


# pylint: disable=too-many-arguments
def add_input_tensor(op_node, tensor_list, op_input_list, is_used_tensor_list,
                     input_tensor_cnt, dyn_input_dict):
    """
    add input tensor

    Parameters
    ----------
    op_node : input fusion op node

    tensor_list : tensor list

    op_input_list : input tensor list

    is_used_tensor_list : used tensor list

    input_tensor_cnt : input tensor used count

    dyn_input_dict : dynamic input dict

    Returns
    -------
    None
    """

    for input_desc in op_node["input_desc"]:
        check_input_desc_not_in_tensor(input_desc, tensor_list)

        if "dyn_index" in input_desc:
            if "dyn_index" not in dyn_input_dict:
                dyn_input_dict["dyn_index"] = []
            dyn_input_dict["dyn_index"].append(
                tensor_list[input_desc["name"]])
        else:
            op_input_list.append(tensor_list[input_desc["name"]])

        is_used_tensor_list.add(tensor_list[input_desc["name"]])
        # count input tensor called by other tensor
        if tensor_list[input_desc["name"]] in input_tensor_cnt:
            input_tensor_cnt[tensor_list[input_desc["name"]]] += 1
        else:
            input_tensor_cnt[tensor_list[input_desc["name"]]] = 1


def get_fusion_op_kernel_name(func_name, node_name, kernel_name):
    """
    get kernel_name kwds of the op
    """
    try:
        if fusion_manager.has_op_params(node_name):
            kwds = fusion_manager.get_op_kwds(node_name)
            if 'kernel_name' in kwds:
                return {}

        opfunc = fusion_manager.get_op_compute_func(func_name)
        if inspect.signature(opfunc).parameters['kernel_name'].kind in \
           (inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return {'kernel_name': kernel_name}
        return {}
    except Exception:           # 'pylint: disable=broad-except
        return {}


def get_op_outputs_attrs(op_node):
    """
    get op outputs and attrs
    """
    list_args = []
    kwds_args = {}
    if 'prebuild_outs_attrs' in op_node:
        outs_attrs = op_node['prebuild_outs_attrs']
        list_args = outs_attrs['list_args']
        kwds_args = outs_attrs['kwds_args']
        return (list_args, kwds_args)

    return (list_args, kwds_args)


def import_op_module(op_node):
    """
    imort op py module if necessary.
    return op func
    """
    py_module_path = op_node.get('py_module_path', '')
    if py_module_path != '' and py_module_path not in sys.path:
        sys.path.append(py_module_path)
    try:
        module_name = op_node.get('module_name', '')
        if module_name != '':
            opm = importlib.import_module(module_name)
            opfunc = getattr(opm, op_node["func_name"])
            return opfunc
        return None
    except Exception:       # pylint: disable=bare-except,broad-except
        return None


def import_dyn_op_module(op_node):
    """
    imort op py module if necessary.
    return op func
    """
    py_module_path = op_node.get('py_module_path', '')
    if py_module_path != '' and py_module_path not in sys.path:
        sys.path.append(py_module_path)
    try:
        module_name = op_node.get('module_name', '')
        if module_name != '':
            dyn_op_module = module_name.split('.')
            dyn_op_module[-1] = 'dynamic'
            dyn_op_module = '.'.join(dyn_op_module)
            importlib.import_module(dyn_op_module)
            op_type = op_node['type']
            return op.get_fusion_compute(op_type)
        return None
    except Exception:       # pylint: disable=bare-except,broad-except
        return None


def call_op_compute(op_node, op_input_list,
                    dyn_input_dict, kernel_name=None):
    """
    call op compute

    Parameters
    ----------
    op_node : input fusion op node

    op_input_list : input tensor list

    dyn_input_dict : dynamic input dict

    Returns
    -------
    op_output_list: output op list tensor
    """
    # call op's compute
    all_args = []
    all_out = []
    all_args_kwds = {}
    kernel_kwds = {}
    with cce():
        op_compute_func = \
            fusion_manager.get_op_compute_func(op_node["func_name"])
        if op_compute_func is None:
            import_op_module(op_node)
            op_compute_func = \
                fusion_manager.get_op_compute_func(op_node["func_name"])

        if fusion_manager.get_op_build_type(op_node["name"]) == "prebuild" \
           or 'prebuild_outs_attrs' in op_node:
            all_args = op_input_list
            # add dyn input para to commpute args
            for dyn_input in dyn_input_dict:
                all_args.append(dyn_input_dict[dyn_input])
            all_list, all_kwds = get_op_outputs_attrs(op_node)
            all_out.extend(all_list)
            all_args_kwds = all_kwds
            kernel_kwds = get_fusion_op_kernel_name(op_node["func_name"],
                                                    op_node["name"],
                                                    kernel_name)
            op_impl_kwds = \
                cce_policy.OpImplPolicy.get_op_impl_mode(op_compute_func,
                                                         op_node['type'])
        else:
            all_args.append(op_input_list)
            for arg in fusion_manager.get_op_args(op_node["name"]):
                all_out.append(arg)
            all_args_kwds = fusion_manager.get_op_kwds(op_node["name"])

        all_args.extend(all_out)
        op_output_list = op_compute_func(*all_args, **kernel_kwds,
                                         **all_args_kwds,
                                         **op_impl_kwds)

    op_output_list = [op_output_list] if isinstance(
        op_output_list, _tensor.Tensor) else op_output_list

    try:
        if op_node["pattern"] == OpPatterns.ELEMWISE_PATTERN.value:
            for idx, tensor in enumerate(op_output_list):
                if tensor.dtype != all_out[idx]["dtype"]:
                    op_output_list[idx] = \
                        _cast_to(op_output_list[idx], all_out[idx]["dtype"])
    except Exception:       # 'pylint: disable=bare-except,broad-except
        pass

    return op_output_list


def check_input_desc_not_in_op(op_node):
    """
    check input description in op or not

    Parameters
    ----------
    op_node : input fusion op node

    Returns
    -------
    None
    """
    if "input_desc" not in op_node:
        raise RuntimeError(
            "Lack of input_desc for op name: %s" % op_node["name"])


def check_output_desc_not_in_op(op_node):
    """
    check output description in op or not

    Parameters
    ----------
    op_node : output fusion op node

    Returns
    -------
    None
    """
    if "output_desc" not in op_node:
        raise RuntimeError(
            "Lack of output_desc for op name: %s" % op_node["name"])


def check_input_desc_not_in_tensor(input_desc, tensor_list):
    """
    check input description in tensor or not

    Parameters
    ----------
    input_desc : input tensor

    tensor_list : tensor list

    Returns
    -------
    None
    """
    if input_desc["name"] not in tensor_list:
        raise RuntimeError(
            'Can not find input tensor: %s' % input_desc["name"])


def check_output_desc_in_tensor(output_desc, tensor_list):
    """
    check output description in tensor or not

    Parameters
    ----------
    output_desc : output tensor

    tensor_list : tensor list

    Returns
    -------
    None
    """
    if output_desc["name"] in tensor_list:
        raise RuntimeError(
            "Output tensor already exists %s" % output_desc["name"])


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def fusion_op_compute(json_str):
    """
    get the output tensor

    Parameters
    ----------
    json_str : input json data

    Returns
    -------
    output tensor
    """
    return fusion_op(json_str, True)


def check_fusion_op_type(op_list):
    """
    check fusion op type
    ----------
    op_list: all ops info

    Returns
    -------
    None
    """
    matmul_elmt_fuse_type = ["Elu", "LeakyRelu", "Gelu", "Softsign", "Relu6",
                             "Relu", "Softplus", "Sigmoid", "Tanh", "Selu",
                             "GeluGrad", "Add", 'AddN', 'FastGelu', 'FastGeluGrad']

    matmul_deq_elwt_fuse_type = ["Elu", "LeakyRelu", "Gelu", "Softsign",
                                 "Relu6", "Relu", "Softplus", "Sigmoid",
                                 "Tanh", "Selu", "GeluGrad", "Add", 'AddN',
                                 'FastGelu', 'FastGeluGrad', "Eltwise",
                                 "Prelu", "Mul", "Power"]

    matmul_fusion = False
    dequant_fusion = False
    for op_node in op_list:
        if "pattern" in op_node:
            if op_node["pattern"] == OpPatterns.MATMUL_PATTERN.value:
                matmul_fusion = True
            if op_node["pattern"] == OpPatterns.ASCEND_DEQUANT_PATTERN.value:
                dequant_fusion = True

    if not matmul_fusion:
        return

    for op_node in op_list:
        if "pattern" in op_node:
            if not dequant_fusion:
                if (op_node["pattern"] == OpPatterns.ELEMWISE_PATTERN.value) \
                        and (op_node["type"] not in matmul_elmt_fuse_type):
                    raise RuntimeError(
                        "Matmul elementwise fusion only support ('Elu', "
                        "'Relu6', 'LeakyRelu','Gelu','Softsign','Relu6', "
                        "'Relu','Selu', 'Sigmoid','Tanh','Softplus', "
                        "'GeluGrad', 'Add', 'AddN', 'FastGelu', "
                        "'FastGeluGrad'), not support fusion with '%s'"
                        % op_node["type"])
            else:
                if (op_node["pattern"] == OpPatterns.ELEMWISE_PATTERN.value) \
                        and (op_node["type"] not in matmul_deq_elwt_fuse_type):
                    raise RuntimeError(
                        "Matmul elementwise fusion only support ('Elu', "
                        "'Relu6', 'LeakyRelu','Gelu','Softsign','Relu6', "
                        "'Relu','Selu', 'Sigmoid','Tanh','Softplus', "
                        "'GeluGrad', 'Add', 'AddN', 'FastGelu', "
                        "'FastGeluGrad', 'Eltwise', 'Prelu', 'Mul', 'Power'), "
                        "not support fusion with '%s'" % op_node["type"])


def get_real_output(sch, output_list):
    """
    get_real_output
    """
    # some schedule will modify out tensor, need update real out tensor
    try:
        if sch.cce_special["real_out_tensor"]:
            real_output = sch.cce_special["real_out_tensor"]
        else:
            real_output = output_list
        return real_output
    except Exception:           # 'pylint: disable=broad-except
        return output_list


def check_single_op(op_json):
    """Check if the json string contains only one op

    Parameters
    ----------
    op_json : json description of fusion op

    Returns
    -------
    succ_flag : boolean
    """
    count = len([operator
                 for operator in op_json['op_list']
                 if operator['type'] != 'Data'])
    return count == 1


def single_op_build(json_data):
    """call op's entry function directly if there's only one single op

    Parameters
    ----------
    json_data : dict
        json description of op

    """
    kernel_name = json_data['fusion_op_name']
    single_op = [operator
                 for operator in json_data['op_list']
                 if operator['type'] != 'Data'][0]
    inout = [op for op in json_data['op_list'] if op['type'] == 'Data']
    op_inputs = single_op['input_desc']

    inputs = []
    for op_inout in inout:
        if op_inout['name'] in [op['name'] for op in op_inputs]:
            inputs.append(op_inout['output_desc'][0])

    for idx, item in enumerate(inputs):
        item.pop('name', None)
        item['dtype'] = item['data_type']
        item.pop('data_type', None)
        if item.get('shape') == 'NULL':
            inputs[idx] = None

    opfunc = import_op_module(single_op)
    list_args, _ = get_op_outputs_attrs(single_op)
    kwargs = \
        cce_policy.OpImplPolicy.get_op_impl_mode(opfunc, single_op['type'])
    op_build_cfg_en()
    opfunc(*inputs, *list_args, kernel_name, **kwargs)


def dump_fusion_json(json_str, dump_path):
    """
    dump fusion json to kernel_meta directory
    """
    # get json data
    json_data = json.loads(json_str)
    json_file_name = json_data["fusion_op_name"]

    # delete '_rl'/'_gl' suffix
    if json_file_name.endswith('_rl') or json_file_name.endswith('_ga'):
        json_file_name = json_file_name[:-3]
        json_data["fusion_op_name"] = json_file_name

    del json_data['SocInfo']['autoTilingMode']
    json_str = json.dumps(json_data)
    dump_path = os.path.realpath(dump_path)

    try:
        if not os.path.exists(dump_path):
            os.mkdir(dump_path)
            os.chmod(dump_path, stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)
    except FileExistsError:
        pass

    try:
        dump_path = os.path.join(dump_path,
                                 "fusion_op_{}.json".format(json_file_name))
        jsonf = open(dump_path, "w")
        jsonf.write(json_str)
        jsonf.close()
        return ""
    except Exception:           # 'pylint: disable=broad-except
        msg = traceback.format_exception_only(sys.exc_info()[0],
                                              sys.exc_info()[1])
        return "".join(msg)


def init_op_cfg(json_data):
    """
    init l1 size, etc...
    """
    if 'l1_size' in json_data:
        cce_policy.set_L1_info("op_L1_space", json_data['l1_size'])
    else:
        cce_policy.set_L1_info("op_L1_space", -1)


def has_dynshape(op_list):
    """
    check if dynamic shape
    """
    for node in op_list:
        if node['type'] == 'Data':
            for data in node['output_desc']:
                if data['shape'] == 'NULL':
                    continue
                if [ele for ele in data['shape'] if ele < 0]:
                    return True
    return False


def modify_duplicated_inputs(json_data):
    """
    rename names of duplicated inputs
    """
    dup_data_names = {}
    for operator in json_data['op_list']:
        if operator['type'] != 'Data':
            continue
        count = dup_data_names.setdefault(operator['name'], [])
        count.append(operator)

    dup_data_names = {key: value for (key, value)
                      in dup_data_names.items()
                      if len(value) > 1}

    dup_indesc_names = {}
    for operator in json_data['op_list']:
        if operator['type'] == 'Data':
            continue
        for indesc in operator['input_desc']:
            if indesc['name'] in dup_data_names.keys():
                count = dup_indesc_names.setdefault(indesc['name'], [])
                count.append(indesc)
    if len(dup_data_names) != len(dup_indesc_names):
        raise RuntimeError('Duplicated names not match')

    for name, ops in dup_data_names.items():
        indesc_names = dup_indesc_names[name]
        if len(ops) != len(indesc_names):
            raise RuntimeError('Duplicated names not match')
        for idx, opdesc in enumerate(zip(ops, indesc_names)):
            new_name = name + '___' + str(idx)
            opdesc[0]['name'] = new_name
            opdesc[0]['output_desc'][0]['name'] = new_name
            opdesc[1]['name'] = new_name


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def fusion_op(json_str, compute_only=False):
    """fusion template

    Parameters
    ----------
    json_str : string
        The input json data.

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    # get json data
    json_data = json.loads(json_str)
    use_int64_mode = False
    for op_node in json_data["op_list"]:
        if "int64mode" in op_node.keys():
            if op_node["int64mode"]:
                use_int64_mode = True
                break

    if use_int64_mode:
        with tvm.api_config.bit_width_64():
            return fusion_op_impl(json_data, compute_only)
    else:
        with tvm.api_config.bit_width_32():
            return fusion_op_impl(json_data, compute_only)


def fusion_op_impl(json_data, compute_only=False):
    """ fusion op impl"""
    if has_dynshape(json_data["op_list"]):
        return fusion_op_dynshape(json_data)

    init_op_cfg(json_data)
    reset_fusion_build_cfg()

    if check_single_op(json_data) and not compute_only:
        single_op_build(json_data)
        return ""

    modify_duplicated_inputs(json_data)

    # get params from json_data
    fusion_op_name = str(json_data["fusion_op_name"])
    op_list = json_data["op_list"]

    # check input args
    check_para.check_kernel_name(fusion_op_name)

    # check fusion op type
    check_fusion_op_type(op_list)

    # init
    tensor_list = {}  # collect all tensors in fusion template
    input_list = []  # record all input tensors for AI Core codegen
    input_tensor_cnt = {}  # record input tensor called count
    output_tensor_cnt = {}  # record output tensor called count
    output_list = []  # record output tensors' name for AI Core codegen
    compute_output_tensor_list = []  # record all compute output tensor
    # record tensor used in fusion_op
    # a tensor which is not used is a output tensor
    is_used_tensor_list = set()
    # combine computes
    params_count = [0]
    cmp_bool_storage_as_1bit = True
    bool_storage_as_1bit_oplist = \
        ["Asinh", "Atanh", "Acosh", "Asin", "Atan2", "Acos", "Pow", "Xlogy",
         "ApproximateEqual", "DataFormatDimMap", "Elu", "Select", "SelectV2",
         "BNLL", "ClipByNormNoDivSum", "BesselI1e", "Expm1", "Log1p"]

    def _collect_out_tensor(op_node):
        for output_desc in op_node["output_desc"]:
            if output_desc["name"] not in tensor_list:
                output_tensor = op_output_list[output_desc["output_index"]]
                output_tensor.op.attrs["addr_type"] = \
                    output_desc.get("addr_type", 0)
                tensor_list[output_desc["name"]] = output_tensor
                compute_output_tensor_list.append(output_tensor)

                # record output tensor called by other tensor
                if output_tensor not in output_tensor_cnt:
                    output_tensor_cnt[output_tensor] = 0

                tmp_cnt = output_tensor_cnt[output_tensor]
                output_tensor_cnt[output_tensor] = tmp_cnt + 1
            else:
                raise RuntimeError(
                    "Output tensor already exists %s" % output_desc["name"])

    for op_node in op_list:
        # op with 'bool_storage_as_1bit' needs to add this config in fusion_op
        if op_node["type"] in bool_storage_as_1bit_oplist:
            cmp_bool_storage_as_1bit = False

        if op_node["type"] == "Data":
            # create placeholder
            create_placeholder_tensor(op_node, tensor_list, input_list,
                                      op_list, params_count)
            continue

        # collect input tensors for this op
        op_input_list = []
        # Assem dynamic input parameter
        dyn_input_dict = {}

        check_input_desc_not_in_op(op_node)
        add_input_tensor(op_node, tensor_list, op_input_list,
                         is_used_tensor_list, input_tensor_cnt, dyn_input_dict)

        # call op's compute
        op_output_list = call_op_compute(op_node, op_input_list,
                                         dyn_input_dict, fusion_op_name)
        check_output_desc_not_in_op(op_node)
        _collect_out_tensor(op_node)

    # find sub-graph output compute
    for tensor in compute_output_tensor_list:
        if tensor not in is_used_tensor_list:
            output_list.append(tensor)
            is_used_tensor_list.add(tensor)
            input_tensor_cnt[tensor] = output_tensor_cnt[tensor]
        # expose the tensor while input cnt < output cnt
        elif output_tensor_cnt[tensor] > input_tensor_cnt[tensor]:
            output_list.append(tensor)
            input_tensor_cnt[tensor] = output_tensor_cnt[tensor]

    if compute_only:
        return output_list

    # generate schedule
    with cce():
        # call auto_schedule
        sch = auto_schedule(output_list)

    input_list = [ele for ele in input_list if ele is not None]
    real_output = get_real_output(sch, output_list)

    # codegen
    config = {"name": fusion_op_name,
              "tensor_list": input_list + real_output,
              "fusion_build_config": get_fusion_build_cfg(),
              "bool_storage_as_1bit": cmp_bool_storage_as_1bit}

    cce_build_code(sch, config)
    return ""


def dyn_op_compute(op_node, placeholders, res_tensors, tensor_usecount):
    """
    call dynmaic op compute
    """
    op_compute = import_dyn_op_module(op_node)
    input_args = []
    desc = op_node["input_desc"]
    for op_input in desc:
        if op_input["shape"] == "NULL":
            input_args.append(None)
        else:
            data_type = op_input.get('data_type')
            if data_type:
                op_input['dtype'] = data_type
            input_param = res_tensors.get(op_input['name'])
            if input_param is None:
                # use dict as op compute input
                input_args.append(op_input)
            else:
                # use tensor as op compute input
                input_args.append(input_param)
                count = tensor_usecount.get(input_param, 0)
                tensor_usecount[input_param] = count + 1

    output_args, kw_args = get_op_outputs_attrs(op_node)
    op_res = op_compute(*input_args, *output_args, **kw_args)

    op_res_placeholders = op_res['op_placeholder']
    op_res_tensors = op_res['op_res']

    op_input_names = [opin['name'] for opin in desc if opin['shape'] != 'NULL']
    for idx, name in enumerate(op_input_names):
        # save op_placeholder
        if op_res_placeholders[idx] not in input_args:
            placeholders[name] = op_res_placeholders[idx]

    op_output_names = [opout['name'] for opout in op_node['output_desc']]
    for idx, name in enumerate(op_output_names):
        # save op res
        res_tensors[name] = op_res_tensors[idx]
    return op_res


def fusion_op_dynshape(json_data):
    """
    fusion op for dynamic shape
    """
    init_op_cfg(json_data)
    reset_fusion_build_cfg()

    op_list = json_data["op_list"]
    fusion_op_name = str(json_data["fusion_op_name"])

    check_para.check_kernel_name(fusion_op_name)

    # check fusion op type
    check_fusion_op_type(op_list)

    placeholders = {}
    res_tensors = {}
    tensor_usecount = {}
    all_res = []
    op_types = [operator['type'] for operator in op_list if operator['type'] != 'Data']
    with op.OperatorContext(op.OpMode.DYNAMIC) as opc:
        opc.set_op_type(op_types)
        with op.ComputeContext():
            for op_node in op_list:
                if op_node['type'] == 'Data':
                    continue
                dyn_op_compute(op_node, placeholders,
                               res_tensors, tensor_usecount)

            for tensor in res_tensors.values():
                if tensor_usecount.get(tensor, 0) == 0:
                    all_res.append(tensor)

        with cce():
            sch = auto_schedule(all_res)

        real_output = get_real_output(sch, all_res)
        tensor_list = list(placeholders.values()) + real_output

        config = {"name": fusion_op_name,
                  "tensor_list": tensor_list,
                  "build_args": get_fusion_build_cfg()}

        _dynamic_build(sch, config)
        return json.dumps(op.get_compile_info())
