#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

fusion template
"""

# pylint: disable=import-error, ungrouped-imports
import os
import sys
import stat
import inspect
import json
import importlib

import te.lang.cce
from te import tvm
from topi import generic
from topi.cce.util import check_kernel_name
from te.lang.cce.te_schedule.cce_schedule_declarations import OpPatterns
from te.platform import set_L1_info
import te.platform.fusion_manager as fm
from te.platform.fusion_manager import fusion_manager
from .cce_policy import OpImplPolicy


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
            elif node["type"] == "DepthwiseConv2D":
                if data_node["name"] == node["input_desc"][0]["name"]:
                    shape = data_node["shape"]
                    if len(shape) == 5:
                        data_node["shape"] = [shape[0], shape[1], 1,\
                                             shape[2], shape[3], shape[4]]
                if data_node["name"] == node["input_desc"][1]["name"]:
                    shape = data_node["shape"]
                    if len(shape) == 6:
                        data_node["shape"] = [shape[0], shape[1], shape[2],\
                                             shape[4], shape[5]]
            elif node["type"] == "MatMulV2":
                trans_matmul_bias_shape(data_node, node)
            elif node["type"] == "BNInferenceD":
                trans_bninference_shape(data_node, node)
            elif node["type"] == "AvgPool":
                trans_avgpool_shape(data_node, node)
            elif node["type"] == "Mul":
                trans_mul_shape(data_node, node, op_list)
            else:
                continue
    except:                     # pylint: disable=bare-except
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
                node_list["type"] == "AvgPool":
                shape = data_node["shape"]
                data_node["shape"] = [shape[0], shape[1], 1,\
                                      shape[2] * shape[3], shape[4]]
                return
    return


def trans_avgpool_shape(data_node, node):
    """
    tansform avgpool shape
    :param data_node:
    :param node:
    """
    all_out = []
    ksize = []
    if data_node["name"] == node["input_desc"][1]["name"]:
        shape = data_node["shape"]
        for arg in fusion_manager.get_op_args(node["name"]):
            all_out.append(arg)
        ksize = all_out[1]
        format_attr = all_out[4]
        if format_attr == "NHWC":
            ksize_hw = ksize[1]
        else:
            ksize_hw = ksize[2]
        if len(shape) == 4:
            data_node["shape"] = [shape[0] // (ksize_hw * ksize_hw), \
                                  ksize_hw, ksize_hw, shape[2], shape[3]]
            return
    return


def trans_fully_connection(data_node, node):
    if data_node["name"] == node["input_desc"][0]["name"]:
        shape = data_node["shape"]
        if len(shape) == 5:
            data_node["shape"] = [shape[0], shape[1] * \
                                  shape[2]*shape[3]*shape[4]]

    if data_node["name"] == node["input_desc"][2]["name"]:
        shape = data_node["shape"]
        if len(shape) == 5:
            data_node["shape"] = [shape[1]*shape[4]]
    return


def trans_bninference_shape(data_node, node):
    if data_node["name"] == node["input_desc"][1]["name"] \
            or data_node["name"] == node["input_desc"][2]["name"]:
        shape = data_node["shape"]
        if data_node["format"] == "NHWC":
            shape = [1, 1, 1] + shape
        elif data_node["format"] == "NCHW":
            shape = [1] + shape + [1, 1]
        data_node["shape"] = shape
    return


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
        except Exception:                 # 'pylint: disable=bare-except
            continue


def conv2d_compress_node(op_node, op_list):
    """
    specific case for conv2d_compress placeholder
    :param data_node:
    :param op_list:
    """
    for op_operator in op_list:
        if op_operator["type"] != "Conv2DCompress":
            continue

        conv2d_input = op_operator["input_desc"]
        try:
            if op_node["name"] != conv2d_input[2]["name"]:
                continue
        except Exception:                 # 'pylint: disable=bare-except
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
        l1_workspace = desc.get("use_L1_workspace", 0)
        l1_fusion_type = desc.get("L1_fusion_type", -1)
        l1_addr_offset = desc.get("L1_addr_offset", 0)

        para_name = "params_%s" % str(params_count[0])

        trans_shape(desc, op_list)

        out = conv2d_compress_node(op_node, op_list)
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
                "use_L1_workspace": l1_workspace,
                "L1_fusion_type":  l1_fusion_type,
                "L1_addr_offset": l1_addr_offset}

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
        else:
            return {}
    except Exception:           # 'pylint: disable=bare-except
        return {}


def get_op_outputs_attrs(op_node):
    """
    get op outputs and attrs
    """
    if fusion_manager.has_op_params(op_node['name']):
        list_args = fusion_manager.get_op_args(op_node["name"])
        kwds_args = fusion_manager.get_op_kwds(op_node["name"])
        return (list_args, kwds_args)

    if 'prebuild_outs_attrs' in op_node:
        outs_attrs = op_node['prebuild_outs_attrs']
        list_args = outs_attrs['list_args']
        kwds_args = outs_attrs['kwds_args']
        return (list_args, kwds_args)

    return ([], {})


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
    with tvm.target.cce():
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
            op_impl_kwds = OpImplPolicy.get_op_impl_mode(op_compute_func,
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
        op_output_list, tvm.tensor.Tensor) else op_output_list

    try:
        if op_node["pattern"] == OpPatterns.ELEMWISE_PATTERN.value:
            for idx, tensor in enumerate(op_output_list):
                if tensor.dtype != all_out[idx]["dtype"]:
                    op_output_list[idx] = \
                        te.lang.cce.cast_to(op_output_list[idx],
                                            all_out[idx]["dtype"])
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
    # get json data
    try:
        json_data = json.loads(json_str)
    except Exception:
        raise RuntimeError('Read input data error')

    # get params from json_data
    fusion_op_name = str(json_data["fusion_op_name"])
    op_list = json_data["op_list"]

    # check input args
    check_kernel_name(fusion_op_name)

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
    for op_node in op_list:
        if op_node["type"] == "Data":
            # create placeholder
            create_placeholder_tensor(op_node, tensor_list, input_list,
                                      op_list, params_count)
            continue

        op_input_list = []
        dyn_input_dict = {}

        check_input_desc_not_in_op(op_node)

        add_input_tensor(op_node, tensor_list, op_input_list,
                         is_used_tensor_list, input_tensor_cnt, dyn_input_dict)

        op_output_list = call_op_compute(op_node, op_input_list,
                                         dyn_input_dict, fusion_op_name)

        check_output_desc_not_in_op(op_node)

        for output_desc in op_node["output_desc"]:
            check_output_desc_in_tensor(output_desc, tensor_list)

            output_tensor = op_output_list[output_desc["output_index"]]
            tensor_list[output_desc["name"]] = output_tensor
            compute_output_tensor_list.append(output_tensor)

            # record output tensor called by other tensor
            if output_tensor not in output_tensor_cnt:
                output_tensor_cnt[output_tensor] = 0
            tmp_cnt = output_tensor_cnt[output_tensor]
            output_tensor_cnt[output_tensor] = tmp_cnt + 1

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

    return output_list


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
                             "GeluGrad", "Add", 'AddN']
    matmul_fusion = False
    for op_node in op_list:
        if "pattern" in op_node:
            if op_node["pattern"] == OpPatterns.MATMUL_PATTERN.value:
                matmul_fusion = True
                break
    for op_node in op_list:
        if matmul_fusion and "pattern" in op_node:
            if (op_node["pattern"] == OpPatterns.ELEMWISE_PATTERN.value) and  \
                    (op_node["type"] not in matmul_elmt_fuse_type):
                raise RuntimeError(
                    "Matmul elementwise fusion only support ('Elu','Relu6'," \
                    "'LeakyRelu','Gelu','Softsign','Relu6','Relu','Selu'," \
                    "'Sigmoid','Tanh','Softplus','GeluGrad', 'Add', 'AddN'), " \
                    "not support fusion with '%s'" % op_node["type"])


def get_real_output(sch, output_list):
    """
    get_real_output
    """
    # some schedule will modify out tensor, need update real out tensor
    if sch.cce_special["real_out_tensor"]:
        real_output = sch.cce_special["real_out_tensor"]
    else:
        real_output = output_list
    return real_output


def check_single_op(json):
    """Check if the json string contains only one op

    Parameters
    ----------
    json : json description of fuson op

    Returns
    -------
    succ_flag : boolean
    """
    count = len([op for op in json['op_list'] if op['type'] != 'Data'])
    return True if count == 1 else False


def single_op_build(json_data):
    """call op's entry function directly if there's only one single op

    Parameters
    ----------
    json_data : dict
        json description of op

    """
    kernel_name = json_data['fusion_op_name']
    single_op = [op for op in json_data['op_list'] if op['type'] != 'Data'][0]
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
    kwargs = OpImplPolicy.get_op_impl_mode(opfunc, single_op['type'])
    fm.op_build_cfg_en()
    opfunc(*inputs, *list_args, kernel_name, **kwargs)


def dump_fusion_json(json_str, json_file_name):
    """
    dump fusion json to kernel_meta directory
    """
    if os.getenv('TE_DUMP_FUSIONOP_JSON') != '1':
        return

    kernel_path = os.path.join(os.getcwd(), "kernel_meta")
    try:
        if not os.path.exists(kernel_path):
            os.mkdir(kernel_path)
            os.chmod(kernel_path, stat.S_IRWXU + stat.S_IRGRP + stat.S_IXGRP)
    except FileExistsError:
        pass

    try:
        kernel_path = os.path.join(kernel_path,
                                   "fusion_op_{}.json".format(json_file_name))
        jsonf = open(kernel_path, "w")
        jsonf.write(json_str)
        jsonf.close()
    except Exception:       # 'pylint: disable=bare-except,broad-except
        return


def init_op_cfg(json_data):
    """
    init l1 size, etc...
    """
    if 'l1_size' in json_data:
        set_L1_info("op_L1_space", json_data['l1_size'])
    else:
        set_L1_info("op_L1_space", -1)


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def fusion_op(json_str):
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
    try:
        json_data = json.loads(json_str)
    except Exception:
        raise RuntimeError('Read input data error')

    init_op_cfg(json_data)

    dump_fusion_json(json_str, json_data["fusion_op_name"])

    if check_single_op(json_data):
        single_op_build(json_data)
        return

    fm.reset_fusion_build_cfg()

    # get params from json_data
    fusion_op_name = str(json_data["fusion_op_name"])
    op_list = json_data["op_list"]

    # check input args
    check_kernel_name(fusion_op_name)

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
    bool_storage_as_1bit_oplist = ["Asinh", "Atanh", "Acosh",
                                   "Asin", "Atan2", "Acos", "Pow"]
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

    # generate schedule
    with tvm.target.cce():
        # call auto_schedule
        sch = generic.auto_schedule(output_list)

    input_list = [ele for ele in input_list if ele is not None]

    real_output = get_real_output(sch, output_list)

    # codegen
    config = {"name": fusion_op_name,
              "tensor_list": input_list + real_output,
              "fusion_build_config": fm.get_fusion_build_cfg(),
              "bool_storage_as_1bit": cmp_bool_storage_as_1bit}

    te.lang.cce.cce_build_code(sch, config)
