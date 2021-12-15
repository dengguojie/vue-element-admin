#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.fully_connection import get_op_support_info
from impl.fully_connection import op_select_format
from impl.fully_connection import fully_connection_compute
from impl.add import add_compute
from impl.relu6 import relu6_compute
from tbe.dsl import auto_schedule
from te import tvm
from te.lang.cce import cce_build_code
from te.platform.cce_conf import te_set_version
from te.tvm.target import cce
from op_test_frame.ut import OpUT
ut_case = OpUT("FullyConnection", None, None)

case1 = {"params": [{'shape': (1, 2, 4, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 2, 4, 2, 16)},
                    {'shape': (16, 1, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                     "ori_format":"FRACTAL_Z", "ori_shape":(16, 1, 16, 16)},
                    None, None,
                    {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 1, 1, 1, 16)},
                    16, False, 1, 0],
         "case_name": "fully_connection_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 16)},
                    {'shape': (4, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                     "ori_format":"FRACTAL_Z", "ori_shape":(4, 2, 16, 16)},
                    {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)},
                    None,
                    {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 1, 1, 1, 16)},
                    32, False, 1, 0],
         "case_name": "fully_connection_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{'shape': (256, 19, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ',
                     "ori_format":"NCHW", "ori_shape":(304, 4096)},
                    {'shape': (256, 256, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                     "ori_format":"NCHW", "ori_shape":(4096, 4096, 1, 1)},
                    {'shape': (1, 256, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NCHW", "ori_shape":(4096, )},
                    None,
                    {'shape': (256, 19, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ',
                     "ori_format":"NCHW", "ori_shape":(304, 4096)},
                    4096, False, 1, 0],
         "case_name": "fully_connection_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 16)},
                    {'shape': (1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                     "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 16)},
                    {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)},
                    None,
                    {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 1, 1, 1, 16)},
                    32, False, 1, 0],
         "case_name": "fully_connection_failed_0",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{'shape': (1, 32, 4, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 32, 4, 2, 16)},
                    {'shape': (256, 256, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                     "ori_format":"FRACTAL_Z", "ori_shape":(256, 256, 16, 16)},
                    None, None,
                    {'shape': (1, 256, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                     "ori_format":"NC1HWC0", "ori_shape":(1, 256, 1, 1, 16)},
                    16, False, 1, 0],
         "case_name": "fully_connection_gemm_mode",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

gevm_L1_attach_equal = {"params": [{'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                        "ori_format":"NHWC", "ori_shape":(1, 1, 1, 16)},
                        {'shape': (1, 1, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z',
                        "ori_format":"HWCN", "ori_shape":(1, 1, 16, 8)},
                        {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                        "ori_format":"NHWC", "ori_shape":(8, )},
                        None,
                        {'shape': (1, 1, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0',
                        "ori_format":"NHWC", "ori_shape":(1, 1, 1, 8)},
                        32, False, 1, 0],
                        "case_name": "gevm_L1_attach_equal",
                        "expect": "success",
                        "format_expect": [],
                        "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], gevm_L1_attach_equal)
# ND -> ND
def test_split_fc(test_arg):
    x = {'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 16)}
    w = {'shape': (4, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(4, 2, 16, 16)}
    b =  {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)}
    y = {'shape': (8, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 2, 1, 1, 16)}
    get_op_support_info(x, w, b, None, y, 32, False, 1)
    op_select_format(x, w, b, None, y, 32, False, 1, 0)

# NZ -> NZ with batch
def test_split_fc_1(test_arg):
    x = {'shape': (2, 1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(2, 1, 2, 16, 16)}
    w = {'shape': (1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 16)}
    b =  {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)}
    y = {'shape': (2, 2, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(2, 2, 2, 16, 16)}
    get_op_support_info(x, w, b, None, y, 32, False, 2)
    op_select_format(x, w, b, None, y, 32, False, 2, 0)

# NZ -> NZ no batch
def test_split_fc_2(test_arg):
    x = {'shape': (1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(1, 2, 16, 16)}
    w = {'shape': (1, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 16)}
    y = {'shape': (2, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(2, 2, 16, 16)}
    get_op_support_info(x, w, None, None, y, 32, False, 1)
    op_select_format(x, w, None, None, y, 32, False, 1, 0)

# ND -> NZ
def test_split_fc_3(test_arg):
    x = {'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 1, 2, 2, 16)}
    w = {'shape': (4, 1, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(1, 2, 16, 16)}
    y = {'shape': (1, 1, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_NZ', "ori_format":"FRACTAL_NZ", "ori_shape":(1, 1, 16, 16)}
    get_op_support_info(x, w, None, None, y, 16, False, 1)
    op_select_format(x, w, None, None, y, 16, False, 1, 0)

# fc+add+relu6 fusion case
def test_fc_add_relu6_fusion_1batch_910_case1(test_arg):
    te_set_version("Ascend910")
    with cce():
        tensor_x = tvm.placeholder((1, 32), name="tensor_a", dtype="float16", attrs={'format': "NC1HWC0",
                                   "ori_shape": (1, 1, 1, 32)})
        tensor_w = tvm.placeholder((2, 8, 16, 16), name="tensor_b", dtype="float16", attrs={'format': "FRACTAL_Z",
                                   "ori_shape": (1, 1, 32, 128)})
        tensor_b = tvm.placeholder((128,), name="tensor_bias", dtype="float16", attrs={'format': "NC1HWC0",
                                   "ori_shape": (128,)})
        y = {"shape": (1, 8, 1, 1, 16), "ori_shape": (1, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
            "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [1, 8, 1, 1, 16],
            "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
            "valid_shape": (), "split_index": 0, "param_name": 'y'}
        fc_out = fully_connection_compute(tensor_x, tensor_w, tensor_b, None, y, 128, False, 1, 0,
                                          kernel_name="fully_connection")
        data_y = tvm.placeholder((1, 1, 1, 1, 1), name="data_2", dtype="float16", attrs={'format': "NHWC",
                                 "ori_shape": (1,)})
        output_z = {"shape": (1, 8, 1, 1, 16), "ori_shape": (1, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
            "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [1, 8, 1, 1, 16],
            "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
            "valid_shape": (), "split_index": 0, "param_name": 'output_z'}
        add_out = add_compute(fc_out, data_y, output_z, False, True, kernel_name="add")
        output_y = {"shape": (1, 8, 1, 1, 16), "ori_shape": (1, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
                    "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [1, 8, 1, 1, 16],
                    "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
                    "valid_shape": (), "split_index": 0, "param_name": 'output_y'}
        out = relu6_compute(add_out, output_y, kernel_name="relu6")
        tensor_list = [tensor_x, tensor_w, tensor_b, data_y, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "fc_add_relu6_fusion_1batch_910_case1",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

def test_fc_add_relu6_fusion_1batch_910_case2(test_arg):
    te_set_version("Ascend910")
    with cce():
        tensor_x = tvm.placeholder((1, 32), name="tensor_a", dtype="float16", attrs={'format': "NC1HWC0",
                                   "ori_shape": (1, 1, 1, 32)})
        tensor_w = tvm.placeholder((2, 8, 16, 16), name="tensor_b", dtype="float16", attrs={'format': "FRACTAL_Z",
                                   "ori_shape": (1, 1, 32, 128)})
        tensor_b = tvm.placeholder((128,), name="tensor_bias", dtype="float16", attrs={'format': "NC1HWC0",
                                   "ori_shape": (128,)})
        y = {"shape": (1, 8, 1, 1, 16), "ori_shape": (1, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
            "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [1, 8, 1, 1, 16],
            "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
            "valid_shape": (), "split_index": 0, "param_name": 'y'}
        fc_out = fully_connection_compute(tensor_x, tensor_w, tensor_b, None, y, 128, False, 1, 0,
                                          kernel_name="fully_connection")
        data_y = tvm.placeholder((1, 1, 1, 1, 1), name="data_2", dtype="float16", attrs={'format': "NHWC",
                                 "ori_shape": (1,)})
        output_z = {"shape": (1, 8, 1, 1, 16), "ori_shape": (1, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
            "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [1, 8, 1, 1, 16],
            "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
            "valid_shape": (), "split_index": 0, "param_name": 'output_z'}
        add_out = add_compute(data_y, fc_out, output_z, False, True, kernel_name="add")
        output_y = {"shape": (1, 8, 1, 1, 16), "ori_shape": (1, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
                    "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [1, 8, 1, 1, 16],
                    "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
                    "valid_shape": (), "split_index": 0, "param_name": 'output_y'}
        out = relu6_compute(add_out, output_y, kernel_name="relu6")
        tensor_list = [tensor_x, tensor_w, tensor_b, data_y, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "fc_add_relu6_fusion_1batch_910_case2",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

# fc+add+relu6 x batch fusion case
def test_fc_add_relu6_fusion_xbatch_910_case1(test_arg):
    te_set_version("Ascend910")
    with cce():
        tensor_x = tvm.placeholder((8, 32), name="tensor_a", dtype="float16", attrs={'format': "NC1HWC0",
                                   "ori_shape": (8, 1, 1, 32)})
        tensor_w = tvm.placeholder((2, 8, 16, 16), name="tensor_b", dtype="float16", attrs={'format': "FRACTAL_Z",
                                   "ori_shape": (1, 1, 32, 128)})
        tensor_b = tvm.placeholder((128,), name="tensor_bias", dtype="float16", attrs={'format': "NC1HWC0",
                                   "ori_shape": (128,)})
        y = {"shape": (8, 8, 1, 1, 16), "ori_shape": (8, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
            "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [8, 8, 1, 1, 16],
            "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
            "valid_shape": (), "split_index": 0, "param_name": 'y'}
        fc_out = fully_connection_compute(tensor_x, tensor_w, tensor_b, None, y, 128, False, 1, 0,
                                          kernel_name="fully_connection")
        data_y = tvm.placeholder((1, 1, 1, 1, 1), name="data_2", dtype="float16", attrs={'format': "NHWC",
                                 "ori_shape": (1,)})
        output_z = {"shape": (8, 8, 1, 1, 16), "ori_shape": (8, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
            "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [8, 8, 1, 1, 16],
            "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
            "valid_shape": (), "split_index": 0, "param_name": 'output_z'}
        add_out = add_compute(fc_out, data_y, output_z, False, True, kernel_name="add")
        output_y = {"shape": (8, 8, 1, 1, 16), "ori_shape": (8, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
                    "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [8, 8, 1, 1, 16],
                    "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
                    "valid_shape": (), "split_index": 0, "param_name": 'output_y'}
        out = relu6_compute(add_out, output_y, kernel_name="relu6")
        tensor_list = [tensor_x, tensor_w, tensor_b, data_y, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "fc_add_relu6_fusion_xbatch_910_case1",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

# fc+add+relu6 x batch fusion case
def test_fc_add_relu6_fusion_xbatch_910_input_y_case(test_arg):
    te_set_version("Ascend910")
    with cce():
        tensor_x = tvm.placeholder((8, 32), name="tensor_a", dtype="float16", attrs={'format': "NC1HWC0",
                                   "ori_shape": (8, 1, 1, 32)})
        tensor_w = tvm.placeholder((2, 8, 16, 16), name="tensor_b", dtype="float16", attrs={'format': "FRACTAL_Z",
                                   "ori_shape": (1, 1, 32, 128)})
        tensor_b = tvm.placeholder((128,), name="tensor_bias", dtype="float16", attrs={'format': "NC1HWC0",
                                   "ori_shape": (128,)})
        y = {"shape": (8, 8, 1, 1, 16), "ori_shape": (8, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
            "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [8, 8, 1, 1, 16],
            "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
            "valid_shape": (), "split_index": 0, "param_name": 'y'}
        fc_out = fully_connection_compute(tensor_x, tensor_w, tensor_b, None, y, 128, False, 1, 0,
                                          kernel_name="fully_connection")
        data_y = tvm.placeholder((1, 1, 1, 1, 1), name="data_2", dtype="float16", attrs={'format': "NHWC",
                                 "ori_shape": (1,)})
        output_z = {"shape": (8, 8, 1, 1, 16), "ori_shape": (8, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
            "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [8, 8, 1, 1, 16],
            "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
            "valid_shape": (), "split_index": 0, "param_name": 'output_z'}
        add_out = add_compute(data_y, fc_out, output_z, False, True, kernel_name="add")
        output_y = {"shape": (8, 8, 1, 1, 16), "ori_shape": (8, 1, 1, 128), "format": "NC1HWC0", "sub_format": 0,
                    "ori_format": "NHWC", "dtype": "float16", "addr_type": 0, "total_shape": [8, 8, 1, 1, 16],
                    "slice_offset": (), "L1_addr_offset": 0, "L1_fusion_type": -1, "L1_workspace_size": -1,
                    "valid_shape": (), "split_index": 0, "param_name": 'output_y'}
        out = relu6_compute(add_out, output_y, kernel_name="relu6")
        tensor_list = [tensor_x, tensor_w, tensor_b, data_y, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "fc_add_relu6_fusion_xbatch_910_case2",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

ut_case.add_cust_test_func(test_func=test_split_fc)
ut_case.add_cust_test_func(test_func=test_split_fc_1)
ut_case.add_cust_test_func(test_func=test_split_fc_2)
ut_case.add_cust_test_func(test_func=test_split_fc_3)
ut_case.add_cust_test_func(test_func=test_fc_add_relu6_fusion_1batch_910_case1)
ut_case.add_cust_test_func(test_func=test_fc_add_relu6_fusion_1batch_910_case2)
ut_case.add_cust_test_func(test_func=test_fc_add_relu6_fusion_xbatch_910_case1)
ut_case.add_cust_test_func(test_func=test_fc_add_relu6_fusion_xbatch_910_input_y_case)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
