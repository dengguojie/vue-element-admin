#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from unittest.mock import MagicMock
from unittest.mock import patch

from te import tvm
from te.tvm.target import cce
from tbe.dsl import auto_schedule
from impl.fully_connection import fully_connection_compute
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
from impl.fix_pipe import fixpipe_compute
from impl.trans_data import trans_data_compute

from op_test_frame.ut import OpUT
ut_case = OpUT("FullyConnection", None, None)


vals = {("CORE_NUM", ): 48,
        ("CUBE_VECTOR_SPLIT",): True,
        ("UB_SIZE", ): 196608,
        ("L0A_SIZE", ): 65536,
        ("L0B_SIZE", ): 65536,
        ("L1_SIZE", ): 524288,
        ("L0C_SIZE", ): 131072,
        ("Intrinsic_fix_pipe_l0c2out",): True,
        ("Intrinsic_fix_pipe_unit_list",): True,
        ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): True
        }
def side_effects(*args):
    return vals[args]



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

m_unalign_dts_case = {"params": [{'shape': (72, 576, 1, 1, 32), 'dtype': 'int8', 'format': 'NC1HWC0',
                                 "ori_format":"NC1HWC0", "ori_shape":(72, 576, 1, 1, 32)},
                                {'shape': (576, 2, 16, 32), 'dtype': 'int8', 'format': 'FRACTAL_Z',
                                 "ori_format":"FRACTAL_Z", "ori_shape":(576, 2, 16, 32)},
                                None, None,
                                {'shape': (1, 63, 1, 1, 16), 'dtype': 'int32', 'format': 'NC1HWC0',
                                 "ori_format":"NC1HWC0", "ori_shape":(1, 63, 1, 1, 16)},
                                72, False, 1, -128],
                     "case_name": "fully_connection_m_unalign_dts_case",
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
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], m_unalign_dts_case)

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
def test_fc_add_relu6_fusion_xbatch_910_input_x_case(test_arg):
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
ut_case.add_cust_test_func(test_func=test_fc_add_relu6_fusion_xbatch_910_input_x_case)
ut_case.add_cust_test_func(test_func=test_fc_add_relu6_fusion_xbatch_910_input_y_case)


def test_fc_5hd25hd_fp16():
    with cce():
        tensor_a = tvm.placeholder((64, 96), name="tensor_a", dtype="float16", attrs={"format": "NC1HWC0", "ori_shape": (64, 96)})
        tensor_b = tvm.placeholder((6, 8, 16, 16), name="tensor_b", dtype="float16", attrs={
            "format": "FRACTAL_Z", "ori_shape":(128, 2, 4, 12), "ori_format": "NCHW"})
        output = {"shape": (8, 4, 16, 16), "dtype": "float16", "format": "NC1HWC0", "ori_format": "ND"}
        res = fully_connection_compute(tensor_a, tensor_b, None, None, output, 1, False, 1)
        sch = auto_schedule(res)

def test_fc_nz2nz_fp16():
    with cce():
        tensor_a = tvm.placeholder((6, 4, 16, 16), name="tensor_a", dtype="float16", attrs={"format": "FRACTAL_NZ", "ori_shape": (64, 96)})
        tensor_b = tvm.placeholder((6, 8, 16, 16), name="tensor_b", dtype="float16", attrs={
            "format": "FRACTAL_Z", "ori_shape":(128, 24, 2, 2), "ori_format": "NCHW"})
        output = {"shape": (8, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
        res = fully_connection_compute(tensor_a, tensor_b, None, None, output, 1, False, 1)
        sch = auto_schedule(res)

def test_fc_5hd25hd_fp32():
    with cce():
        tensor_a = tvm.placeholder((64, 96), name="tensor_a", dtype="float32", attrs={"format": "NC1HWC0", "ori_shape": (64, 96)})
        tensor_b = tvm.placeholder((12, 8, 16, 8), name="tensor_b", dtype="float32", attrs={
            "format": "FRACTAL_Z", "ori_shape":(128, 2, 4, 12), "ori_format": "NCHW"})
        output = {"shape": (8, 4, 16, 16), "dtype": "float32", "format": "NC1HWC0", "ori_format": "ND"}
        res = fully_connection_compute(tensor_a, tensor_b, None, None, output, 1, False, 1)
        sch = auto_schedule(res)

def test_fc_nz2nz_fp32():
    with cce():
        tensor_a = tvm.placeholder((12, 4, 16, 8), name="tensor_a", dtype="float32", attrs={"format": "FRACTAL_NZ", "ori_shape": (64, 96)})
        tensor_b = tvm.placeholder((12, 8, 16, 8), name="tensor_b", dtype="float32", attrs={
            "format": "FRACTAL_Z", "ori_shape":(128, 24, 2, 2), "ori_format": "NCHW"})
        output = {"shape": (8, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_format": "ND"}
        res = fully_connection_compute(tensor_a, tensor_b, None, None, output, 1, False, 1)
        sch = auto_schedule(res)

def test_fc_fixpipe():
    with cce():
        tensor_a = tvm.placeholder((12, 4, 16, 8), name="tensor_a", dtype="float32", attrs={"format": "FRACTAL_NZ", "ori_shape": (64, 96)})
        tensor_b = tvm.placeholder((12, 8, 16, 8), name="tensor_b", dtype="float32", attrs={
            "format": "FRACTAL_Z", "ori_shape":(128, 24, 2, 2), "ori_format": "NCHW"})
        output = {"shape": (8, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_format": "ND"}
        fc_res = fully_connection_compute(tensor_a, tensor_b, None, None, output, 1, False, 1)
        output_dict = {"shape": (64, 128), "format": "ND", "dtype": "float32"}
        res = fixpipe_compute(fc_res, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        sch = auto_schedule(res)

def test_fc_fixpipe_nhwc():
    with cce():
        tensor_a = tvm.placeholder((12, 4, 16, 8), name="tensor_a", dtype="float32", attrs={
            "format": "FRACTAL_NZ", "ori_shape": (64, 96)})
        tensor_b = tvm.placeholder((12, 8, 16, 8), name="tensor_b", dtype="float32", attrs={
            "format": "FRACTAL_Z", "ori_shape":(128, 24, 2, 2), "ori_format": "NCHW"})
        output = {"shape": (8, 4, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_format": "ND"}
        fc_res = fully_connection_compute(tensor_a, tensor_b, None, None, output, 1, False, 1)
        output_dict = {"shape": (64, 1, 1, 128), "format": "NHWC", "dtype": "float32"}
        res = fixpipe_compute(fc_res, None, None, None, None, None, None, None, None, None, output_dict, [], [], "")
        sch = auto_schedule(res)

def test_split_fc_4(test_arg):
    x = {'shape': (8, 1, 2, 2, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 64)}
    w = {'shape': (4, 2, 16, 16), 'dtype': 'float16', 'format': 'FRACTAL_Z', "ori_format":"FRACTAL_Z", "ori_shape":(4, 2, 16, 16)}
    b =  {'shape': (1, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(1, 2, 1, 1, 16)}
    y = {'shape': (8, 2, 1, 1, 16), 'dtype': 'float16', 'format': 'NC1HWC0', "ori_format":"NC1HWC0", "ori_shape":(8, 2, 1, 1, 16)}
    get_op_support_info(x, w, b, None, y, 32, False, 1)
    op_select_format(x, w, b, None, y, 32, False, 1, 0)


def test_fc_nd_transdata():
    with cce():
        tensor_a_ori = tvm.placeholder((1, 1, 1, 8), name="tensor_a", dtype="float16")
        tensor_a = trans_data_compute(tensor_a_ori, None, "NHWC", "NC1HWC0")
        tensor_b = tvm.placeholder((1, 2, 16, 16), name="tensor_b", dtype="float16", attrs={
            "format": "FRACTAL_Z", "ori_shape":(1, 1, 8, 32), "ori_format": "NHWC"})
        output = {"shape": (1, 2, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC"}
        fc_res = fully_connection_compute(tensor_a, tensor_b, None, None, output, 1, False, 1)
        x2 = tvm.placeholder((1, 2, 1, 1, 16), name="add_01",
                             attrs={"format": "NC1HWC0", "ori_format": "NHWC",
                                    "shape":(1, 2, 1, 1, 16), "ori_shape":(1, 1, 1, 32)})
        res = fixpipe_compute(fc_res, x2, None, None, None, None, None, None, None, None, output, [], [], "")
        sch = auto_schedule(res)

# test mock case
def test_mock_cases(test_args):
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
        with patch("impl.util.platform_adapter.tbe_platform.intrinsic_check_support", MagicMock(side_effect=side_effects)):
            with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=side_effects)):
                test_fc_5hd25hd_fp16()
                test_fc_5hd25hd_fp32()
                test_fc_nz2nz_fp16()
                test_fc_nz2nz_fp32()
                test_fc_fixpipe()
                test_split_fc_3("")
                test_split_fc("")
                test_split_fc_1("")
                test_split_fc_2("")
                test_split_fc_4("")
                test_fc_fixpipe_nhwc()
                test_fc_nd_transdata()
                vals.update({
                    ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): False
                })
                test_fc_nd_transdata()

ut_case.add_cust_test_func(test_func=test_mock_cases)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
