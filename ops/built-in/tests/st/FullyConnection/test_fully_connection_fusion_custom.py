#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.fully_connection import fully_connection_compute
from impl.add import add_compute
from impl.relu6 import relu6_compute
from tbe.dsl import auto_schedule
from te import tvm
from te.lang.cce import cce_build_code
from te.platform.cce_conf import te_set_version
from te.tvm.target import cce

# fc+add+relu6 fusion case
def test_fc_add_relu6_fusion_1batch_910_case1():
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

def test_fc_add_relu6_fusion_1batch_910_case2():
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
def test_fc_add_relu6_fusion_xbatch_910_case1():
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
def test_fc_add_relu6_fusion_xbatch_910_case2():
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

if __name__ == '__main__':
    test_fc_add_relu6_fusion_1batch_910_case1()
    test_fc_add_relu6_fusion_1batch_910_case2()
    test_fc_add_relu6_fusion_xbatch_910_case1()
    test_fc_add_relu6_fusion_xbatch_910_case2()