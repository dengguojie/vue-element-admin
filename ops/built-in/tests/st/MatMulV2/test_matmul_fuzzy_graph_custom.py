#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from unittest.mock import patch
from unittest.mock import MagicMock

from impl.confusion_transpose_d import confusion_transpose_d_compute
from impl.dynamic.mat_mul import matmul_generalization
from impl.mat_mul import mat_mul_compute
from impl.mat_mul import mat_mul
from impl.ascend_requant import ascend_requant_compute
from tbe.common.context import op_context
from tbe.dsl import auto_schedule
from te import tvm
from te.lang.cce import cce_build_code
from te.platform.cce_conf import te_set_version
from te.tvm.target import cce


def test_matmul_generalization_upper_bound_input1():
    input_x1_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (16369, None)], 'ori_range': [(1, 48), (16369, None)]}
    input_x2_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (1, 48)], 'ori_range': [(1, 48), (1, 48)]}
    output_dynamic ={'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (16369, None)], 'ori_range': [(1, 48), (16369, None)]}
    bias_dynamic = None

    matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_y=output_dynamic,
                          trans_a=False, trans_b=False, kernel_name="matmul_generalization",
                          generalize_config={"mode": "keep_rank"})

def test_matmul_generalization_unknown_rank():
    input_x1_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (16369, 2147483647)], 'ori_range': [(1, 48), (16369, 2147483647)]}
    input_x2_dynamic = {'ori_shape': [-2], 'dtype': 'float16', 'shape': [-2], 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (1, 48)], 'ori_range': [(1, 48), (1, 48)]}
    output_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (16369, 2147483647)], 'ori_range': [(1, 48), (16369, 2147483647)]}
    bias_dynamic = None
    matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_y=output_dynamic,
                          trans_a=False, trans_b=False, kernel_name="matmul_generalization",
                          generalize_config={"mode": "keep_rank"})

def test_matmul_generalization_lower_bound_input2():
    input_x1_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (1, 48)], 'ori_range': [(1, 48), (1, 48)]}
    input_x2_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(2147483677, 2147483677), (1, 48)], 'ori_range': [(2147483677, 2147483677), (1, 48)]}
    output_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(2147483677, 2147483677), (1, 48)], 'ori_range': [(2147483677, 2147483677), (1, 48)]}
    bias_dynamic = {'ori_shape': -1, 'dtype': 'float16', 'shape': -1, 'format': 'ND', 'ori_format': 'ND', 'range': ((2147483677, 2147483677),), 'ori_range': ((2147483677, 2147483677),)}
    matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_y=output_dynamic,
                          trans_a=False, trans_b=False, kernel_name="matmul_generalization",
                          generalize_config={"mode": "keep_rank"})

def test_matmul_confusion_transpose_1_128_910():
    te_set_version("Ascend910")
    with cce():
        x1 = tvm.placeholder((64, 8, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (128, 1024)}, dtype="float16")
        x2 = tvm.placeholder((64, 64, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (1024, 1024)}, dtype="float16")
        output_y = {"shape": (64, 8, 16, 16), "dtype": "float16", "ori_shape": (128, 1024), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(x1, x2, None, None, output_y)
        y = {"shape": (1, 16, 4, 8, 16, 16), "ori_shape": (1, 16, 128, 64), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 1, 3], (1, 128, 16, 64), False)
        tensor_list = [x1, x2, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "matmul_confusion_transpose_910",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

def test_matmul_confusion_transpose_30_224_910():
    te_set_version("Ascend910")
    with cce():
        x1 = tvm.placeholder((64, 420, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (6720, 1024)}, dtype="float16")
        x2 = tvm.placeholder((64, 64, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (1024, 1024)}, dtype="float16")
        output_y = {"shape": (64, 420, 16, 16), "dtype": "float16", "ori_shape": (6720, 1024), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(x1, x2, None, None, output_y)
        y = {"shape": (30, 16, 4, 14, 16, 16), "ori_shape": (30, 16, 224, 64), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 1, 3], (30, 224, 16, 64), False)
        tensor_list = [x1, x2, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "matmul_confusion_transpose_910",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)


def test_matmul_confusion_transpose_30_224_310():
    te_set_version("Ascend310")
    try:
        with cce():
            x1 = tvm.placeholder((64, 420, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (6720, 1024)}, dtype="float16")
            x2 = tvm.placeholder((64, 64, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (1024, 1024)}, dtype="float16")
            output_y = {"shape": (64, 420, 16, 16), "dtype": "float16", "ori_shape": (6720, 1024), "format": "FRACTAL_NZ", "ori_format": "ND"}
            matmul_out = mat_mul_compute(x1, x2, None, None, output_y)
            y = {"shape": (30, 16, 4, 14, 16, 16), "ori_shape": (30, 16, 224, 64), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
            out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 1, 3], (30, 224, 16, 64), False)
    except RuntimeError as e:
        print("test_matmul_confusion_transpose_30_224_310 success")

def test_matmul_confusion_transpose_30_224_perm_invalid():
    te_set_version("Ascend910B")
    try:
        with cce():
            x1 = tvm.placeholder((64, 420, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (6720, 1024)}, dtype="float16")
            x2 = tvm.placeholder((64, 64, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (1024, 1024)}, dtype="float16")
            output_y = {"shape": (64, 420, 16, 16), "dtype": "float16", "ori_shape": (6720, 1024), "format": "FRACTAL_NZ", "ori_format": "ND"}
            matmul_out = mat_mul_compute(x1, x2, None, None, output_y)
            y = {"shape": (30, 16, 4, 14, 16, 16), "ori_shape": (30, 16, 224, 64), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
            out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 3, 1], (30, 224, 16, 64), False)
    except RuntimeError as e:
        print("test_matmul_confusion_transpose_30_224_perm_invalid success")


def test_matmul_confusion_transpose_710():
    te_set_version("Ascend710")
    with cce():
        x1 = tvm.placeholder((48, 64, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (1024, 768)}, dtype="float16")
        x2 = tvm.placeholder((48, 48, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (768, 768)}, dtype="float16")
        output_y = {"shape": (48, 64, 16, 16), "dtype": "float16", "ori_shape": (1024, 768), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(x1, x2, None, None, output_y)
        y = {"shape": (8, 12, 4, 8, 16, 16), "ori_shape": (8, 12, 128, 64), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 1, 3], (8, 128, 12, 64), False)
        tensor_list = [x1, x2, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "matmul_confusion_transpose_710",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

def test_matmul_requant_bl1_fullload_bind_multicore():
    def custom_tiling(*args):
        tiling = {'AL0_matrix': [125, 1, 16, 32, 1, 1], 'AL1_shape': [32, 1, 1, 1], 'AUB_channel_wise_flag': None, 'AUB_shape': None, 'A_overhead_opt_flag': 0, 'BL0_matrix': [], 'BL1_shape': [], 'BUB_channel_wise_flag': None, 'BUB_shape': None, 'B_overhead_opt_flag': 0, 'CL0_matrix': [1, 125, 16, 16, 1, 1], 'CUB_channel_wise_flag': False, 'CUB_matrix': [
            1, 125, 16, 16, 1, 1], 'batch_bef_group_flag': 0, 'block_dim': [1, 8, 1, 1], 'manual_pingpong_buffer': {'AL0_pbuffer': 1, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 1, 'BL1_pbuffer': 1, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1}, 'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}
        return tiling

    te_set_version("Ascend710")
    with patch("tbe.common.tiling.tiling_api.get_tiling", MagicMock(side_effect=custom_tiling())):
        with cce():
            input_0 = tvm.placeholder((16, 625, 16, 32), name="i0", dtype="int8",
                                      attrs={"ori_shape": (10000, 512),
                                             "format": "ND",
                                             "ori_format": "FRACTAL_NZ"})
            input_1 = tvm.placeholder((16, 16, 16, 32), name="i1", dtype="int8",
                                      attrs={"ori_shape": (512, 256),
                                      "format": "FRACTAL_Z",
                                             "ori_format": "HWCN"})
            input_2 = tvm.placeholder((256,), name="i2", dtype="int32",
                                      attrs={"ori_shape": (256),
                                             "format": "ND",
                                             "ori_format": "ND"})
            output_y1 = {"shape": (16, 625, 16, 16), "ori_shape": (10000, 256), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "int32"}
            output_mm = mat_mul_compute(input_0, input_1, input_2, None, output_y1)

            input_3 = tvm.placeholder((256,), name="i3", dtype="uint64",
                                      attrs={"ori_shape": (256,),
                                             "format": "NC1HWC0",
                                             "ori_format": "NCHW"})
            output_y2 = {"shape": (8, 625, 16, 32), "ori_shape": (10000, 256), "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "int8"}
            output_fusion_op = ascend_requant_compute(output_mm, input_3, output_y2, relu_flag=False)

            sch = auto_schedule(output_fusion_op)
            config = {"name": "test_matmul_requant_bl1_fullload_bind_multicore", "tensor_list":[input_0, input_1, input_2, output_fusion_op]}
            cce_build_code(sch, config)

def test_matmul_requant_bl1_bl0_status_ori_equal():
    def custom_tiling(*args):
        tiling = {'AL0_matrix': [2, 1, 16, 16, 1, 1], 'AL1_shape': [32, 1, 1, 1], 'AUB_channel_wise_flag': None, 'AUB_shape': None, 'A_overhead_opt_flag': 0, 'BL0_matrix': [2, 2, 16, 16, 1, 1], 'BL1_shape': [32, 1, 1, 1], 'BUB_channel_wise_flag': None, 'BUB_shape': None, 'B_overhead_opt_flag': 0, 'CL0_matrix': [2, 2, 16, 16, 1, 1], 'CUB_channel_wise_flag': False, 'CUB_matrix': [
            2, 2, 16, 16, 1, 1], 'batch_bef_group_flag': 0, 'block_dim': [1, 1, 1, 1], 'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2, 'BL1_pbuffer': 1, 'BUB_pbuffer': 1, 'CL0_pbuffer': 1, 'CUB_pbuffer': 2, 'UBG_pbuffer': 1}, 'n_bef_batch_flag': 0, 'n_bef_group_flag': 0, 'tbe_compile_para': 0}
        return tiling

    te_set_version("Ascend910")
    with patch("tbe.common.tiling.tiling_api.get_tiling", MagicMock(side_effect=custom_tiling())):
        with cce():
            input_x1 = {'shape': [2,2,16,16], 'ori_shape': [32,32], 'format': 'FRACTAL_NZ', 'ori_format': 'ND', 'dtype': 'float16'}
            input_x2 = {'shape': [2,2,16,16], 'ori_shape': [32,32], 'format': 'FRACTAL_NZ', 'ori_format': 'ND', 'dtype': 'float16'}
            bias = None
            output_y = {'shape': [2,2,16,16], 'ori_shape': [32,32], 'format': 'FRACTAL_NZ', 'ori_format': 'ND', 'dtype': 'float16'}
            mat_mul(input_x1, input_x2, bias, output_y=output_y)

if __name__ == '__main__':
    test_matmul_generalization_upper_bound_input1()
    test_matmul_generalization_unknown_rank()
    test_matmul_generalization_lower_bound_input2()
    test_matmul_confusion_transpose_1_128_910()
    test_matmul_confusion_transpose_30_224_910()
    test_matmul_confusion_transpose_30_224_310()
    test_matmul_confusion_transpose_30_224_perm_invalid()
    test_matmul_confusion_transpose_710()
