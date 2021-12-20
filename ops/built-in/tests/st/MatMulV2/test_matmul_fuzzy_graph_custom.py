#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.confusion_transpose_d import confusion_transpose_d_compute
from impl.dynamic.mat_mul import matmul_generalization
from impl.mat_mul import mat_mul_compute
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

if __name__ == '__main__':
    test_matmul_generalization_upper_bound_input1()
    test_matmul_generalization_unknown_rank()
    test_matmul_generalization_lower_bound_input2()
    test_matmul_confusion_transpose_1_128_910()
    test_matmul_confusion_transpose_30_224_910()
    test_matmul_confusion_transpose_30_224_310()
    test_matmul_confusion_transpose_30_224_perm_invalid()
    test_matmul_confusion_transpose_710()
