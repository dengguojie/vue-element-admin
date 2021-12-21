#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from te import tvm
from te.tvm.target import cce
from tbe.dsl import auto_schedule
from impl.mat_mul import mat_mul_compute
from impl.trans_data import trans_data_compute
from impl.ascend_dequant import ascend_dequant_compute
from impl.ascend_requant import ascend_requant_compute
from impl.add import add_compute
from impl.fix_pipe import fixpipe_compute
from impl.fix_pipe import fix_pipe


def test_matmul_ND2ND_fp16():
    with cce():
        x1 = tvm.placeholder((16, 32), name="x1", dtype="float16", attrs={"ori_shape": (16, 32), "format": "ND", "ori_format": "ND"})
        x2 = tvm.placeholder((32, 16), name="x2", dtype="float16", attrs={"ori_shape": (32, 16), "format": "ND", "ori_format": "ND"})
        x1_trans = trans_data_compute(x1, None, "ND", "FRACTAL_NZ")
        x2_trans = trans_data_compute(x2, None, "ND", "FRACTAL_NZ")
        y = {"shape": (1, 1, 16, 16), "ori_shape": (16, 16), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        dx_res = mat_mul_compute(x1_trans, x2_trans, None, None, y, False, False, 0)
        trans_out = {"shape": (16, 16), "ori_shape": (16, 16), "format": "ND", "ori_format": "ND", "dtype": "float16"}
        out = trans_data_compute(dx_res, trans_out, "FRACTAL_NZ", "ND")
        sch = auto_schedule(out)

def test_matmul_ND2ND_int8():
    with cce():
        tensor_a_ori = tvm.placeholder((64, 96), name="tensor_a_ori", dtype="int8")
        tensor_b_ori = tvm.placeholder((64, 128), name="tensor_b_ori", dtype="int8")
        tensor_a = trans_data_compute(tensor_a_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        tensor_b = trans_data_compute(tensor_b_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        output_y = {"shape": (8, 6, 16, 16), "dtype": "int32", "ori_shape": (96, 128), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(tensor_a, tensor_b, None, None, output_y, True, False, 0)
        out = trans_data_compute(matmul_out, None, src_format="FRACTAL_NZ", dst_format="ND")
        tensor_list = [tensor_a_ori, tensor_b_ori, out]
        sch = auto_schedule(out)

def test_matmul_ND2ND_fp32():
    with cce():
        tensor_a_ori = tvm.placeholder((64, 96), name="tensor_a_ori", dtype="float32")
        tensor_b_ori = tvm.placeholder((64,128), name="tensor_b_ori", dtype="float32")
        tensor_a = trans_data_compute(tensor_a_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        tensor_b = trans_data_compute(tensor_b_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        output_y = {"shape": (16, 6, 16, 8), "dtype": "float32", "ori_shape": (96, 128), "format": "ND", "ori_format": "ND"}
        matmul_out = mat_mul_compute(tensor_a, tensor_b, None, None, output_y, True, False, 0)
        out = trans_data_compute(matmul_out, None, src_format="FRACTAL_NZ", dst_format="ND")
        tensor_list = [tensor_a_ori, tensor_b_ori, out]
        sch = auto_schedule(out)

def test_matmul_ND2ND_fp32_1():
    with cce():
        tensor_a_ori = tvm.placeholder((64, 32), name="tensor_a_ori", dtype="float32")
        tensor_b_ori = tvm.placeholder((96, 32), name="tensor_b_ori", dtype="float32")
        tensor_a = trans_data_compute(tensor_a_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        tensor_b = trans_data_compute(tensor_b_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        output_y = {"shape": (12, 4, 16, 8), "dtype": "float32", "ori_shape": (64, 96), 
                    "format": "ND", "ori_format": "ND"}
        matmul_out = mat_mul_compute(tensor_a, tensor_b, None, None, output_y, False, True, 0)
        out = trans_data_compute(matmul_out, None, src_format="FRACTAL_NZ", dst_format="ND")
        tensor_list = [tensor_a_ori, tensor_b_ori, out]
        sch = auto_schedule(out)

def test_matmul_NZ2ND_fp16():
    with cce():
        x1 = tvm.placeholder((1, 2, 16, 16), name="x1", dtype="float16", attrs={"ori_shape": (32, 16), "format": "FRACTAL_NZ", "ori_format": "ND"})
        x2 = tvm.placeholder((1, 2, 16, 16), name="x2", dtype="float16", attrs={"ori_shape": (32, 16), "format": "FRACTAL_NZ", "ori_format": "ND"})
        y = {"shape": (2, 2, 16, 16), "ori_shape": (32, 32), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        dx_res = mat_mul_compute(x1, x2, None, None, y, False, True, 0)
        trans_out = {"shape": (32, 32), "ori_shape": (32, 32), "format": "ND", "ori_format": "ND", "dtype": "float16"}
        out = trans_data_compute(dx_res, trans_out, "FRACTAL_NZ", "ND")
        sch = auto_schedule(out)

def test_matmul_ND2NZ_fp16():
    with cce():
        x1 = tvm.placeholder((64, 1024), name="x1", dtype="float16", attrs={"ori_shape": (64, 1024), "format": "ND", "ori_format": "ND"})
        x2 = tvm.placeholder((1024, 32), name="x2", dtype="float16", attrs={"ori_shape": (1024, 32), "format": "ND", "ori_format": "ND"})
        x1_trans = trans_data_compute(x1, None, "ND", "FRACTAL_NZ")
        x2_trans = trans_data_compute(x2, None, "ND", "FRACTAL_NZ")
        y = {"shape": (2, 4, 16, 16), "ori_shape": (64, 32), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        bias = tvm.placeholder((32,), name="bias", dtype="float32", attrs={"ori_shape": (32,), "format": "ND", "ori_format": "ND"})
        out = mat_mul_compute(x1_trans, x2_trans, bias, None, y, False, False, 0)
        sch = auto_schedule(out)

def test_matmul_dequant():
    with cce():
        x1 = tvm.placeholder((64, 1, 16, 32), name="tensor_a", dtype="int8", attrs={"ori_shape": (2, 2048), "format": "FRACTAL_NZ", "ori_format": "ND"})
        x2 = tvm.placeholder((32, 128, 16, 32), name="tensor_b", dtype="int8", attrs={"ori_shape": (2048, 1001), "format": "FRACTAL_NZ", "ori_format": "ND"})
        bias = tvm.placeholder((1008,), name="tensor_bias", dtype="int32", attrs={"format": "ND", "ori_format": "ND", "ori_shape": (1001,)})
        output_y = {"shape": (63, 1, 16, 16), "dtype": "int32", "ori_shape": (2, 1001), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(x1, x2, bias, None, output_y, False, False, 0)
        deq_scale = tvm.placeholder((1, 63, 1, 1, 16), name='deq_scale', dtype="uint64", attrs={"ori_shape": (1001, ), "format": "NC1HWC0", "ori_format": "ND"})
        res = ascend_dequant_compute(matmul_out, deq_scale, None, sqrt_mode=False, relu_flag=False, kernel_name="ascend_dequant")
        tensor_list = [x1, x2, bias, deq_scale, res]
        sch = auto_schedule(res)

def test_matmul_dequant_1():
    with cce():
        x1 = tvm.placeholder((32, 2, 16, 32), name="tensor_a", dtype="int8", attrs={"ori_shape": (32, 1024), "format": "FRACTAL_NZ", "ori_format": "ND"})
        x2 = tvm.placeholder((1, 2, 16, 32), name="tensor_b", dtype="int8", attrs={"ori_shape": (32, 32), "format": "FRACTAL_Z", "ori_format": "ND"})
        bias = tvm.placeholder((32,), name="tensor_bias", dtype="int32", attrs={"format": "ND", "ori_format": "ND", "ori_shape": (32,)})
        output_y = {"shape": (2, 64, 16, 16), "dtype": "int32", "ori_shape": (1024, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(x1, x2, bias, None, output_y, True, False, 0)
        deq_scale = tvm.placeholder((1, 2, 1, 1, 16), name='deq_scale', dtype="uint64", attrs={"ori_shape": (32, ), "format": "NC1HWC0", "ori_format": "ND"})
        res = ascend_dequant_compute(matmul_out, deq_scale, None, sqrt_mode=False, relu_flag=False, kernel_name="ascend_dequant")
        tensor_list = [x1, x2, bias, deq_scale, res]
        sch = auto_schedule(res)

def test_matmul_ND2ND_fp16_batch():
    with cce():
        x1 = tvm.placeholder((5, 32, 16), name="x1", dtype="float16", attrs={"ori_shape": (5, 32, 16), "format": "ND", "ori_format": "ND"})
        x2 = tvm.placeholder((32, 16), name="x2", dtype="float16", attrs={"ori_shape": (5, 32, 16), "format": "ND", "ori_format": "ND"})
        x1_trans = trans_data_compute(x1, None, "ND", "FRACTAL_NZ")
        x2_trans = trans_data_compute(x2, None, "ND", "FRACTAL_NZ")
        y = {"shape": (5, 1, 1, 16, 16), "ori_shape": (16, 16), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        dx_res = mat_mul_compute(x1_trans, x2_trans, None, None, y, True, False, 0)
        trans_out = {"shape": (5, 16, 16), "ori_shape": (5, 16, 16), "format": "ND", "ori_format": "ND", "dtype": "float16"}
        out = trans_data_compute(dx_res, trans_out, "FRACTAL_NZ", "ND")
        sch = auto_schedule(out)

def test_matmul_requant():
    with cce():
        input_x1 = tvm.placeholder((4, 2, 16, 32), name="x1", dtype="int8", attrs={"ori_shape": (32, 128), "format": "FRACTAL_NZ", "ori_format": "ND"})
        input_x2 = tvm.placeholder((4, 2, 16, 32), name="x2", dtype="int8", attrs={"ori_shape": (128, 32), "format": "FRACTAL_Z", "ori_format": "ND"})
        bias = tvm.placeholder((32,), name="bias", dtype="int32", attrs={"ori_shape": (32,), "format": "ND", "ori_format": "ND"})
        output_y = {"shape": (2, 2, 16, 16), "ori_shape": (32, 32), "dtype": "int32", "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(input_x1, input_x2, bias, None, output_y, False, False, 0)
        req_scale = tvm.placeholder((1, 2, 1, 1, 16), name="deq_scale", dtype="uint64", attrs={"ori_shape": (32,), "format": "NC1HWC0", "ori_format": "ND"})
        out = ascend_requant_compute(matmul_out, req_scale, None, False)
        tensor_list = [input_x1, input_x2, bias, req_scale, out]
        sch = auto_schedule(out)

def test_matmul_requant_1():
    with cce():
        input_x1 = tvm.placeholder((4, 2, 16, 32), name="x1", dtype="int8", attrs={"ori_shape": (32, 128), "format": "FRACTAL_NZ", "ori_format": "ND"})
        input_x2 = tvm.placeholder((4, 2, 16, 32), name="x2", dtype="int8", attrs={"ori_shape": (128, 32), "format": "FRACTAL_Z", "ori_format": "ND"})
        bias = tvm.placeholder((32,), name="bias", dtype="int32", attrs={"ori_shape": (32,), "format": "ND", "ori_format": "ND"})
        output_y = {"shape": (2, 2, 16, 16), "ori_shape": (32, 32), "dtype": "int32", "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(input_x1, input_x2, bias, None, output_y, False, False, 0)
        req_scale = tvm.placeholder((1, 2, 1, 1, 16), name="deq_scale", dtype="uint64", attrs={"ori_shape": (32,), "format": "NC1HWC0", "ori_format": "ND"})
        out = ascend_requant_compute(matmul_out, req_scale, None, False)
        tensor_list = [input_x1, input_x2, bias, req_scale, out]
        sch = auto_schedule(out)

def test_matmul_NZ2NZ_fp16():
    with cce():
        x1 = tvm.placeholder((8, 4, 16, 16), name="x1", dtype="float16", attrs={"ori_shape": (64, 128), "format": "FRACTAL_NZ", "ori_format": "ND"})
        x2 = tvm.placeholder((4, 8, 16, 16), name="x2", dtype="float16", attrs={"ori_shape": (128, 64), "format": "FRACTAL_NZ", "ori_format": "ND"})
        y = {"shape": (8, 8, 16, 16), "ori_shape": (128, 128), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        out = mat_mul_compute(x1, x2, None, None, y, True, True, 0)
        sch = auto_schedule(out)

def test_matmul_NZ2NZ_int8():
    with cce():
        tensor_a_ori = tvm.placeholder((3, 4, 16, 32), name="tensor_a", dtype="int8", attrs={"format": "FRACTAL_NZ", "ori_format": "ND", "ori_shape":(64, 96)})
        tensor_b_ori = tvm.placeholder((4, 4, 16, 32), name="tensor_b", dtype="int8", attrs={"format": "FRACTAL_NZ", "ori_format": "ND", "ori_shape": (64, 128)})
        bias = tvm.placeholder((1024,), name="tensor_bias", dtype="int32", attrs={"format": "ND", "ori_format": "ND"})
        output_y = {"shape": (8, 6, 16, 16), "dtype": "int32", "ori_shape": (96, 128), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(tensor_a_ori, tensor_b_ori, bias, None, output_y, True, False, 0)
        tensor_list = [tensor_a_ori, tensor_b_ori, bias, matmul_out]
        sch = auto_schedule(matmul_out)

def test_matmul_add():
    with cce():
        x1 = tvm.placeholder((8, 4, 16, 16), name="x1", dtype="float16", attrs={"ori_shape": (64, 128), "format": "FRACTAL_NZ", "ori_format": "ND"})
        x2 = tvm.placeholder((4, 8, 16, 16), name="x2", dtype="float16", attrs={"ori_shape": (128, 64), "format": "FRACTAL_NZ", "ori_format": "ND"})
        y = {"shape": (8, 8, 16, 16), "ori_shape": (128, 128), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        matmul_out = mat_mul_compute(x1, x2, None, None, y, True, True, 0)
        x3 = tvm.placeholder((8, 8, 16, 16), name='add_input', dtype="float16", attrs={"ori_shape": (128, 128), "format": "FRACTAL_NZ", "ori_format": "ND"})
        out = add_compute(matmul_out, x3, None)
        tensor_list = [x1, x2, x3, out]
        sch = auto_schedule(out)

def test_matmul_add_not_align():
     with cce():
        x1 = tvm.placeholder((2, 2, 16, 16), name="x1", dtype="float16", attrs={"ori_shape": (32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"})
        x2 = tvm.placeholder((2, 2, 16, 16), name="x2", dtype="float16", attrs={"ori_shape": (32, 30), "format": "FRACTAL_NZ", "ori_format": "ND"})
        y = {"shape": (2, 2, 16, 16), "ori_shape": (32, 30), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        trans_out = {"shape": (32, 30), "ori_shape": (32, 30), "format": "ND", "ori_format": "ND", "dtype": "float16"}
        matmul_out = mat_mul_compute(x1, x2, None, None, y, False, False, 0)
        trans_out = trans_data_compute(matmul_out, trans_out, "FRACTAL_NZ", "ND")
        x3 = tvm.placeholder((1,), name='add_input', dtype="float16", attrs={"ori_shape": (128, 128), "format": "FRACTAL_NZ", "ori_format": "ND"})
        out = add_compute(trans_out, x3, None)
        tensor_list = [x1, x2, x3, out]
        sch = auto_schedule(out)   

def test_matmul_dequant_add():
    with cce():
        x1 = tvm.placeholder((32, 2, 16, 32), name="tensor_a", dtype="int8", attrs={"ori_shape": (32, 1024), "format": "FRACTAL_NZ", "ori_format": "ND"})
        x2 = tvm.placeholder((1, 2, 16, 32), name="tensor_b", dtype="int8", attrs={"ori_shape": (32, 32), "format": "FRACTAL_Z", "ori_format": "ND"})
        bias = tvm.placeholder((32,), name="tensor_bias", dtype="int32", attrs={"format": "ND", "ori_format": "ND", "ori_shape": (32,)})
        output_y = {"shape": (2, 64, 16, 16), "dtype": "int32", "ori_shape": (1024, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(x1, x2, bias, None, output_y, True, False, 0)
        deq_scale = tvm.placeholder((1, 2, 1, 1, 16), name='deq_scale', dtype="uint64", attrs={"ori_shape": (32, ), "format": "NC1HWC0", "ori_format": "ND"})
        deq_out = ascend_dequant_compute(matmul_out, deq_scale, None, sqrt_mode=False, relu_flag=False, kernel_name="ascend_dequant")
        x3 = tvm.placeholder((2, 64, 16, 16), name='add_input', dtype="float16", attrs={"ori_shape": (128, 128), "format": "FRACTAL_NZ", "ori_format": "ND"})
        res = add_compute(deq_out, x3, None)
        tensor_list = [x1, x2, bias, deq_scale, res]
        sch = auto_schedule(res)

def test_matmul_fixpipe_0():
    import tbe
    with tbe.common.context.op_context.OpContext("pre-static"):
        with cce():
            x1 = tvm.placeholder((4, 2, 16, 16), name="tensor_a", dtype="float16", attrs={"ori_shape": (32, 64), "format": "FRACTAL_NZ", "ori_format": "ND"})
            x2 = tvm.placeholder((2, 4, 16, 16), name="tensor_b", dtype="float16", attrs={"ori_shape": (64, 32), "format": "FRACTAL_NZ", "ori_format": "ND"})
            bias = tvm.placeholder((32,), name="tensor_bias", dtype="float32", attrs={"format": "ND", "ori_format": "ND", "ori_shape": (32,)})
            output_y = {"shape": (2, 2, 16, 16), "dtype": "float16", "ori_shape": (32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
            matmul_out = mat_mul_compute(x1, x2, bias, None, output_y, False, False, 0)
            y = {"shape": (32, 32), "dtype": "float16", "ori_shape": (32, 32), "format": "ND", "ori_format": "ND"}
            res = fixpipe_compute(matmul_out, None, None, None, None, None, None, None, None, None, y, [], [], "")
            tensor_list = [x1, x2, bias, res]
            sch = auto_schedule(res)

def test_matmul_fixpipe_1():
    import tbe
    with tbe.common.context.op_context.OpContext("pre-static"):
        with cce():
            x1 = tvm.placeholder((4, 2, 16, 16), name="tensor_a", dtype="float16", attrs={"ori_shape": (32, 64), "format": "FRACTAL_NZ", "ori_format": "ND"})
            x2 = tvm.placeholder((2, 4, 16, 16), name="tensor_b", dtype="float16", attrs={"ori_shape": (64, 32), "format": "FRACTAL_NZ", "ori_format": "ND"})
            bias = tvm.placeholder((32,), name="tensor_bias", dtype="float32", attrs={"format": "ND", "ori_format": "ND", "ori_shape": (32,)})
            output_y = {"shape": (4, 2, 16, 8), "dtype": "float32", "ori_shape": (32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
            matmul_out = mat_mul_compute(x1, x2, bias, None, output_y, False, False, 0)
            y = {"shape": (4, 2, 16, 8), "dtype": "float32", "ori_shape": (32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
            res = fixpipe_compute(matmul_out, None, None, None, None, None, None, None, None, None, y, [], [], "")
            tensor_list = [x1, x2, bias, res]
            sch = auto_schedule(res)

def test_matmul_fixpipe_2():
    import tbe
    with tbe.common.context.op_context.OpContext("pre-static"):
        with cce():
            x1 = tvm.placeholder((2, 2, 16, 32), name="tensor_a", dtype="int8", attrs={"ori_shape": (32, 64), "format": "FRACTAL_NZ", "ori_format": "ND"})
            x2 = tvm.placeholder((1, 4, 16, 32), name="tensor_b", dtype="int8", attrs={"ori_shape": (64, 32), "format": "FRACTAL_NZ", "ori_format": "ND"})
            bias = tvm.placeholder((32,), name="tensor_bias", dtype="int32", attrs={"format": "ND", "ori_format": "ND", "ori_shape": (32,)})
            output_y = {"shape": (2, 2, 16, 16), "dtype": "int32", "ori_shape": (32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
            matmul_out = mat_mul_compute(x1, x2, bias, None, output_y, False, False, 0)
            deq = tvm.placeholder((1, 2, 1, 1, 16), name='deq', dtype="uint64", attrs={"ori_shape": (32, ), "format": "NC1HWC0", "ori_format": "ND"})
            y = {"shape": (2, 2, 16, 16), "dtype": "float16", "ori_shape": (32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
            res = fixpipe_compute(matmul_out, None, None, None, None, None, None, None, None, None, y, [], [], "")
            tensor_list = [x1, x2, bias, deq, res]
            sch = auto_schedule(res)
