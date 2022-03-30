from op_test_frame.ut import OpUT

import te
import tbe.dsl as tbe_dsl
from te import tvm
from tbe.dsl.compute.mmad_compute import matmul
from impl.util.platform_adapter import tbe

ut_case = OpUT("MatMul", None, None)


def test_old_matmul_fractal_nz(test_arg):
    shape_a = [1, 1, 16, 16]
    shape_b = [1, 1, 16, 16]
    src_type = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_type)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_type)
    result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_nd(test_arg):
    shape_a = [16, 16]
    shape_b = [16, 16]
    src_type = "float16"
    tensor_a = tvm.placeholder(shape_a, name='tensor_a', attrs={'format': 'ND', 'ori_shape': [16, 16]}, dtype=src_type)
    tensor_b = tvm.placeholder(shape_b, name='tensor_b', attrs={'format': 'ND', 'ori_shape': [16, 16]}, dtype=src_type)
    result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='ND', format_b='ND')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_fractal_nz_trans(test_arg):
    shape_a = [1, 1, 16, 16]
    shape_b = [1, 1, 16, 16]
    src_type = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_type)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_type)
    result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_nd_trans(test_arg):
    shape_a = [16, 16]
    shape_b = [16, 16]
    src_type = "float16"
    tensor_a = tvm.placeholder(shape_a, name='tensor_a', attrs={'format': 'ND', 'ori_shape': [16, 16]}, dtype=src_type)
    tensor_b = tvm.placeholder(shape_b, name='tensor_b', attrs={'format': 'ND', 'ori_shape': [16, 16]}, dtype=src_type)
    result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='ND', format_b='ND')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_gemv(test_arg):
    shape_a = [16, 1, 16, 16]
    shape_b = [1, 16, 16, 1]
    src_type = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 256]
                               },
                               dtype=src_type)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [256, 1]
                               },
                               dtype=src_type)
    result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_gevm(test_arg):
    shape_a = [16, 1, 1, 16]
    shape_b = [1, 16, 16, 16]
    src_type = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [1, 256]
                               },
                               dtype=src_type)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [256, 16]
                               },
                               dtype=src_type)
    result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_fractal_int8_to_fp16(test_arg):
    shape_a = [1, 1, 16, 32]
    shape_b = [1, 1, 16, 32]
    src_type = "int8"
    dst_dtype = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 32]
                               },
                               dtype=src_type)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_Z',
                                   'ori_shape': [32, 16]
                               },
                               dtype=src_type)
    result = matmul(tensor_a,
                    tensor_b,
                    trans_a=True,
                    trans_b=False,
                    format_a='FRACTAL_NZ',
                    format_b='FRACTAL_NZ',
                    dst_dtype=dst_dtype)
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_add_ub_fusion(test_arg):
    shape_a = [1, 1, 16, 16]
    shape_b = [1, 1, 16, 16]
    src_type = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_type)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_type)
    matmul_result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    bias = tvm.const(0, dtype=src_type)
    result = te.lang.cce.vadds(matmul_result, bias)
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_nz)
ut_case.add_cust_test_func(test_func=test_old_matmul_nd)
ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_nz_trans)
ut_case.add_cust_test_func(test_func=test_old_matmul_nd_trans)
ut_case.add_cust_test_func(test_func=test_old_matmul_gemv)
ut_case.add_cust_test_func(test_func=test_old_matmul_gevm)
ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_int8_to_fp16)
ut_case.add_cust_test_func(test_func=test_old_matmul_add_ub_fusion)

if __name__ == '__main__':
    ut_case.run()
    exit(0)