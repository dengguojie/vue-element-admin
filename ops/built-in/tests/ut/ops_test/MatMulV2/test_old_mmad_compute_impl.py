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
    src_dtype = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_dtype)
    result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_fractal_nz_1(test_arg):
    shape_a = [1, 1, 16, 16]
    shape_b = [1, 1, 16, 16]
    src_dtype = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_dtype)
    result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_nd(test_arg):
    shape_a = [16, 16]
    shape_b = [16, 16]
    src_dtype = "float16"
    tensor_a = tvm.placeholder(shape_a, name='tensor_a', attrs={'format': 'ND', 'ori_shape': [16, 16]}, dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b, name='tensor_b', attrs={'format': 'ND', 'ori_shape': [16, 16]}, dtype=src_dtype)
    result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='ND', format_b='ND')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_fractal_nz_trans(test_arg):
    shape_a = [1, 1, 16, 16]
    shape_b = [1, 1, 16, 16]
    src_dtype = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_dtype)
    result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_nd_trans(test_arg):
    shape_a = [16, 16]
    shape_b = [16, 16]
    src_dtype = "float16"
    tensor_a = tvm.placeholder(shape_a, name='tensor_a', attrs={'format': 'ND', 'ori_shape': [16, 16]}, dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b, name='tensor_b', attrs={'format': 'ND', 'ori_shape': [16, 16]}, dtype=src_dtype)
    result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='ND', format_b='ND')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_gemv(test_arg):
    shape_a = [16, 1, 16, 16]
    shape_b = [1, 16, 16, 1]
    src_dtype = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 256]
                               },
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [256, 1]
                               },
                               dtype=src_dtype)
    result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_gevm(test_arg):
    shape_a = [16, 1, 1, 16]
    shape_b = [1, 16, 16, 16]
    src_dtype = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [1, 256]
                               },
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [256, 16]
                               },
                               dtype=src_dtype)
    result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)


def test_old_matmul_fractal_int8_to_fp16(test_arg):
    shape_a = [1, 1, 16, 32]
    shape_b = [1, 1, 16, 32]
    src_dtype = "int8"
    dst_dtype = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 32]
                               },
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_Z',
                                   'ori_shape': [32, 16]
                               },
                               dtype=src_dtype)
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
    src_dtype = "float16"
    tensor_a = tvm.placeholder(shape_a,
                               name='tensor_a',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b,
                               name='tensor_b',
                               attrs={
                                   'format': 'FRACTAL_NZ',
                                   'ori_shape': [16, 16]
                               },
                               dtype=src_dtype)
    matmul_result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
    bias = tvm.const(0, dtype=src_dtype)
    result = te.lang.cce.vadds(matmul_result, bias)
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)

def test_old_matmul_format_a_invalid(test_arg):
    try:
        shape_a = [1, 1, 16, 16, 16]
        shape_b = [1, 1, 16, 16]
        src_dtype = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'NC1HWC0',
                                       'ori_shape': [1, 1, 16, 16]
                                   },
                                   dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_dtype)
        result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='NC1HWC0', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("format_b is in valid!")
    else:
        raise RuntimeError("run case fail")

def test_old_matmul_format_b_invalid(test_arg):
    try:
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 1, 16, 16]
        src_dtype = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'NC1HWC0',
                                       'ori_shape': [1, 1, 16, 16]
                                   },
                                   dtype=src_dtype)
        result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='NC1HWC0')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("format_a is in valid!")
    else:
        raise RuntimeError("run case fail")

def test_old_matmul_fractal_ND_format_invalid(test_arg):
    try:
        shape_a = [1, 1, 16, 16]
        shape_b = [16, 16]
        src_dtype = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [1, 1, 16, 16]
                                   },
                                   dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_dtype)
        result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='ND')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("not support A is fractal and B is ND!")
    else:
        raise RuntimeError("run case fail")

def test_old_matmul_nd_nd_dtype_invalid(test_arg):
    try:
        shape_a = [16, 16]
        shape_b = [16, 16]
        src_dtype = "float32"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_dtype)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='ND', format_b='ND')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("only support 'float16' input datatype for 'ND' and 'ND' format!")
    else:
        raise RuntimeError("run case fail")

def test_old_matmul_nd_fractal_dtype_invalid(test_arg):
    try:
        shape_a = [16, 16]
        shape_b = [1, 1, 16, 16]
        src_a_type = "float32"
        src_b_type = "uint8"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_a_type)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_b_type)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='ND', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("only support float16 & float16 and uint8/int8 input datatype!")
    else:
        raise RuntimeError("run case fail")

def test_old_matmul_trans_a_dtype_invalid(test_arg):
    try:
        shape_a = [16, 16]
        shape_b = [1, 1, 16, 16]
        src_a_type = "int8"
        src_b_type = "int8"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_a_type)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_b_type)
        result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=False, format_a='ND', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("not support A transpose for u8/s8 input and 'ND' and 'fractal'float16 & float16 and uint8/int8 input datatype!")
    else:
        raise RuntimeError("run case fail")

def test_old_matmul_check_shape_len_failed_1(test_arg):
    try:
        shape_a = [1, 1, 1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        src_a_type = "float16"
        src_b_type = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [1, 1, 16, 16]
                                   },
                                   dtype=src_a_type)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_b_type)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("The length of B should be equal to length of A or length of A + 1 or length of A - 1!")
    else:
        raise RuntimeError("run case fail")


def test_old_matmul_check_shape_len_failed_2(test_arg):
    try:
        shape_a = [1, 1, 1, 1, 16, 16]
        shape_b = [1, 1, 1, 16, 16]
        src_a_type = "float16"
        src_b_type = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [1, 1, 16, 16]
                                   },
                                   dtype=src_a_type)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [1, 16, 16]
                                   },
                                   dtype=src_b_type)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("for fractal input data, only support tensor's dim is 4 or 5!")
    else:
        raise RuntimeError("run case fail")

def test_old_matmul_check_shape_len_failed_3(test_arg):
    try:
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        src_a_type = "float16"
        src_b_type = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [1, 1, 16, 16]
                                   },
                                   dtype=src_a_type)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_b_type)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='ND', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("for nd input data, only support tensor's dim is 2 or 3!")
    else:
        raise RuntimeError("run case fail")


def test_old_matmul_check_shape_len_failed_4(test_arg):
    try:
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 1, 1, 16, 16]
        src_a_type = "float16"
        src_b_type = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_a_type)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [1, 1, 16, 16]
                                   },
                                   dtype=src_b_type)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("for fractal input data, only support tensor's dim is 4 or 5!")
    else:
        raise RuntimeError("run case fail")


def test_old_matmul_check_shape_len_failed_5(test_arg):
    try:
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        src_a_type = "float16"
        src_b_type = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_a_type)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [1, 1, 16, 16]
                                   },
                                   dtype=src_b_type)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='FRACTAL_NZ', format_b='ND')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("for nd input data, only support tensor's dim is 2 or 3!")
    else:
        raise RuntimeError("run case fail")

def test_old_matmul_check_shape_len_failed_6(test_arg):
    try:
        shape_a = [16, 16]
        shape_b = [1, 1, 16, 16]
        src_a_type = "float16"
        src_b_type = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_a_type)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [1, 1, 16, 16]
                                   },
                                   dtype=src_b_type)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='FRACTAL_NZ', format_b='ND')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("for nd input data, only support tensor's dim is 2 or 3!")
    else:
        raise RuntimeError("run case fail")


def test_old_matmul_invalid_dst_dtype(test_arg):
    try:
        shape_a = [16, 16]
        shape_b = [16, 16]
        src_dtype = "float16"
        dst_dtype = "float"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'ND',
                                       'ori_shape': [16, 16]
                                   },
                                   dtype=src_dtype)
        result = matmul(tensor_a,
                        tensor_b,
                        trans_a=False,
                        trans_b=False,
                        format_a='ND',
                        format_b='ND',
                        dst_dtype=dst_dtype)
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print(
            "the dst type {} is not in dst dtype check list: ['float16', 'float32', 'int32', 'int8']".format(dst_dtype))
    else:
        raise RuntimeError("run case fail")


def test_old_matmul_fractal_shape_check_failed_1(test_arg):
    try:
        shape_a = [1, 1, 15, 15]
        shape_b = [1, 1, 16, 16]
        src_dtype = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [15, 15]
                                   },
                                   dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [15, 15]
                                   },
                                   dtype=src_dtype)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("the last two dims of input shape of tensor_a is in valid!")
    else:
        raise RuntimeError("run case fail")

def test_old_matmul_fractal_shape_check_failed_2(test_arg):
    try:
        shape_a = [2, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        src_dtype = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                name='tensor_a',
                                attrs={
                                    'format': 'FRACTAL_NZ',
                                    'ori_shape': [16, 32]
                                },
                                dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b,
                                name='tensor_b',
                                attrs={
                                    'format': 'FRACTAL_NZ',
                                    'ori_shape': [16, 16]
                                },
                                dtype=src_dtype)
        result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("reduce axis not same!")
    else:
        raise RuntimeError("run case fail")

def test_old_matmul_fractal_shape_check_failed_3(test_arg):
    try:
        shape_a = [16, 32]
        shape_b = [1, 1, 16, 16]
        src_dtype = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                name='tensor_a',
                                attrs={
                                    'format': 'ND',
                                    'ori_shape': [16, 32]
                                },
                                dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b,
                                name='tensor_b',
                                attrs={
                                    'format': 'FRACTAL_NZ',
                                    'ori_shape': [16, 16]
                                },
                                dtype=src_dtype)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='ND', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("reduce axis not same!")
    else:
        raise RuntimeError("run case fail")


def test_old_matmul_fractal_shape_check_failed_4(test_arg):
    try:
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 15, 15]
        src_dtype = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [15, 15]
                                   },
                                   dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [15, 15]
                                   },
                                   dtype=src_dtype)
        result = matmul(tensor_a, tensor_b, trans_a=False, trans_b=False, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("the last two dims of input shape of tensor_b is in valid!")
    else:
        raise RuntimeError("run case fail")


def test_old_matmul_gemv_invalid(test_arg):
    try:
        shape_a = [16, 1, 1, 16]
        shape_b = [1, 16, 16, 1]
        src_dtype = "float16"
        tensor_a = tvm.placeholder(shape_a,
                                   name='tensor_a',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [1, 256]
                                   },
                                   dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b,
                                   name='tensor_b',
                                   attrs={
                                       'format': 'FRACTAL_NZ',
                                       'ori_shape': [256, 1]
                                   },
                                   dtype=src_dtype)
        result = matmul(tensor_a, tensor_b, trans_a=True, trans_b=True, format_a='FRACTAL_NZ', format_b='FRACTAL_NZ')
        with tvm.target.cce():
            schedule = tbe.auto_schedule(result)
    except RuntimeError as e:
        print("in gemv mode, input shape M and N can't both be 1!")
    else:
        raise RuntimeError("run case fail")



ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_nz)
ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_nz_1)
ut_case.add_cust_test_func(test_func=test_old_matmul_nd)
ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_nz_trans)
ut_case.add_cust_test_func(test_func=test_old_matmul_nd_trans)
ut_case.add_cust_test_func(test_func=test_old_matmul_gemv)
ut_case.add_cust_test_func(test_func=test_old_matmul_gevm)
ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_int8_to_fp16)
ut_case.add_cust_test_func(test_func=test_old_matmul_add_ub_fusion)
ut_case.add_cust_test_func(test_func=test_old_matmul_format_a_invalid)
ut_case.add_cust_test_func(test_func=test_old_matmul_format_b_invalid)
ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_ND_format_invalid)
ut_case.add_cust_test_func(test_func=test_old_matmul_nd_nd_dtype_invalid)
ut_case.add_cust_test_func(test_func=test_old_matmul_nd_fractal_dtype_invalid)
ut_case.add_cust_test_func(test_func=test_old_matmul_trans_a_dtype_invalid)
ut_case.add_cust_test_func(test_func=test_old_matmul_check_shape_len_failed_1)
ut_case.add_cust_test_func(test_func=test_old_matmul_check_shape_len_failed_2)
ut_case.add_cust_test_func(test_func=test_old_matmul_check_shape_len_failed_3)
ut_case.add_cust_test_func(test_func=test_old_matmul_check_shape_len_failed_4)
ut_case.add_cust_test_func(test_func=test_old_matmul_check_shape_len_failed_5)
ut_case.add_cust_test_func(test_func=test_old_matmul_invalid_dst_dtype)
ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_shape_check_failed_1)
ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_shape_check_failed_2)
ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_shape_check_failed_3)
ut_case.add_cust_test_func(test_func=test_old_matmul_fractal_shape_check_failed_4)
ut_case.add_cust_test_func(test_func=test_old_matmul_gemv_invalid)

if __name__ == '__main__':
    ut_case.run()
    exit(0)