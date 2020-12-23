import numpy as np

import te.lang.cce as tbe
from te import tvm


def _get_shape_dims(shape_a, shape_b, trans_a, trans_b):
    if trans_a:
        k, m, block_m, block_k = shape_a
    else:
        m, k, block_k, block_m = shape_a
    if trans_b:
        n, k, block_k, block_n = shape_b
    else:
        k, n, block_n, block_k = shape_b
    return m, block_m, k, block_k, n, block_n


def _calc_by_cpu(
        input_a,
        input_b,
        bias,
        output,
        data_a,
        data_b,
        data_bias,
        trans_a=False,
        trans_b=False,
):
    shape_a = input_a.get("shape")
    shape_b = input_b.get("shape")
    shape_bias = bias.get("shape")
    shape_dst = output.get("shape")
    src_dtype = input_a.get("dtype")
    dst_dtype = output.get("dtype")
    format_a = input_a.get("format")
    format_b = input_b.get("format")
    if format_a != "ND":
        trans_a = not trans_a
        trans_b = not trans_b
    tensor_a = tvm.placeholder(shape_a, name='tensor_a',
                               attrs={'format': format_a},
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b, name='tensor_b',
                               attrs={'format': format_b},
                               dtype=src_dtype)
    tensor_bias = None
    shape_bias_length = len(shape_bias)
    if shape_bias_length > 0:
        tensor_bias = tvm.placeholder(
            shape_bias, name='tensor_bias',
            dtype=dst_dtype)
    result = tbe.matmul(tensor_a=tensor_a, tensor_b=tensor_b,
                        trans_a=trans_a, trans_b=trans_b,
                        format_a=format_a, format_b=format_b,
                        dst_dtype=dst_dtype, tensor_bias=tensor_bias)

    schedule = tvm.create_schedule(result.op)
    if tensor_bias is None:
        f_matmul = tvm.build(schedule, [tensor_a, tensor_b, result], "c", "llvm", name="f_matmul")
    else:
        f_matmul = tvm.build(schedule, [tensor_a, tensor_b, tensor_bias, result], "c", "llvm", name="f_matmul")
    ctx = tvm.cpu(0)
    res = tvm.nd.array(np.zeros(shape_dst, dtype=dst_dtype).astype(src_dtype), ctx)
    if format_a != "ND":
        m, block_m, k, block_k, n, block_n = _get_shape_dims(shape_a, shape_b, trans_a, trans_b)
        data_a = data_a.reshape([m, block_m, k, block_k]).transpose([2, 0, 1, 3])
        data_b = data_b.reshape([k, block_k, n, block_n]).transpose([2, 0, 1, 3])
    ctx = tvm.cpu(0)
    data_a = tvm.nd.array(data_a, ctx)
    data_b = tvm.nd.array(data_b, ctx)
    if tensor_bias is None:
        f_matmul(data_a, data_b, res)
    else:
        data_bias = tvm.nd.array(data_bias, ctx)
        f_matmul(data_a, data_b, data_bias, res)
    res = res.asnumpy()
    if format_a != "ND":
        m, block_m, _, _, n, block_n = _get_shape_dims(shape_a, shape_b, trans_a, trans_b)
        res = res.transpose([1, 2, 0, 3]).reshape([m * block_m, n * block_n])
    return res


def _calc_valid(shape_a, shape_bias, data_a, data_b, data_bias, trans_a, trans_b):
    if trans_a:
        shape_a = list(reversed(shape_a))
        data_a = data_a.transpose()
    if trans_b:
        data_b = data_b.transpose()
    res = np.matmul(data_a, data_b)
    if len(shape_bias) > 0:
        data_bias = np.tile(data_bias, (shape_a[0], 1))
        res = res + data_bias
    return res


def matmul_cpu_validation(test_case):
    ori_shape_a = test_case[0]["ori_shape"]
    src_dtype = test_case[0]["dtype"]
    ori_shape_b = test_case[1]["ori_shape"]
    shape_bias = test_case[2]["shape"]
    dst_dtype = test_case[4]["dtype"]
    trans_a = test_case[5]
    trans_b = test_case[6]
    data_a = np.random.uniform(size=ori_shape_a).astype(src_dtype)
    data_b = np.random.uniform(size=ori_shape_b).astype(src_dtype)
    data_bias = np.array([]).astype(dst_dtype)
    shape_bias_length = len(shape_bias)
    if shape_bias_length > 0:
        data_bias = np.random.uniform(size=shape_bias).astype(dst_dtype)
    cpu_res = _calc_by_cpu(
        test_case[0], test_case[1], test_case[2], test_case[4], data_a,
        data_b, data_bias, trans_a, trans_b)
    valid_res = _calc_valid(ori_shape_a, shape_bias, data_a, data_b, data_bias, trans_a, trans_b)
    tvm.testing.assert_allclose(valid_res, cpu_res, 1e-3, 1e-3)


if __name__ == "__main__":
    case = {
        "params": [
            {"shape": (6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 96), "ori_format": "ND"},
            {"shape": (4, 6, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 64), "ori_format": "ND"},
            {"shape": (64, ), "dtype": "float16", "format": "ND", "ori_shape": (64, ), "ori_format": "ND"},
            None,
            {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64), "ori_format": "ND"},
            False,
            False
        ],
        "case_name": "MatMul_1",
        "expect": "success",
        "format_expect": [],
        "support_expect": True
    }
    matmul_cpu_validation(case["params"])
