from case_util import Tensor, extract_case_to_dict
from op_test_frame.common import precision_info


def golden_matmul_v2(input_x1,
                     input_x2,
                     bias,
                     offset_w={},
                     output_y={},
                     trans_a=False,
                     trans_b=False,
                     offset_x=0,
                     kernel_name="matmul"):
    import numpy as np
    import math

    a = np.array(input_x1.get("value"))
    b = np.array(input_x2.get("value"))
    assert a.dtype == 'float16' and b.dtype == 'float16', f"only support float16 now"

    if input_x1.get('format') == 'FRACTAL_NZ':
        if trans_a:
            # m1, k1, k0, m0 -> m1, m0, k1, k0
            a = np.transpose(a, [x for x in range(
                len(a.shape) - 4)] + [-4, -1, -3, -2])
        else:
            # k1, m1, m0, k0 -> m1, m0, k1, k0
            a = np.transpose(a, [x for x in range(
                len(a.shape) - 4)] + [-3, -2, -4, -1])
        *_, m1, m0, k1, k0 = a.shape
        a = a.reshape(list(a.shape[:len(a.shape)-4]) +
                      [m1*m0, k1*k0]).astype('float32')
    elif input_x1.get('format') == 'ND':
        if trans_a:
            a = np.transpose(a, [1, 0])
        else:
            pass
    else:
        raise RuntimeError(f'not support format of a is {input_x1["format"]}')

    if input_x2.get('format') == 'FRACTAL_NZ':
        if trans_b:
            # k1, n1, n0, k0 -> k1, k0, n1, n0
            b = np.transpose(b, [x for x in range(
                len(b.shape) - 4)] + [-4, -1, -3, -2])
        else:
            # n1, k1, k0, n0 -> k1, k0, n1, n0
            b = np.transpose(b, [x for x in range(
                len(b.shape) - 4)] + [-3, -2, -4, -1])
        *_, k1, k0, n1, n0 = b.shape
        b = b.reshape(list(b.shape[:len(b.shape)-4]) +
                      [k1*k0, n1*n0]).astype('float32')
        b = b[[slice(None, x) for x in input_x2['ori_shape']]]
    elif input_x2.get('format') == 'ND':
        if trans_b:
            b = np.transpose(b, [1, 0])
        else:
            pass
    else:
        raise RuntimeError(f'not support format of b is {input_x2["format"]}')

    res = np.matmul(a, b)

    if bias is not None:
        np_bias = np.array(bias.get("value"))
        np_bias = np.pad(np_bias, [(0, 0)]*(len(np_bias.shape)-1) +
                         [(0, math.ceil(np_bias.shape[-1] / 16)*16-np_bias.shape[-1])])
        if np_bias.dtype == 'float16':
            np_bias = np_bias.astype('float32')
        res += np_bias

    if output_y['format'] == 'FRACTAL_NZ':
        # b..., m, n -> b..., m1, m0, n1, n0 -> b..., n1, m1, m0, n0
        res = res.reshape(list(res.shape[:len(res.shape)-2]) +
                          [res.shape[-2] // 16, 16, res.shape[-1] // 16, 16])
        res = np.transpose(res, [x for x in range(
            len(res.shape) - 4)] + [-2, -4, -3, -1]).astype('float16')
    elif output_y['format'] == 'ND':
        pass
    else:
        raise RuntimeError(
            f'not support format of output is {output_y["format"]}')

    return res


pass_cases = [
    [Tensor((32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((32, 64), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     Tensor((64,), 'float16', 'ND', 'input'),
     None,
     Tensor((32, 64), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((32, 64), 'float32', 'ND', 'output'),
     False, False, 0],
    [Tensor((32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     Tensor((64,), 'float16', 'ND', 'input'),
     None,
     Tensor((32, 64), 'float32', 'ND', 'output'),
     False, False, 0],
    [Tensor((128, 32), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((32, 64), 'float16', 'ND', 'output'),
     True, False, 0],
    [Tensor((128, 32), 'float16', 'ND', 'input'),
     Tensor((64, 128), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((32, 64), 'float16', 'ND', 'output'),
     True, True, 0],
    [Tensor((32, 128), 'float16', 'ND', 'input'),
     Tensor((64, 128), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((32, 64), 'float16', 'ND', 'output'),
     False, True, 0],
    [Tensor((32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'FRACTAL_NZ', 'input'),
     None,
     None,
     Tensor((32, 64), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'FRACTAL_NZ', 'input'),
     None,
     None,
     Tensor((32, 64), 'float32', 'ND', 'output'),
     False, False, 0],
    [Tensor((1, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'FRACTAL_NZ', 'input'),
     None,
     None,
     Tensor((1, 64), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((1, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((1, 64), 'float16', 'ND', 'output'),
     False, False, 0],
]

cases = [
]

case_low_priority = [
    # CHECK: GEMV shape of bias must be m*n
    [Tensor((32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 1), 'float16', 'ND', 'input'),
     Tensor((1,), 'float16', 'ND', 'input'),
     None,
     Tensor((32, 1), 'float16', 'ND', 'output'),
     False, False, 0],
]

# compare failed
cases = [
    [Tensor((32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 1), 'float16', 'FRACTAL_NZ', 'input'),
     None,
     None,
     Tensor((32, 1), 'float32', 'ND', 'output'),
     False, False, 0],
]

vit_cases = [
    [Tensor((1576, 3072), 'float16', 'ND', 'input'),  # 1
     Tensor((1576, 768), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3072, 768), 'float16', 'ND', 'output'),
     True, False, 0],
    [Tensor((1576, 3072), 'float16', 'ND', 'input'),  # 2
     Tensor((3072, 768), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((1576, 768), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((1576, 3072), 'float16', 'ND', 'input'),  # 3
     Tensor((768, 3072), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((1576, 768), 'float16', 'ND', 'output'),
     False, True, 0],
    [Tensor((1576, 768), 'float16', 'ND', 'input'),  # 4
     Tensor((1576, 3072), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((768, 3072), 'float16', 'ND', 'output'),
     True, False, 0],
    [Tensor((1576, 768), 'float16', 'ND', 'input'),  # 5
     Tensor((1576, 768), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((768, 768), 'float16', 'ND', 'output'),
     True, False, 0],
    [Tensor((1576, 768), 'float16', 'ND', 'input'),  # 6
     Tensor((3072, 768), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((1576, 3072), 'float16', 'ND', 'output'),
     False, True, 0],
    [Tensor((1576, 768), 'float16', 'ND', 'input'),  # 7
     Tensor((768, 3072), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((1576, 3072), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((1576, 768), 'float16', 'ND', 'input'),  # 8
     Tensor((768, 768), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((1576, 768), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((1576, 768), 'float16', 'ND', 'input'),  # 9
     Tensor((768, 768), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((1576, 768), 'float16', 'ND', 'output'),
     False, True, 0],
    [Tensor((8, 10), 'float16', 'ND', 'input'),  # 10
     Tensor((768, 10), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((8, 768), 'float16', 'ND', 'output'),
     False, True, 0],
    [Tensor((8, 768), 'float16', 'ND', 'input'),  # 11
     Tensor((768, 10), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((8, 10), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((8, 768), 'float16', 'ND', 'input'),  # 12
     Tensor((8, 10), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((768, 10), 'float16', 'ND', 'output'),
     True, False, 0],
]

bert_cases = [
    [Tensor((304, 1024), 'float16', 'ND', 'input'),
     Tensor((30522, 1024), 'float16', 'ND', 'input'),
     Tensor((30522,), 'float16', 'ND', 'input'),
     None,
     Tensor((304, 30522), 'float16', 'ND', 'output'),
     False, True, 0],
]

cases = bert_cases
cases = [extract_case_to_dict(x, calc_expect_func=golden_matmul_v2,
                              precision_standard=precision_info.PrecisionStandard(0.005, 0.005), ) for x in cases]
