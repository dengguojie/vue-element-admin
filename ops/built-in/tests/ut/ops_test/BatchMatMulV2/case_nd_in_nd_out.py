from case_util import Tensor, extract_case_to_dict
from op_test_frame.common import precision_info


def golden_batch_matmul_v2(input_x,
                           input_y,
                           bias,
                           offset_w={},
                           output_z={},
                           trans_a=False,
                           trans_b=False,
                           offset_x=0,
                           kernel_name="matmul"):
    import numpy as np
    import math

    a = np.array(input_x.get("value"))
    b = np.array(input_y.get("value"))
    assert a.dtype == 'float16' and b.dtype == 'float16', f"only support float16 now"

    if input_x.get('format') == 'FRACTAL_NZ':
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
    elif input_x.get('format') == 'ND':
        if trans_a:
            a = np.transpose(
                a, [x for x in range(len(a.shape) - 2)] + [-1, -2])
        else:
            pass
    else:
        raise RuntimeError(f'not support format of a is {input_x["format"]}')

    if input_y.get('format') == 'FRACTAL_NZ':
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
    elif input_y.get('format') == 'ND':
        if trans_b:
            b = np.transpose(
                b, [x for x in range(len(b.shape) - 2)] + [-1, -2])
        else:
            pass
    else:
        raise RuntimeError(f'not support format of b is {input_y["format"]}')

    res = np.matmul(a, b)

    if bias is not None:
        np_bias = np.array(bias.get("value"))
        np_bias = np.pad(np_bias, [(0, 0)]*(len(np_bias.shape)-1) +
                         [(0, math.ceil(np_bias.shape[-1] / 16)*16-np_bias.shape[-1])])
        if np_bias.dtype == 'float16':
            np_bias = np_bias.astype('float32')
        res += np_bias

    if output_z['format'] == 'FRACTAL_NZ':
        # b..., m, n -> b..., m1, m0, n1, n0 -> b..., n1, m1, m0, n0
        res = res.reshape(list(res.shape[:len(res.shape)-2]) +
                          [res.shape[-2] // 16, 16, res.shape[-1] // 16, 16])
        res = np.transpose(res, [x for x in range(
            len(res.shape) - 4)] + [-2, -4, -3, -1]).astype('float16')
    elif output_z['format'] == 'ND':
        if list(output_z['shape']) != list(res.shape):
            # work for in: ND, FRACTAL_NZ out: ND
            res = res[[slice(None, x) for x in output_z['shape']]]
    else:
        raise RuntimeError(
            f'not support format of output is {output_z["format"]}')

    return res


pass_cases = [
    [Tensor((3, 2, 32, 128), 'float16', 'ND', 'input'),
     Tensor((3, 2, 128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((3, 2, 32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((3, 32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 32, 64), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((3, 2, 32, 128), 'float16', 'ND', 'input'),
     Tensor((3, 2, 128, 64), 'float16', 'ND', 'input'),
     Tensor((64,), 'float16', 'ND', 'input'),
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((3, 2, 32, 128), 'float16', 'ND', 'input'),
     Tensor((3, 2, 128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 2, 32, 64), 'float32', 'ND', 'output'),
     False, False, 0],
    [Tensor((3, 2, 32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 2, 32, 64), 'float32', 'ND', 'output'),
     False, False, 0],
    [Tensor((3, 32, 128), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 32, 64), 'float32', 'ND', 'output'),
     False, False, 0],
    [Tensor((3, 2, 32, 128), 'float16', 'ND', 'input'),
     Tensor((3, 2, 128, 64), 'float16', 'ND', 'input'),
     Tensor((64,), 'float16', 'ND', 'input'),
     None,
     Tensor((3, 2, 32, 64), 'float32', 'ND', 'output'),
     False, False, 0],
    [Tensor((3, 2, 32, 128), 'float16', 'ND', 'input'),
     Tensor((3, 2, 64, 128), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     False, True, 0],
    [Tensor((3, 2, 32, 128), 'float16', 'ND', 'input'),
     Tensor((64, 128), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     False, True, 0],
    [Tensor((3, 32, 128), 'float16', 'ND', 'input'),
     Tensor((64, 128), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 32, 64), 'float16', 'ND', 'output'),
     False, True, 0],
    [Tensor((3, 2, 32, 128), 'float16', 'ND', 'input'),
     Tensor((3, 2, 64, 128), 'float16', 'ND', 'input'),
     Tensor((64,), 'float16', 'ND', 'input'),
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     False, True, 0],
    [Tensor((3, 2, 128, 32), 'float16', 'ND', 'input'),
     Tensor((3, 2, 64, 128), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     True, True, 0],
    [Tensor((3, 2, 128, 32), 'float16', 'ND', 'input'),
     Tensor((64, 128), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     True, True, 0],
    [Tensor((3, 128, 32), 'float16', 'ND', 'input'),
     Tensor((64, 128), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 32, 64), 'float16', 'ND', 'output'),
     True, True, 0],
    [Tensor((3, 2, 128, 32), 'float16', 'ND', 'input'),
     Tensor((3, 2, 64, 128), 'float16', 'ND', 'input'),
     Tensor((64,), 'float16', 'ND', 'input'),
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     True, True, 0],
    [Tensor((3, 2, 128, 32), 'float16', 'ND', 'input'),
     Tensor((3, 2, 128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     True, False, 0],
    [Tensor((3, 2, 128, 32), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     True, False, 0],
    [Tensor((3, 128, 32), 'float16', 'ND', 'input'),
     Tensor((128, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((3, 32, 64), 'float16', 'ND', 'output'),
     True, False, 0],
    [Tensor((3, 2, 128, 32), 'float16', 'ND', 'input'),
     Tensor((3, 2, 128, 64), 'float16', 'ND', 'input'),
     Tensor((64,), 'float16', 'ND', 'input'),
     None,
     Tensor((3, 2, 32, 64), 'float16', 'ND', 'output'),
     True, False, 0],
]

cases = [
    # accuracy is not up to standard
    # [Tensor((12, 2, 32, 128), 'float16', 'ND', 'input'),
    #  Tensor((2, 128, 64), 'float16', 'ND', 'input'),
    #  None,
    #  None,
    #  Tensor((12, 2, 32, 64), 'float16', 'ND', 'output'),
    #  False, False, 0],
]

vit_cases = [
    [Tensor((8, 12, 197, 197), 'float16', 'ND', 'input'),
     Tensor((8, 12, 197, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((8, 12, 197, 64), 'float16', 'ND', 'output'),
     False, False, 0],
    [Tensor((8, 12, 197, 197), 'float16', 'ND', 'input'),
     Tensor((8, 12, 197, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((8, 12, 197, 64), 'float16', 'ND', 'output'),
     True, False, 0],
    [Tensor((8, 12, 197, 64), 'float16', 'ND', 'input'),
     Tensor((8, 12, 197, 64), 'float16', 'ND', 'input'),
     None,
     None,
     Tensor((8, 12, 197, 197), 'float16', 'ND', 'output'),
     False, True, 0],
]
# cases = pass_cases
cases = vit_cases

cases = [extract_case_to_dict(x, calc_expect_func=golden_batch_matmul_v2,
                              precision_standard=precision_info.PrecisionStandard(0.005, 0.005), ) for x in cases]
