from op_test_frame.common import precision_info
from collections import namedtuple
import math
import re

Tensor = namedtuple('Tensor', 'ori_shape dtype format param_type ori_format', defaults=('ND',))

def _golden_batchmatmul_v2(input_x, input_y, bias=None, offset_w={}, output_z={}, trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul"):
    import numpy as np
    import math

    a = np.array(input_x.get("value"))
    b = np.array(input_y.get("value"))
    assert len(a.shape) >= 4 and len(b.shape) >= 4, f"only support Nz now"
    assert a.dtype == 'float16' and b.dtype == 'float16', f"only support float16 now"

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

    res = np.matmul(a, b)

    if bias is not None:
        np_bias = np.array(bias.get("value"))
        np_bias = np.pad(np_bias, [(0, 0)]*(len(np_bias.shape)-1) +
                         [(0, math.ceil(np_bias.shape[-1] / 16)*16-np_bias.shape[-1])])
        res += np_bias.astype('float32')

    # b..., m, n -> b..., m1, m0, n1, n0 -> b..., n1, m1, m0, n0
    res = res.reshape(list(res.shape[:len(res.shape)-2]) +
                      [res.shape[-2] // 16, 16, res.shape[-1] // 16, 16])
    res = np.transpose(res, [x for x in range(
        len(res.shape) - 4)] + [-2, -4, -3, -1]).astype('float16')
    return res

def simply_format(format: str):
    mapping = {'FRACTAL_NZ': 'Nz'}

    if format not in mapping:
        return format

    return mapping[format]

def simply_dtype(dtype: str):
    mapping = {'float16': 'f16', 'float32': 'f32'}

    if dtype not in mapping:
        return dtype

    return mapping[dtype]

def _gen_case_name(params):
    info = []
    for item in params:
        if isinstance(item, dict):
            info.append('_'.join(str(x) for x in item['ori_shape']))
            info.append(simply_format(item['format']))
            info.append(simply_dtype(item['dtype']))
        else:
            info.append(str(item))
    return '_'.join(info)

def fractal_shape(dtype):
    """
    >>> fractal_shape('int8')
    (16, 32)
    >>> fractal_shape('float16')
    (16, 16)
    >>> fractal_shape('float32')
    (16, 8)
    """
    import re
    res = re.match("[^\d]+(\d+)", dtype)
    assert res is not None
    bit_of_dtype = int(res[1])
    assert (32 * 8) % bit_of_dtype == 0

    return 16, (32 * 8) // bit_of_dtype

def shape_nd_to_Nz(shape, dtype='float16'):
    """
    >>> shape_nd_to_Nz([3,17])
    [2, 1, 16, 16]
    >>> shape_nd_to_Nz([4,5,3,17])
    [4, 5, 2, 1, 16, 16]
    """
    assert dtype == 'float16', f"not support {dtype} yet"
    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]
    a0, b0 = fractal_shape(dtype)
    return list(batch) + [math.ceil(b/b0), math.ceil(a/a0), a0, b0]

def _shape_from_to(shape: list, src_format: str, dst_format: str, dtype: str):
    if src_format == dst_format:
        return shape

    mapping_helper = {('ND', 'FRACTAL_NZ'): shape_nd_to_Nz}
    if (src_format, dst_format) not in mapping_helper:
        raise RuntimeError(f"Please implement trans func from format {src_format} to {dst_format}")
    
    return mapping_helper[(src_format, dst_format)](shape, dtype)


def _extract_tensor(tensor: Tensor):
    return {'ori_shape': tensor.ori_shape,
            'shape': _shape_from_to(tensor.ori_shape, tensor.ori_format, tensor.format, tensor.dtype),
            'format': tensor.format,
            'ori_format': tensor.ori_format,
            'dtype': tensor.dtype,
            'param_type': tensor.param_type, # for precision_case
            }


def _extract_case(case):
    detail_case = {}

    detail_case['params'] = []
    for item in case:
        if isinstance(item, Tensor):
            detail_case['params'].append(_extract_tensor(item))
        else:
            detail_case['params'].append(item)

    # compile
    detail_case['expect'] = 'success'

    # precision
    detail_case['calc_expect_func'] = _golden_batchmatmul_v2
    detail_case['precision_standard'] = precision_info.PrecisionStandard(0.005, 0.005)

    detail_case['case_name'] = _gen_case_name(detail_case['params'])

    return detail_case

# operator only supports bias as [n], shielding all the previous ones, provide schedule is [n]
# schedule only supports [n], [1,n], [1,1,n]
cases = [
    [Tensor((2, 4, 32, 64), 'float16', 'FRACTAL_NZ', 'input'),
     Tensor((4, 64, 16), 'float16', 'FRACTAL_NZ', 'input'),
     Tensor((15,), 'float16', 'ND', 'input'),
     None,
     Tensor((2, 4, 32, 16), 'float16', 'FRACTAL_NZ', 'output'), False, False],
    [Tensor((2, 4, 32, 64), 'float16', 'FRACTAL_NZ', 'input'),
     Tensor((4, 64, 16), 'float16', 'FRACTAL_NZ', 'input'),
     Tensor((1, 15,), 'float16', 'ND', 'input'),
     None,
     Tensor((2, 4, 32, 16), 'float16', 'FRACTAL_NZ', 'output'), False, False],
    [Tensor((2, 4, 32, 64), 'float16', 'FRACTAL_NZ', 'input'),
     Tensor((4, 64, 16), 'float16', 'FRACTAL_NZ', 'input'),
     Tensor((1, 1, 1, 15,), 'float16', 'ND', 'input'),
     None,
     Tensor((2, 4, 32, 16), 'float16', 'FRACTAL_NZ', 'output'), False, False],
    [Tensor((2, 4, 32, 64), 'float16', 'FRACTAL_NZ', 'input'),
     Tensor((4, 64, 16), 'float16', 'FRACTAL_NZ', 'input'),
     Tensor((1, 1, 15,), 'float16', 'ND', 'input'),
     None,
     Tensor((2, 4, 32, 16), 'float16', 'FRACTAL_NZ', 'output'), False, False],
    # [Tensor((2, 4, 32, 64), 'float16', 'FRACTAL_NZ', 'input'),
    #  Tensor((4, 64, 16), 'float16', 'FRACTAL_NZ', 'input'),
    #  Tensor((15,), 'float32', 'ND', 'input'), # not support float32 in single op
    #  None,
    #  Tensor((2, 4, 32, 16), 'float16', 'FRACTAL_NZ', 'output'), False, False],
    # [Tensor((2, 4, 32, 64), 'float16', 'FRACTAL_NZ', 'input'),
    #  Tensor((4, 64, 16), 'float16', 'FRACTAL_NZ', 'input'),
    #  Tensor((1, 15,), 'float32', 'ND', 'input'), # not support float32 in single op
    #  None,
    #  Tensor((2, 4, 32, 16), 'float16', 'FRACTAL_NZ', 'output'), False, False],
    # [Tensor((2, 4, 32, 64), 'float16', 'FRACTAL_NZ', 'input'),
    #  Tensor((4, 64, 16), 'float16', 'FRACTAL_NZ', 'input'),
    #  Tensor((1, 1, 1, 15,), 'float32', 'ND', 'input'), # not support float32 in single op
    #  None,
    #  Tensor((2, 4, 32, 16), 'float16', 'FRACTAL_NZ', 'output'), False, False],
    # [Tensor((2, 4, 32, 64), 'float16', 'FRACTAL_NZ', 'input'),
    #  Tensor((4, 64, 16), 'float16', 'FRACTAL_NZ', 'input'),
    #  Tensor((1, 1, 15,), 'float32', 'ND', 'input'), # not support float32 in single op
    #  None,
    #  Tensor((2, 4, 32, 16), 'float16', 'FRACTAL_NZ', 'output'), False, False],
    ]


base_cases = [_extract_case(x) for x in cases]

compile_cases = base_cases
precision_cases = base_cases