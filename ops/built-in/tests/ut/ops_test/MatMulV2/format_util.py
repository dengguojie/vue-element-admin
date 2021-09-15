import math


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


def shape_nd_to_Nz(shape, dtype='float16', before_mmad=True):
    """
    >>> shape_nd_to_Nz([3,17])
    [2, 1, 16, 16]
    >>> shape_nd_to_Nz([4,5,3,17])
    [4, 5, 2, 1, 16, 16]
    >>> shape_nd_to_Nz([3,17], dtype='int8')
    [1, 1, 16, 32]
    >>> shape_nd_to_Nz([16,27], dtype='int32')
    [4, 1, 16, 8]
    >>> shape_nd_to_Nz([16,27], dtype='int32', before_mmad=False)
    [2, 1, 16, 16]
    """
    assert (dtype, before_mmad) in (
        ('float16', True),
        ('int8', True),
        ('int32', False), ('int32', True)), f"Please implement shape ND to FRACTAL_NZ with dtype {dtype} on {shape} {'before mmad' if before_mmad else 'after mmad'}"

    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]

    if before_mmad:
        a0, b0 = fractal_shape(dtype)
    else:
        a0, b0 = 16, 16

    return list(batch) + [math.ceil(b/b0), math.ceil(a/a0), a0, b0]


def shape_hwcn_to_fz(shape, dtype='float16'):
    """
    >>> shape_hwcn_to_fz((1,1,1,17))
    [1, 2, 16, 16]
    >>> shape_hwcn_to_fz((1,1,1,17), dtype='int8')
    [1, 2, 16, 32]
    """
    assert dtype in ('float16', 'int8'), f"not support {dtype} yet"
    assert len(shape) == 4
    h, w, c, n = shape
    n0, c0 = fractal_shape(dtype)
    return [math.ceil(h*w*c/c0), math.ceil(n/n0), n0, c0]


def shape_to_nc1hwc0(shape, src_format, dtype='float16'):
    assert src_format in ('nchw', 'hwcn', 'nhwc')
    assert len(shape) == 4
    n, c, h, w = [src_format.index(x) for x in 'nchw']
    _, c0 = fractal_shape(dtype)
    return [n, math.ceil(c/c0), h, w, c0]


def shape_nchw_to_nc1hwc0(shape, dtype='float16'):
    return shape_to_nc1hwc0(shape, 'nchw', dtype)


def shape_hwcn_to_nc1hwc0(shape, dtype='float16'):
    return shape_to_nc1hwc0(shape, 'nchw', dtype)


def shape_nhwc_to_nc1hwc0(shape, dtype='float16'):
    return shape_to_nc1hwc0(shape, 'nchw', dtype)


def change_shape_from_to(shape: list, src_format: str, dst_format: str, dtype: str):
    if src_format == dst_format:
        return shape

    mapping_helper = {('ND', 'FRACTAL_NZ'): shape_nd_to_Nz,
                      ('HWCN', 'FRACTAL_Z'): shape_hwcn_to_fz,
                      ('NCHW', 'NC1HWC0'): shape_nchw_to_nc1hwc0,
                      ('NHWC', 'NC1HWC0'): shape_nhwc_to_nc1hwc0,
                      ('hwcn', 'NC1HWC0'): shape_hwcn_to_nc1hwc0}
    if (src_format, dst_format) not in mapping_helper:
        raise RuntimeError(
            f"Please implement trans func from format {src_format} to {dst_format} on {shape}")

    return mapping_helper[(src_format, dst_format)](shape, dtype)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
