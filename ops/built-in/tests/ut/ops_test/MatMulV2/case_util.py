from collections import namedtuple
from itertools import product
from inspect import isfunction
import re
import copy

from format_util import change_shape_from_to


Tensor = namedtuple(
    'Tensor', 'ori_shape dtype format param_type ori_format', defaults=('ND',))


def _simple_format(format: str):
    mapping = {'FRACTAL_NZ': 'Nz', 'FRACTAL_Z': 'Zn'}

    if format not in mapping:
        return format

    return mapping[format]


def _simple_dtype(dtype: str):
    for src, dst in (('uint', 'u'), ('float', 'f'), ('int', 's')):
        dtype = dtype.replace(src, dst)

    return dtype


def _gen_case_name(params, prefix=''):
    info = []

    if prefix:
        info.append(prefix)
    for item in params:
        if isinstance(item, dict):
            info.append(_simple_dtype(item['dtype']))
            info.append('_'.join(str(x) for x in item['ori_shape']))
            info.append(_simple_format(item['format']))
        elif isinstance(item, bool):
            info.append('T' if item else 'F')
        elif item is None:
            pass
        else:
            info.append(str(item))
    return '_'.join(info)


def _extract_tensor_to_dict(tensor: Tensor):
    return {'ori_shape': tensor.ori_shape,
            'shape': change_shape_from_to(tensor.ori_shape, tensor.ori_format, tensor.format, tensor.dtype),
            'format': tensor.format,
            'ori_format': tensor.ori_format,
            'dtype': tensor.dtype,
            'param_type': tensor.param_type,  # for precision_case
            }


def extract_case_to_dict(case, case_name=_gen_case_name, calc_expect_func=None, precision_standard=None):
    detail_case = {}

    detail_case['params'] = []
    for item in case:
        if isinstance(item, Tensor):
            detail_case['params'].append(_extract_tensor_to_dict(item))
        else:
            detail_case['params'].append(item)

    if isfunction(case_name):
        detail_case['case_name'] = case_name(detail_case['params'])
    else:
        detail_case['case_name'] = case_name

    # compile
    detail_case['expect'] = 'success'

    # precision
    if calc_expect_func is not None:
        detail_case['calc_expect_func'] = calc_expect_func
    if precision_standard is not None:
        detail_case['precision_standard'] = precision_standard

    return detail_case

def extract_fusion_case_to_dict(case, case_name=_gen_case_name, calc_expect_func=None, precision_standard=None):
    d = extract_case_to_dict(case, case_name=case_name,
                             calc_expect_func=calc_expect_func, precision_standard=precision_standard)
    d["params"] = update_params_with_placeholder(d["params"])
    return d

def update_params_with_placeholder(param):
    from te import tvm

    cnt = 0
    for idx, item in enumerate(param):
        if isinstance(item, dict) and item['param_type'] == 'input':
            attrs = {
                'format': item['format'],
                'ori_format': item['ori_format'],
                'ori_shape': item['ori_shape']
            }

            param[idx] = tvm.placeholder(
                item['shape'], item['dtype'], name=f'param_{cnt}', attrs=attrs)
            cnt += 1

    return param


def apply_choice(base_case, choice):
    if len(choice) == 0:
        return base_case

    def apply(case, indices, values):
        for idx_value, idx in enumerate(indices):
            case[idx] = values[idx_value]
        return case

    values = product(*[x[1] for x in choice])
    indices = [x[0] for x in choice]

    cases = []
    for value in values:
        new_case = copy.deepcopy(base_case)
        cases.append(apply(new_case, indices, value))
    return cases