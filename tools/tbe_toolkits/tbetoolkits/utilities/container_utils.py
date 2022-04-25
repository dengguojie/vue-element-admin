#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Precious Container Related Utilities
"""
# Standard Packages
import math
import logging
from typing import Any
from typing import Tuple
from typing import Optional
from typing import Sequence
from functools import reduce

# Third-Party Packages
import numpy
from .classes import SWITCHES

# Global Storage
global_storage: Optional[SWITCHES] = None


def get(container: Sequence, idx: int, out_of_range=None):
    """
    If a container contains only one element, return that element for whatever the idx is
    :param container: A container
    :param idx: index
    :param out_of_range: Out of Range return
    :return: element
    """
    if len(container) == 1:
        return container[-1]
    else:
        if idx >= len(container) or (idx < 0 and abs(idx) > len(container)):
            if out_of_range:
                return out_of_range
            else:
                logging.warning("Detected out of range access to container %s, "
                                "access index is %d", str(container), idx)
        return container[idx]


def get_global_storage() -> SWITCHES:
    """
    Return global storage structure
    :return:
    """
    return global_storage


def set_global_storage(new_storage: SWITCHES):
    """
    Set global storage structure
    :return:
    """
    global global_storage
    global_storage = new_storage


def shape_product(shape: Tuple[int, ...], initial=1):
    """
    Get shape dimension product
    :param shape: A shape_like container
    :param initial: Initial value if shape is empty
    :return: int
    """
    return reduce(lambda x, y: x * y, shape, initial)


def check_equal_length(*containers: Sequence) -> bool:
    """
    Check if containers all in the same length
    :param containers: multiple containers
    :return: bool
    """
    if len(containers) == 0:
        return True
    all_length = tuple(len(container) == len(containers[0]) for container in containers)
    return all(all_length)


def is_shape(shape: tuple, allowed_type: tuple = (int,)) -> bool:
    """
    Check if input args contains shapes only
    :param shape: a container
    :param allowed_type: a container
    :return: bool
    """
    if not isinstance(shape, Sequence):
        return False
    for dim in shape:
        if not isinstance(dim, allowed_type):
            return False
    return True


def eliminate_scalar_shapes(args: tuple) -> tuple:
    """
    Replace all empty tuple like () with (1,)
    :param args: shape_like containers
    :return:
    """
    if args is None:
        return ()
    return tuple(None if i is None else i if len(i) > 0 else (1,) for i in args)


def sum_len_of_sequences(*args) -> int:
    """
    Get sum of the length of all sequences
    :param args:
    :return:
    """
    result = 0
    for s in args:
        result += len(s)
    return result


def parse_tiling_data(tiling_data: Any) -> Tuple[Optional[bytes], Optional[tuple]]:
    """
    Parse tiling data
    :param tiling_data:
    :return:
    """
    int64_shape_enable = get_global_storage().int64_shape_mode
    # big-endian mode uint32 Tiling data
    be_uint32 = numpy.dtype(numpy.uint32)
    be_uint32 = be_uint32.newbyteorder('<')
    # big-endian mode uint64 Tiling data
    be_uint64 = numpy.dtype(numpy.uint64)
    be_uint64 = be_uint64.newbyteorder('<')
    if isinstance(tiling_data, tuple):
        if int64_shape_enable:
            tiling_data_np_array = numpy.array(tiling_data, dtype=be_uint64)
        else:
            tiling_data_np_array = numpy.array(tiling_data, dtype=be_uint32)
        tiling_data_bytes = tiling_data_np_array.tobytes()
        tuple_tiling_data = tuple(tiling_data_np_array)
    elif isinstance(tiling_data, bytes):
        tiling_data_bytes = tiling_data
        if len(tiling_data) % 4 == 0:
            tiling_data_np_array = numpy.frombuffer(tiling_data, dtype=be_uint32)
            tuple_tiling_data = tuple(tiling_data_np_array)
        else:
            tuple_tiling_data = ("TILING_NOT_4BYTE_ALIGNED",)
    elif tiling_data is None:
        return None, None
    else:
        raise TypeError("Tiling data parsing error, received unknown object: " + str(tiling_data))
    return tiling_data_bytes, tuple_tiling_data


def get_str_tiling_data(dyn_tuple_tiling_data: tuple, dyn_compile_info: dict, dyn_tiling_key: int):
    """
    Get string tiling data
    :param dyn_tuple_tiling_data:
    :param dyn_compile_info:
    :param dyn_tiling_key:
    :return:
    """
    is_tik = False
    if len(dyn_tuple_tiling_data) > 0:
        if "_vars" in dyn_compile_info:
            _vars = {int(k): v for k, v in dyn_compile_info["_vars"].items()}
            if dyn_tiling_key in _vars:
                tiling_indexes = _vars[dyn_tiling_key]
                if len(_vars[dyn_tiling_key]) == \
                        len(dyn_tuple_tiling_data):
                    dict_tiling_data = dict(zip(tiling_indexes, dyn_tuple_tiling_data))
                    dyn_str_tiling_data = str(dict_tiling_data)
                else:
                    dyn_str_tiling_data = f"Tiling data not match with compile_info _vars: {dyn_tuple_tiling_data}"
            elif str(dyn_tiling_key) in _vars:
                tiling_indexes = _vars[str(dyn_tiling_key)]
                if len(_vars[str(dyn_tiling_key)]) == \
                        len(dyn_tuple_tiling_data):
                    dict_tiling_data = dict(zip(tiling_indexes, dyn_tuple_tiling_data))
                    dyn_str_tiling_data = str(dict_tiling_data)
                else:
                    dyn_str_tiling_data = f"Tiling data not match with compile_info _vars: {dyn_tuple_tiling_data}"
            else:
                dyn_str_tiling_data = "Tiling_key %s not found in _vars key %s, treat as Tik operator" \
                                      % (str(dyn_tiling_key),
                                         str(tuple(_vars.keys())))
                is_tik = True
        else:
            dyn_str_tiling_data = f"{dyn_tuple_tiling_data}"
            is_tik = True
    else:
        dyn_str_tiling_data = "Tiling data is empty"
    return dyn_str_tiling_data, is_tik


def bfloat16_conversion(container):
    """
    Convert bfloat16 string to numpy dtype
    :param container:
    :return:
    """
    # noinspection PyUnresolvedReferences
    import tensorflow
    return [dtype if dtype != "bfloat16" else tensorflow.bfloat16.as_numpy_dtype for dtype in container]


def param_transformation(params: dict, signatures: tuple) -> dict:
    """
    Fix params by signatures
    :param params:
    :param signatures:
    :return:
    """
    params = params.copy()
    for param in tuple(params.keys()):
        if param == "axis" and "axes" in signatures and "axes" not in params:
            params["axes"] = params["axis"]
            del params[param]
            continue
        if param == "axes" and "axis" in signatures and "axis" not in params:
            params["axis"] = params["axes"]
            del params[param]
            continue
        if param not in signatures:
            logging.debug("Params transformation removing useless param %s by signatures %s" % (param, signatures))
            del params[param]
            continue
    return params


def tuple_flatten(my_tuple: Sequence) -> tuple:
    """
    Flatten a tuple
    :param my_tuple:
    :return:
    """
    result = []
    for ele in my_tuple:
        if isinstance(ele, (tuple, list, set)):
            for true_ele in tuple_flatten(ele):
                result.append(true_ele)
        else:
            result.append(ele)
    return tuple(result)


def apply_as_list(inputs: Sequence, as_list_distribution: Sequence):
    if as_list_distribution:
        result = []
        last_num = 0
        for num in as_list_distribution:
            if num == 0:
                result.append(inputs[last_num])
                last_num += 1
            else:
                result.append(inputs[last_num:last_num + num])
                last_num += num
        if last_num < len(inputs):
            for tensor in inputs[last_num:]:
                result.append(tensor)
    else:
        result = inputs
    return result


def table_print(data: Sequence[Tuple[str, ...]]):
    result = ""
    cells_width = [0]
    sub_row_size = []
    minimum_table_width = 0
    # Determine Table Width
    for row in data:
        sub_row_size.append(0)
        if len(row) == 1:
            sub_rows = row[0].split("\n")
            for sub_row in sub_rows:
                if minimum_table_width < len(sub_row) + 2:
                    minimum_table_width = len(sub_row) + 2
            sub_row_size[-1] = len(sub_rows)
        else:
            for idx, column in enumerate(row):
                if len(cells_width) <= idx:
                    cells_width.append(0)
                sub_rows = column.split("\n")
                for sub_row in sub_rows:
                    if cells_width[idx] < len(sub_row) + 2:
                        cells_width[idx] = len(sub_row) + 2
                if sub_row_size[-1] < len(sub_rows):
                    sub_row_size[-1] = len(sub_rows)

    # Check for Minimum Table Width Requirements
    if sum(cells_width) < minimum_table_width:
        for idx, cell_width in enumerate(cells_width):
            if cell_width < minimum_table_width // len(cells_width):
                cells_width[idx] = math.ceil(minimum_table_width / len(cells_width))
    char_length = sum(cells_width) + len(cells_width) * 2 - 1
    for idx, row in enumerate(data):
        length = len(row)
        result += '+' + char_length * '-' + '+'
        result += '\n'
        sub_result = []
        for sub_row_idx in range(sub_row_size[idx]):
            if length == 1:
                sub_result.append('| ' + str(row[0]).split('\n')[sub_row_idx].ljust(
                    sum(cells_width) + 2 * len(cells_width) - 2) + '|')
            else:
                sub_row = [str(column).split('\n')[sub_row_idx] if sub_row_idx < len(str(column).split('\n')) else ""
                           for column in row]
                sub_result.append('|' + '|'.join(' ' + sub_row[i].ljust(cells_width[i])
                                                 for i in range(length)) + '|')
        result += '\n'.join(sub_result)
        result += '\n'
    result += '+' + char_length * '-' + '+'
    return result
