#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Parser for each csv field
"""
# Standard Packages
import time
import logging
from typing import Union

# Third-Party Packages
from ...utilities import is_shape


def _is_inference(value: str) -> bool:
    # Check if inference is needed
    if value in ["ELEWISE", "REDUCE"]:
        return True
    # Check if limited inference is needed
    if "ELEWISE" in value and len(value) > 7:
        if isinstance(eval(value[7:]), (tuple, list)):
            return True
        else:
            raise ValueError("%s is not a valid limited inference value!" % value)
    if "REDUCE" in value and len(value) > 6:
        if isinstance(eval(value[6:]), (tuple, list)):
            return True
        else:
            raise ValueError("%s is not a valid limited inference value!" % value)
    return False


def process_bool(value: str) -> bool:
    value = value.upper()
    if value == "TRUE":
        value = "True"
    elif value == "FALSE":
        value = "False"
    try:
        parsed = eval(value)
    except:
        raise TypeError("%s is not a valid boolean value!" % value)
    if parsed:
        return True
    return False


def process_string(value) -> str:
    # noinspection PyBroadException
    try:
        result = eval(value)
    except:
        return value
    else:
        if isinstance(result, type(None)):
            return result
        else:
            return value


def _shapelike(value: str,
               allow_inference=False, positive_only=False,
               allow_float=False, allow_sub_value_none=False, allow_none=False) -> Union[tuple, str, type(None)]:
    if allow_inference:
        if _is_inference(value):
            return value
    parsed = eval(value)
    # No inference needed, treat as normal
    if parsed is None and allow_none:
        return None
    if not isinstance(parsed, (tuple, list)):
        raise TypeError("%s is not a valid shapelike value!" % value)
    # Make sure all shapelike container is a tuple
    parsed = tuple(parsed)
    # Check if empty
    if len(parsed) == 0:
        logging.warning("Detected empty shapelike value, be careful that such representation will be interpreted to "
                        "(1,)")
        return (1,),
    # Iterate through all sub_values, set allowed_type
    allowed_type = (int,)
    if allow_float:
        allowed_type += (float,)
    if allow_sub_value_none:
        allowed_type += (type(None),)
    if all(isinstance(sub_value, allowed_type) for sub_value in parsed):
        # sub_value is single value, convert to tuple and return
        if positive_only and not all((i > 0 for i in parsed if i is not None)):
            logging.warning("shapelike value should not have non-positive dim %s" % value)
        return parsed,
    for idx, sub_value in enumerate(parsed):
        if isinstance(sub_value, (tuple, list)):
            if positive_only:
                if not is_shape(sub_value, allowed_type):
                    raise TypeError("%s is not a valid shape in type %s" % (str(sub_value), str(allowed_type)))
                if not all((i > 0 for i in sub_value)):
                    logging.warning("shapelike value should not have non-positive dim %s" % value)
                if not all((isinstance(i, allowed_type) for i in sub_value)):
                    raise ValueError("shapelike value should not have invalid dim %s" % value)
            else:
                if not is_shape(sub_value, allowed_type):
                    raise TypeError("%s is not a valid shape in type %s" % (str(sub_value), str(allowed_type)))
                if not all((isinstance(i, allowed_type) for i in sub_value)):
                    raise ValueError("shapelike value should not have invalid dim %s" % value)
        elif isinstance(sub_value, type(None)):
            pass
        else:
            raise TypeError("%s of %s is not a valid shapelike value for its corresponding field" % (str(sub_value),
                                                                                                     value))
    new_parsed = tuple(tuple(element) if element is not None else None for element in parsed)
    return new_parsed


def process_dynamic_shapelike(value: str) -> tuple:
    """
    For shapelike (1, 7, -1)
    :param value:
    :return:
    """
    return _shapelike(value, allow_none=True)


def process_dynamic_inferable_shapelike(value: str):
    """
    For shapelike REDUCE or (1, 7, -1)
    :param value:
    :return:
    """
    return _shapelike(value, allow_inference=True, allow_sub_value_none=True, allow_none=True)


def shapelike_stc(value: str):
    """
    For shapelike (34, 16, 16)
    :param value:
    :return:
    """
    return _shapelike(value, positive_only=True)


def shapelike_stc_ex(value: str):
    """
    For shapelike ELEWISE or (34, 16, 16)
    :param value:
    :return:
    """
    return _shapelike(value, allow_inference=True, positive_only=True, allow_sub_value_none=True)


def shapelike_float(value: str):
    """
    For shapelike (1.1001, 3.263)
    :param value:
    :return:
    """
    return _shapelike(value, positive_only=True, allow_float=True)


def shapelike_float_signed(value: str):
    """
    For shapelike (None, -1.129) or (1.236, None)
    :param value:
    :return:
    """
    return _shapelike(value, allow_float=True, allow_sub_value_none=True)


def rangelike(value: str):
    """
    For multiple shapelike ((None, 3), (55, None)
    :param value:
    :return:
    """
    parsed = eval(value)
    if not isinstance(parsed, (tuple, list)):
        raise TypeError("%s is not a valid rangelike value." % value)
    result = []
    for _range in parsed:
        result.append(_shapelike(str(_range), allow_none=True, allow_sub_value_none=True))
    return tuple(result)


def _container(value: str, allowed_type: Union[type, tuple, list]):
    result = eval(value)
    for t in allowed_type:
        if isinstance(result, t):
            if result is None:
                return result
            return (result,)
    result = tuple(result)
    for element in result:
        if allowed_type and not isinstance(element, allowed_type):
            raise TypeError("Received type %s for element %s instead of %s"
                            % (str(type(element)), str(element), str(allowed_type)))
    return result


def string_container(value: str) -> tuple:
    """
    Container for multiple string
    :param value:
    :return:
    """
    # noinspection PyBroadException
    try:
        result = _container(value, (str, type(None)))
    except:
        result = (value,)
    return result


def process_eval(value: str):
    return eval(value)


def int_container(value: str) -> tuple:
    """
    Container for multiple integer
    :param value:
    :return:
    """
    if not value:
        return ()
    try:
        result = _container(value, (int, type(None)))
    except TypeError as terr:
        raise TypeError(("Invalid value %s for int_container: " + str(terr.args))
                        % value)
    except Exception as e:
        raise ValueError(("Invalid value %s for int_container: " + str(e.args))
                         % value)
    else:
        return result


def process_int(value: str) -> int:
    """
    Integer
    :param value:
    :return:
    """
    try:
        result = eval(value)
    except:
        raise TypeError(("Invalid value %s for int: " + value)
                        % value)
    else:
        if isinstance(result, int):
            return result
        else:
            raise TypeError(("Invalid value %s for int: " + value)
                            % value)


def process_float(value: str) -> float:
    """
    Integer
    :param value:
    :return:
    """
    try:
        result = float(value)
    except:
        raise TypeError(("Invalid value %s for float: " + value)
                        % value)
    else:
        return result


def process_dict(value: str) -> dict:
    """
    dictionary
    :param value:
    :return:
    """
    try:
        result = eval(value)
    except:
        raise ValueError("Invalid value %s for dict" % value)
    else:
        if isinstance(result, dict):
            return result
        elif isinstance(result, type(None)):
            return result
        else:
            raise TypeError("Value %s is not a dict" % value)
