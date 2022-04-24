from sch_test_frame.ut import OpUT
from tbe.common.utils import get_op_compile_unique_key
import random

op_infos = {
    "op_type": "Transpose",
    "inputs": (
        {
            "shape": (1000, 168, 64),
            "ori_shape": (1000, 168, 64),
            "format": "ND",
            "sub_format": 0,
            "ori_format": "ND",
            "dtype": 'float32',
            "addr_type": 0,
            "valid_shape": (),
            "slice_offset": (),
            "sgt_slice_shape": (),
            "L1_workspace_aize": -1,
            "L1_addr_offset": 0,
            "L1_fusion_type": -1,
            "total_shape": [1000, 168, 64],
            "split_index": 0,
            'name': "x"
        },
        (
            {
                "shape": (1000, 168, 64),
                "ori_shape": (1000, 168, 64),
                "format": "ND",
                "sub_format": 0,
                "ori_format": "ND",
                "dtype": 'float32',
                "addr_type": 0,
                "valid_shape": (),
                "slice_offset": (),
                "sgt_slice_shape": (),
                "L1_workspace_aize": -1,
                "L1_addr_offset": 0,
                "L1_fusion_type": -1,
                "total_shape": [1000, 168, 64],
                "split_index": 0,
                'name': "x"
            },
        ),
        None
    ),
    "outputs": (
        {
            "shape": (1000, 168, 64),
            "ori_shape": (1000, 168, 64),
            "format": "ND",
            "sub_format": 0,
            "ori_format": "ND",
            "dtype": 'float32',
            "addr_type": 0,
            "valid_shape": (),
            "slice_offset": (),
            "sgt_slice_shape": (),
            "L1_workspace_aize": -1,
            "L1_addr_offset": 0,
            "L1_fusion_type": -1,
            "total_shape": [1000, 168, 64],
            "split_index": 0,
            'name': "x"
        },
    ),
    "attrs": [
        {
            "impl_mode": "high_performanc",
            "options": {
                "invalid_datarm": True
            }
         },
        {
            "impl_mode": "high_performanc",
            "options": {
                "invalid_datarm": True
            }
        }
    ],
    "extra_params": {
        "name": "aaa",
        "dtype": "999",
        "shape": {
            "a": 1,
            "c": 3,
            "b": 2
        }
    }
}


class Info:
    def __init__(self):
        pass

    @property
    def op_type(self):
        return op_infos.get('op_type')

    @property
    def inputs(self):
        return op_infos.get('inputs')

    @property
    def outputs(self):
        return op_infos.get('outputs')

    @property
    def attrs(self):
        return op_infos.get('attrs')

    @property
    def extra_params(self):
        return op_infos.get('extra_params')


op_info = [Info()]

ut_case = OpUT("tik_kb_query_key", "tik_kb_query_key.test_tik_kb_query_key_impl")


def test_get_op_compile_unique_key(_):
    op_sha = get_op_compile_unique_key(op_info[-1].op_type,
                                       op_info[-1].inputs,
                                       op_info[-1].outputs,
                                       op_info[-1].attrs,
                                       op_info[-1].extra_params)

    return op_sha == "transpose_7ef617a404f4663abd2a5c12dcb20e12ad0a7b7b7735ec697a76348f6f044a11"


def test_get_op_compile_unique_key_op_type_error(_):
    op_type = random.choice([{"op_type": "Transpose"}, True, 555, ["Transpose"]])
    err_msg = "The get_op_compile_unique_key api param op_type must be str, but get %s" % type(op_type)
    try:
        op_sha = get_op_compile_unique_key(op_type,
                                           op_info[-1].inputs,
                                           op_info[-1].outputs,
                                           op_info[-1].attrs,
                                           op_info[-1].extra_params)
    except RuntimeError as e:
        return err_msg in str(e)
    return False


def test_get_op_compile_unique_key_inputs_type_error(_):
    err_param = random.choice(["Transpose", 555, {"name": 'x'}, True])
    err_msg = " must be list or tuple, but get %s" % type(err_param)

    try:
        op_sha = get_op_compile_unique_key(op_info[-1].op_type,
                                           err_param,
                                           op_info[-1].outputs,
                                           op_info[-1].attrs,
                                           op_info[-1].extra_params)
    except RuntimeError as e:
        return err_msg in str(e)
    return False


def test_get_op_compile_unique_key_outputs_type_error(_):
    err_param = random.choice(["Transpose", 555, {"name": 'x'}, True])
    err_msg = " must be list or tuple, but get %s" % type(err_param)

    try:
        op_sha = get_op_compile_unique_key(op_info[-1].op_type,
                                           op_info[-1].inputs,
                                           err_param,
                                           op_info[-1].attrs,
                                           op_info[-1].extra_params)
    except RuntimeError as e:
        return err_msg in str(e)
    return False


def test_get_op_compile_unique_key_attrs_type_error(_):
    err_param = random.choice(["Transpose", 555, {"name": 'x'}, True])
    err_msg = " must be list or tuple, but get %s" % type(err_param)

    try:
        op_sha = get_op_compile_unique_key(op_info[-1].op_type,
                                           op_info[-1].inputs,
                                           op_info[-1].outputs,
                                           err_param,
                                           op_info[-1].extra_params)
    except RuntimeError as e:
        return err_msg in str(e)
    return False


def test_get_op_compile_unique_key_str(_):
    op_str = get_op_compile_unique_key(op_info[-1].op_type,
                                       op_info[-1].inputs,
                                       op_info[-1].outputs,
                                       op_info[-1].attrs,
                                       op_info[-1].extra_params,
                                       is_sha=False)
    try:
        assert "op_type" in op_str
        assert "inputs" in op_str
        assert "outputs" in op_str
        assert "attrs" in op_str
        assert "extra_params" in op_str
    except AssertionError:
        return False
    return True


ut_case.add_cust_test_func(test_func=test_get_op_compile_unique_key)
ut_case.add_cust_test_func(test_func=test_get_op_compile_unique_key_op_type_error)
ut_case.add_cust_test_func(test_func=test_get_op_compile_unique_key_inputs_type_error)
ut_case.add_cust_test_func(test_func=test_get_op_compile_unique_key_outputs_type_error)
ut_case.add_cust_test_func(test_func=test_get_op_compile_unique_key_attrs_type_error)
ut_case.add_cust_test_func(test_func=test_get_op_compile_unique_key_str)
