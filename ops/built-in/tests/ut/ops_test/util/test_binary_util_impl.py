# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
ut for util_binary.py
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("util_binary", "impl.util.util_binary")


# pylint: disable=unused-argument, import-outside-toplevel
def test_util_binary_api(test_arg):
    """
    test for util_binary api
    """
    from impl.dynamic import register_binary_match
    register_binary_match("Add")
    from impl.util.util_binary import get_bit_len
    assert get_bit_len("float16") == 16
    from impl.util.util_binary import binary_match
    add_match_func = binary_match("Add")
    assert add_match_func
    assert not add_match_func(
        {
            "dtype": "None",
            "format": "ND",
            "shape": [-2],
            "ori_shape": [-2],
            "ori_format": "ND"
        }, {
            "dtype": "int32",
            "format": "ND",
            "shape": [-2],
            "ori_shape": [-2],
            "ori_format": "ND"
        }, {
            "dtype": "int32",
            "format": "ND",
            "shape": [-2],
            "ori_shape": [-2],
            "ori_format": "ND"
        },
        generalize_config={"mode": "keep_rank"})

    from impl.util.util_binary import get_module_name
    assert get_module_name("Add") == "add"
    assert get_module_name("GatherV2D") == "gather_v2_d"
    assert get_module_name("LayerNormBetaGammaBackpropV2", "ascend910") == "layer_norm_beta_gamma_backprop_v2"


def test_match_tenser(test_arg):
    """
    test for test_match_tenser
    """
    from impl.util.util_binary import match_tenser

    # case 1, all shape is equal, return true
    target_tensor = {"dtype": "int32", "format": "ND", "shape": [-2], "ori_shape": [-2], "ori_format": "ND"}
    input_tensor = {"dtype": "int32", "format": "ND", "shape": [-2], "ori_shape": [-2], "ori_format": "ND"}
    assert match_tenser(input_tensor, target_tensor)

    # case 2, dtype is not equal, return false
    input_tensor = {"dtype": "uint32", "format": "ND", "shape": [-2], "ori_shape": [-2], "ori_format": "ND"}
    assert not match_tenser(input_tensor, target_tensor)

    # case 3, dtype is not equal, but Byte size is equal, return true
    target_tensor1 = {
        "dtype": "int32",
        "format": "ND",
        "shape": [-2],
        "ori_shape": [-2],
        "dtype_match_mode": "DtypeByte",
        "ori_format": "ND"
    }
    assert match_tenser(input_tensor, target_tensor1)

    # case 4, format is not equal, return false
    input_tensor = {
        "dtype": "int32",
        "format": "NC1HWC0",
        "shape": [-2],
        "ori_shape": [-2],
        "ori_format": "NC1HWC0"
    }
    assert not match_tenser(input_tensor, target_tensor)

    # case 5, format is not equal, return
    target_tensor1 = {
        "dtype": "int32",
        "format": "ND",
        "shape": [-2],
        "ori_shape": [-2],
        "dtype_match_mode": "DtypeByte",
        "ori_format": "ND",
        "format_match_mode": "FormatAgnostic"
    }
    assert match_tenser(input_tensor, target_tensor1)

    # case 6, all shape is equal and not unknown rank, return true
    target_tensor = {"dtype": "int32", "format": "ND", "shape": [-1], "ori_shape": [-1], "ori_format": "ND"}
    input_tensor = {"dtype": "int32", "format": "ND", "shape": [-1], "ori_shape": [-1], "ori_format": "ND"}
    assert match_tenser(input_tensor, target_tensor)

    # case 7, support bool == int8
    target_tensor = {"dtype": "bool", "format": "ND", "shape": [-1], "ori_shape": [-1], "ori_format": "ND"}
    input_tensor = {"dtype": "int8", "format": "ND", "shape": [-1], "ori_shape": [-1], "ori_format": "ND"}
    assert match_tenser(input_tensor, target_tensor)

    # case 8, support bool != uint8
    target_tensor = {"dtype": "bool", "format": "ND", "shape": [-1], "ori_shape": [-1], "ori_format": "ND"}
    input_tensor = {"dtype": "uint8", "format": "ND", "shape": [-1], "ori_shape": [-1], "ori_format": "ND"}
    assert not match_tenser(input_tensor, target_tensor)


def test_match_attr(test_arg):
    """
    test for test_match_attr
    """
    # test match attr
    from impl.util.util_binary import match_attr
    # case 0: target attr is None, return true
    target_attr = {"name": "groups", "dtype": "int", "value": None}
    input_attr = 0
    assert match_attr(input_attr, target_attr)

    # case 1: target attr is fixed, and input_attr == target_attr.value, return true
    target_attr = {"name": "groups", "dtype": "int", "value": 0}
    assert match_attr(input_attr, target_attr)

    # case 2: target attr is fixed, and input_attr != target_attr.value, return false
    target_attr = {"name": "groups", "dtype": "int", "value": 1}
    assert not match_attr(input_attr, target_attr)

    # case 3: target attr do not have values, return false
    target_attr = {"name": "groups", "dtype": "int"}
    assert not match_attr(input_attr, target_attr)
    # case 4: target attr is None, return false
    assert not match_attr(input_attr, None)


def test_update_args(test_arg):
    """
    test for test_args
    """
    from impl.util.util_binary import update_args
    target_rule = {
        "attrs": [{
            "dtype": "bool",
            "name": "align_corners",
            "value": True
        }, {
            "dtype": "bool",
            "name": "half_pixel_centers",
            "value": True
        }],
        "inputs": [{
            "index": 0,
            "shape": [-2],
            "dtype": "float32",
            "format": "NC1HWC0"
        }, {
            "index": 1,
            "shape": [-2],
            "dtype": "int32",
            "format": "ND"
        }],
        "outputs": [{
            "index": 0,
            "shape": [-2],
            "dtype": "float32",
            "format": "NC1HWC0"
        }]
    }
    input_1 = {
        'shape': (-1, -1, -1, -1, 16),
        'ori_shape': (-1, -1, -1, -1),
        'format': 'NC1HWC0',
        'sub_format': 0,
        'ori_format': 'NHWC',
        'dtype': 'float16',
        'addr_type': 0,
        'total_shape': [-1, -1, -1, -1, 16],
        'slice_offset': (),
        'L1_addr_offset': 0,
        'L1_fusion_type': -1,
        'L1_workspace_size': -1,
        'valid_shape': (),
        'split_index': 0,
        'range': ((1, None), (1, None), (1, None), (1, None), (16, 16)),
        'ori_range': ((1, None), (1, None), (1, None), (1, None)),
        'param_name': 'images'
    }
    input_2 = {
        'shape': (2,),
        'ori_shape': (2,),
        'format': 'NHWC',
        'sub_format': 0,
        'ori_format': 'NHWC',
        'dtype': 'int32',
        'addr_type': 0,
        'total_shape': [2],
        'slice_offset': (),
        'L1_addr_offset': 0,
        'L1_fusion_type': -1,
        'L1_workspace_size': -1,
        'valid_shape': (),
        'split_index': 0,
        'range': [[2, 2]],
        'ori_range': (),
        'param_name': 'size'
    }
    output_1 = {
        'shape': (-1, -1, -1, -1, 16),
        'ori_shape': (-1, -1, -1, -1),
        'format': 'NC1HWC0',
        'sub_format': 0,
        'ori_format': 'NHWC',
        'dtype': 'float32',
        'addr_type': 0,
        'total_shape': [-1, -1, -1, -1, 16],
        'slice_offset': (),
        'L1_addr_offset': 0,
        'L1_fusion_type': -1,
        'L1_workspace_size': -1,
        'valid_shape': (),
        'split_index': 0,
        'range': ((1, None), (1, None), (0, None), (0, None), (16, 16)),
        'ori_range': ((1, None), (0, None), (0, None), (1, None)),
        'param_name': 'y'
    }

    input_args = (input_1, input_2, output_1, False, False)
    args_res = update_args(input_args, target_rule)
    assert list(args_res[0].get("shape")) == [-2]
    assert args_res[3]
    assert args_res[4]


def test_import_lib(test_arg):
    """
    test_import_lib
    """
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.dynamic.binary_query_register"))


def test_match(test_arg):
    from impl.util.util_binary import BinaryMatchBase
    input_1 = {
        'shape': (-1, -1, -1, -1, 16),
        'ori_shape': (-1, -1, -1, -1),
        'format': 'NC1HWC0',
        'sub_format': 0,
        'ori_format': 'NHWC',
        'dtype': 'float16',
        'addr_type': 0,
        'total_shape': [-1, -1, -1, -1, 16],
        'slice_offset': (),
        'L1_addr_offset': 0,
        'L1_fusion_type': -1,
        'L1_workspace_size': -1,
        'valid_shape': (),
        'split_index': 0,
        'range': ((1, None), (1, None), (1, None), (1, None), (16, 16)),
        'ori_range': ((1, None), (1, None), (1, None), (1, None)),
        'param_name': 'images'
    }
    input_2 = {
        'shape': (2,),
        'ori_shape': (2,),
        'format': 'NHWC',
        'sub_format': 0,
        'ori_format': 'NHWC',
        'dtype': 'int32',
        'addr_type': 0,
        'total_shape': [2],
        'slice_offset': (),
        'L1_addr_offset': 0,
        'L1_fusion_type': -1,
        'L1_workspace_size': -1,
        'valid_shape': (),
        'split_index': 0,
        'range': [[2, 2]],
        'ori_range': (),
        'param_name': 'size'
    }
    output_1 = {
        'shape': (-1, -1, -1, -1, 16),
        'ori_shape': (-1, -1, -1, -1),
        'format': 'NC1HWC0',
        'sub_format': 0,
        'ori_format': 'NHWC',
        'dtype': 'float32',
        'addr_type': 0,
        'total_shape': [-1, -1, -1, -1, 16],
        'slice_offset': (),
        'L1_addr_offset': 0,
        'L1_fusion_type': -1,
        'L1_workspace_size': -1,
        'valid_shape': (),
        'split_index': 0,
        'range': ((1, None), (1, None), (0, None), (0, None), (16, 16)),
        'ori_range': ((1, None), (0, None), (0, None), (1, None)),
        'param_name': 'y'
    }
    input_args = (input_1, input_2, output_1, False, False)
    input_key_compile = {
        BinaryMatchBase.GENERALIZATIO_KEY_NAME: {
            BinaryMatchBase.GENERALIZATIO_MODE_KEY_NAME: BinaryMatchBase.GENERALIZATIO_MODE_COMPILE
        }
    }
    input_key_binary = {
        BinaryMatchBase.GENERALIZATIO_KEY_NAME: {
            BinaryMatchBase.GENERALIZATIO_MODE_KEY_NAME: BinaryMatchBase.GENERALIZATIO_MODE_BINARY
        }
    }
    rule_op = BinaryMatchBase("Add")

    assert rule_op.match_result(*input_args, **input_key_compile) is None
    assert rule_op.match_result(*input_args, **input_key_binary) is None
    rule_op.get_binary_rule()
    assert rule_op.match_result(*input_args, **input_key_binary) is None

    # match case
    rule_op.input_num = 2
    rule_op.output_num = 1
    rule_op.attr_num = 0
    rule_op.arg_minest_num = 3
    target_inputs = [{
        "shape": [-2],
        "dtype": "float16",
        "format": "ND"
    }, {
        "shape": [-2],
        "dtype": "float16",
        "format": "ND"
    }]
    target_outputs = [{"shape": [-2], "dtype": "float16", "format": "ND"}]
    target_attrs = []
    rule_op.binary_rule_list = [{"inputs": target_inputs, "outputs": target_outputs, "attrs": target_attrs}]
    match_input_args = ({
        "shape": [10],
        "dtype": "float16",
        "format": "ND"
    }, {
        "shape": [10],
        "dtype": "float16",
        "format": "ND"
    }, {
        "shape": [10],
        "dtype": "float16",
        "format": "ND"
    })
    match_res = rule_op.match_result(*match_input_args, **input_key_binary)
    assert match_res[0][0] == {"shape": [-2], "dtype": "float16", "format": "ND"}


def test_match_format(test_arg):
    from impl.util.util_binary import match_format
    input_tensor = {'format': 'NC1HWC0'}
    target_tensor = {'format': 'ND'}
    assert not match_format(input_tensor, target_tensor)
    input_tensor = {'format': 'NCHW'}
    assert match_format(input_tensor, target_tensor)


ut_case.add_cust_test_func("all", test_func=test_util_binary_api)
ut_case.add_cust_test_func("all", test_func=test_match_tenser)
ut_case.add_cust_test_func("all", test_func=test_match_attr)
ut_case.add_cust_test_func("all", test_func=test_update_args)
ut_case.add_cust_test_func("all", test_func=test_import_lib)
ut_case.add_cust_test_func("all", test_func=test_match)
ut_case.add_cust_test_func("all", test_func=test_match_format)


if __name__ == "__main__":
    ut_case.run(["Ascend910A"])
