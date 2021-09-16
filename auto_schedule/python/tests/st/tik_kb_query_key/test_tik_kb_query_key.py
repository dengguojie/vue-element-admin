import unittest
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


class TestGetOpCompileUniqueKey(unittest.TestCase):

    def tearDown(self):
        pass

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    def setUpClass(cls):
        pass

    def test_get_op_compile_unique_key(self):
        op_sha = get_op_compile_unique_key(op_info[-1].op_type,
                                           op_info[-1].inputs,
                                           op_info[-1].outputs,
                                           op_info[-1].attrs,
                                           op_info[-1].extra_params)

        assert op_sha == "transpose_7ef617a404f4663abd2a5c12dcb20e12ad0a7b7b7735ec697a76348f6f044a11"

    def test_get_op_compile_unique_key_op_type_error(self):
        op_type = random.choice([{"op_type": "Transpose"}, True, 555, ["Transpose"]])
        err_msg = "The get_op_compile_unique_key api param op_type must be str, but get %s" % type(op_type)
        op_info = [Info()]
        self.assertRaisesRegex(RuntimeError, err_msg, get_op_compile_unique_key,
                               op_type,
                               op_info[-1].inputs,
                               op_info[-1].outputs,
                               op_info[-1].attrs,
                               op_info[-1].extra_params)

    def test_get_op_compile_unique_key_others_type_error(self):
        err_param = random.choice(["Transpose", 555, {"name": 'x'}, True])
        err_msg = " must be list or tuple, but get %s" % type(err_param)

        self.assertRaisesRegex(RuntimeError, "inputs" + err_msg, get_op_compile_unique_key,
                               op_info[-1].op_type,
                               err_param,
                               op_info[-1].outputs,
                               op_info[-1].attrs,
                               op_info[-1].extra_params)

        self.assertRaisesRegex(RuntimeError, "outputs" + err_msg, get_op_compile_unique_key,
                               op_info[-1].op_type,
                               op_info[-1].inputs,
                               err_param,
                               op_info[-1].attrs,
                               op_info[-1].extra_params)

        self.assertRaisesRegex(RuntimeError, "attrs" + err_msg, get_op_compile_unique_key,
                               op_info[-1].op_type,
                               op_info[-1].inputs,
                               op_info[-1].outputs,
                               err_param,
                               op_info[-1].extra_params)


def main():
    unittest.main()
    exit(0)


if __name__ == '__main__':
    main()
