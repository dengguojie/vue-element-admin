import unittest
from collections import OrderedDict

import pytest
from unittest import mock
from op_gen.interface import utils
from op_gen.interface.op_file_aicore import OpFileAiCore


class TestOpFileAiCoreMethods(unittest.TestCase):

    def test_generate_input_output_info_cfg(self):
        op_file = OpFileAiCore

        parsed_info = OrderedDict([('x', {
            'ir_type_list': ['DT_FLOAT', ' DT_INT32'],
            'param_type': 'required', 'format_list': ['NCHW']})])
        str_template = "input{index}.name={name}\ninput{index}.dtype={dtype}\ninput{index}.paramType={paramType}\ninput{index}.format={format}"
        self._mapping_info_cfg_type = mock.Mock(return_value="int8")
        ret = op_file._generate_input_output_info_cfg(self, parsed_info, str_template)
        print(ret)
        golden_ret = "input0.name=x\ninput0.dtype=int8,int8\ninput0.paramType=required\ninput0.format=NCHW,NCHW"
        self.assertEqual(golden_ret,ret)

if __name__ == '__main__':
    unittest.main()