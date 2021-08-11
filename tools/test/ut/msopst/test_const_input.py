import unittest
import numpy
from op_test_frame.st.interface import utils


class TestUtilsMethods(unittest.TestCase):
    def test_deal_with_const(self):
        const_input = utils.ConstInput(True)
        input_desc = {'format': 'NHWC', 'shape': [2], 'type': 'int32',
                      'value': [48, 48],
                      'is_const': True, 'name': 'x2'}
        const_input.deal_with_const(input_desc)

    def test_get_acl_const_status_with_value(self):
        desc_dict = {'format': 'NHWC', 'shape': [2], 'type': 'int32',
                     'value': [48, 48], 'is_const': True, 'name': 'x2'}
        res_desc_dic = {'format': 'NHWC', 'type': 'int32', 'shape': [2]}
        utils.ConstInput.add_const_info_in_acl_json(desc_dict, res_desc_dic)

    def test_get_acl_const_status_with_data_distribute(self):
        desc_dict = desc_dict = {'format': 'NHWC', 'shape': [2], 'type': 'int32',
                                 'data_distribute': 'uniform', 'value_range': [0.1, 1.0],
                                 'is_const': True, 'name': 'x2'}
        res_desc_dic = {'format': 'NHWC', 'type': 'int32', 'shape': [2]}
        utils.ConstInput.add_const_info_in_acl_json(desc_dict, res_desc_dic)

    @unittest.mock.patch('op_test_frame.st.interface.utils.np.fromfile')
    def test_get_acl_const_status_with_value_bin_file(self, getattr_mock):
        desc_dict = {'format': 'NHWC', 'shape': [2], 'type': 'int32',
                     'value': "a.bin", 'is_const': True, 'name': 'x2'}
        res_desc_dic = {'format': 'NHWC', 'type': 'int32', 'shape': [2]}
        getattr_mock.return_value = numpy.array([0, 0])
        utils.ConstInput.add_const_info_in_acl_json(desc_dict, res_desc_dic)

    def test_get_acl_const_status(self):
        testcase_struct = {
            'op': '',
            'input_desc': [
                {
                    'format': 'NHWC', 'shape': [2], 'type': 'int32',
                    'data_distribute': 'uniform',
                    'value_range': [0.1, 1.0], 'value': [48, 48],
                    'is_const': True, 'name': 'x2'}]}
        utils.ConstInput.get_acl_const_status(testcase_struct)


if __name__ == '__main__':
    unittest.main()
