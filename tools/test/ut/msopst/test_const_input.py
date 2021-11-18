import unittest
import numpy
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.data_generator import DataGenerator
from op_test_frame.st.interface.st_report import OpSTReport


class TestUtilsMethods(unittest.TestCase):
    def test_deal_with_const(self):
        const_input = utils.ConstInput(True)
        input_desc = {'format': 'NHWC', 'shape': [2], 'type': 'int32',
                      'value': [48, 48],
                      'is_const': True, 'name': 'x2'}
        for_fuzz = True
        const_input.deal_with_const(input_desc, for_fuzz)
        for_fuzz = False
        const_input.deal_with_const(input_desc, for_fuzz)

    def test_get_acl_const_status_with_value(self):
        desc_dict = {'format': 'NHWC', 'shape': [2], 'type': 'int32',
                     'value': [48, 48], 'is_const': True, 'name': 'x2'}
        res_desc_dic = {'format': 'NHWC', 'type': 'int32', 'shape': [2]}
        utils.ConstInput.add_const_info_in_acl_json(desc_dict, res_desc_dic, "", "xx", 0)

    def test_get_acl_const_status_with_data_distribute(self):
        desc_dict = desc_dict = {'format': 'NHWC', 'shape': [2], 'type': 'int32',
                                 'data_distribute': 'uniform', 'value_range': [0.1, 1.0],
                                 'is_const': True, 'name': 'x2'}
        res_desc_dic = {'format': 'NHWC', 'type': 'int32', 'shape': [2]}
        utils.ConstInput.add_const_info_in_acl_json(desc_dict, res_desc_dic, "", "xx", 0)

    @unittest.mock.patch('op_test_frame.st.interface.utils.np.fromfile')
    def test_get_acl_const_status_with_value_bin_file(self, getattr_mock):
        desc_dict = {'format': 'NHWC', 'shape': [2], 'type': 'int32',
                     'value': "a.bin", 'is_const': True, 'name': 'x2'}
        res_desc_dic = {'format': 'NHWC', 'type': 'int32', 'shape': [2]}
        getattr_mock.return_value = numpy.array([0, 0])
        utils.ConstInput.add_const_info_in_acl_json(desc_dict, res_desc_dic, "", "xx", 0)

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

    def test_gen_scalar_data_with_value(self):
        report = OpSTReport()
        data_generator = DataGenerator([], '/home', True, report)
        data = data_generator.gen_data_with_value((), [5], numpy.int32)
        expect_data = numpy.array([5], dtype=numpy.int32)
        self.assertEqual(data, expect_data)


if __name__ == '__main__':
    unittest.main()
