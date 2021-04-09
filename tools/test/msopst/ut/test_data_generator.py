import unittest
import pytest
import numpy as np
from unittest import mock
from op_test_frame.st.interface import utils
from op_test_frame.st.interface.data_generator import DataGenerator
from op_test_frame.st.interface.st_report import OpSTReport


class TestUtilsMethods(unittest.TestCase):

    def test_generate1(self):
        report = OpSTReport()
        data_generator = DataGenerator([], '/home', True, report)
        distribution_list = ['normal', 'beta', 'laplace', 'triangular', 'relu',
                             'sigmoid', 'softmax', 'tanh']
        for i in distribution_list:
            data_generator.gen_data((64,6), 1, 10, np.bool, i)

    def test_generate_error(self):
        report = OpSTReport()
        with pytest.raises(utils.OpTestGenException) as error:
            data_generator = DataGenerator([], '/home', True, report)
            data_generator.gen_data((64,6), 1, 10, np.bool, 'error')
        self.assertEqual(error.value.args[0],
                         utils.OP_TEST_GEN_WRITE_FILE_ERROR)


if __name__ == '__main__':
    unittest.main()