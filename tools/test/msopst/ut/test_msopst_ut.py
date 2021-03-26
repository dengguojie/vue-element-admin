import unittest
import pytest
from unittest import mock
from op_test_frame.st.interface import utils


class TestUtilsMethods(unittest.TestCase):

    def test_print_error_log(self):
        utils.print_error_log("test error log")


if __name__ == '__main__':
    unittest.main()