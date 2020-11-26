import unittest
import os

current_file_path = __file__
cfg_ut_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),"ut/op_cfg_test")


def test_op_cfg():
    print(cfg_ut_path)
    all_tests = unittest.TestLoader().discover(start_dir=cfg_ut_path, pattern="test_*.py")

    suite = unittest.TestSuite()
    suite.addTests(all_tests)

    # 返回值result为type(result) = <class 'unittest.runner.TextTestResult'>
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    total_test_num = result.testsRun
    error_test_num = len(result.errors)
    fail_test_num = len(result.failures)
    if error_test_num > 0 or fail_test_num > 0:
        exit(-1)


if __name__ == "__main__":
    test_op_cfg()