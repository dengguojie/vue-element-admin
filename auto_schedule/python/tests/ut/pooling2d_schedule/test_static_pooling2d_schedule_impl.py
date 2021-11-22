# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

from tbe.dsl.static_schedule.pooling2d_schedule import _check_blockdims
from te import tvm
import te.lang.cce as tbe
from impl.strided_write import strided_write_compute

warnings.filterwarnings("ignore")


ut_case = OpUT("pooling2d_schedule", "cce_schedule.test_static_pooling2d_schedule_impl")


def test_check_blockdims(_):
    try:
        batch_size, batch_factor, device_core_num = (65536, 1, 32)
        _check_blockdims(batch_size, batch_factor, device_core_num)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_gap_not_support_fuse_stride_write(_):
    try:
        data1 = tvm.placeholder((2, 4, 16, 16, 16), name='data1', dtype="float16")

        res = tbe.pooling2d(data1, (16, 16), (2, 2), "GAP", "SAME", (0, 0, 0, 0), (1, 1), 0, 0,
                            {}, "high_performance")
        res = strided_write_compute(res, None, 1, 4)

        tensor_list = [data1, res]
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        config = {
            "print_ir": False,
            "name": test_gap_not_support_fuse_stride_write,
            "tensor_list": tensor_list
        }
        tbe.cce_build_code(sch, config)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_check_l1_fusion_para_error1(_):
    try:
        data1 = tvm.placeholder((1, 12, 17, 17, 16), name='data1', dtype="float16")

        res = tbe.pooling2d(data1, (3, 3), (1, 1), "AVG", "SAME", (0, 0, 0, 0), (1, 1), 0, 0,
                            {"l1_fusion_type": 0, "in_l1_flag": 0}, "high_performance")
        tensor_list = [data1, res]
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        config = {
            "print_ir": False,
            "name": test_gap_not_support_fuse_stride_write,
            "tensor_list": tensor_list
        }
        tbe.cce_build_code(sch, config)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_check_l1_fusion_para_error2(_):
    try:
        data1 = tvm.placeholder((1, 12, 17, 17, 16), name='data1', dtype="float16")

        res = tbe.pooling2d(data1, (3, 3), (1, 1), "AVG", "SAME", (0, 0, 0, 0), (1, 1), 0, 0,
                            {"l1_fusion_type": 0, "L1_addr_flag": 1, "L1_valid_size": 0}, "high_performance")
        tensor_list = [data1, res]
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        config = {
            "print_ir": False,
            "name": test_gap_not_support_fuse_stride_write,
            "tensor_list": tensor_list
        }
        tbe.cce_build_code(sch, config)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_check_l1_fusion_para_error3(_):
    try:
        data1 = tvm.placeholder((1, 12, 17, 17, 16), name='data1', dtype="float16")

        res = tbe.pooling2d(data1, (3, 3), (1, 1), "AVG", "SAME", (0, 0, 0, 0), (1, 1), 0, 0,
                            {"l1_fusion_type": 0, "L1_addr_flag": 0, "L1_valid_size": 5224}, "high_performance")
        tensor_list = [data1, res]
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        config = {
            "print_ir": False,
            "name": test_gap_not_support_fuse_stride_write,
            "tensor_list": tensor_list
        }
        tbe.cce_build_code(sch, config)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True

test_func_list = [
    test_check_blockdims,
    test_gap_not_support_fuse_stride_write,
    test_check_l1_fusion_para_error1,
    test_check_l1_fusion_para_error2,
    test_check_l1_fusion_para_error3,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
