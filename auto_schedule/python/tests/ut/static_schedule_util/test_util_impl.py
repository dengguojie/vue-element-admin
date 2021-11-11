from sch_test_frame.ut import OpUT
from tbe.dsl.static_schedule import util
from te import tvm


ut_case = OpUT("static_schedule_util", "shape_util.test_util_impl")


def test_fake_node_fuse_fun(_):
    tensor0 = tvm.placeholder([16, 16, 16], name='tensor0', dtype="float16")
    tensor1 = tvm.placeholder([16, 16], name='tensor1', dtype="float16")
    result = util.fake_node_fuse_fun([tensor0, tensor1])
    return result == -1


case_list = [
    test_fake_node_fuse_fun,
]
for item in case_list:
    ut_case.add_cust_test_func(test_func=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)