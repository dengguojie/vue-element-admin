# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

import tbe
from tbe.dsl.unify_schedule.vector.reduce import reduce_tilingcase
from tbe.dsl.unify_schedule.vector.reduce import vector_schedule
from tbe.dsl.unify_schedule.vector.reduce import reduce_atomic_schedule

warnings.filterwarnings("ignore")

ut_case = OpUT("context", "context.test_dynamic_reduce_func_impl", "dsl_context")


def test_reduceTilingCase_eq(_):
    reduceTilingCase = reduce_tilingcase.ReduceTilingCase()
    reduceTilingCase2 = reduce_tilingcase.ReduceTilingCase()
    return reduceTilingCase.__eq__(reduceTilingCase2)


def test_reduceTilingCase_ne(_):
    reduceTilingCase = reduce_tilingcase.ReduceTilingCase()
    reduceTilingCase2 = reduce_tilingcase.ReduceTilingCase()
    reduceTilingCase2.block_split_axis_index = 10
    return reduceTilingCase.__ne__(reduceTilingCase2)


def test_reduceTilingCase_tilingKey(_):
    try:
        reduce_tilingcase._get_tiling_key(312312, 23223223, 3123, 23131, 232312, (1, 2, 3, 4, 5, 6), [1, 3])
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90003":
            return True
    return False


def test_reduceTilingCase_raisingError(_):
    try:
        reduce_tilingcase._raise_error("Error")
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90003":
            return True
    return False


def test_vectorSchedule_PlaceHolder_ne(_):
    placeHolder = vector_schedule.VectorSchedule.Placeholder(vector_schedule.VectorSchedule.Placeholder.PlaceholderType.RFACTOR_TENSOR,"2233")
    placeHolder2 = vector_schedule.VectorSchedule.Placeholder(vector_schedule.VectorSchedule.Placeholder.PlaceholderType.TILING_OUTER,"43242")

    return placeHolder.__ne__(placeHolder2)


test_func_list = [
    test_reduceTilingCase_eq,
    test_reduceTilingCase_ne,
    test_reduceTilingCase_tilingKey,
    test_reduceTilingCase_raisingError,
    test_vectorSchedule_PlaceHolder_ne,

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