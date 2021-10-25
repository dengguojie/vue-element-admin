# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
from tbe import tvm
import tbe
from tbe.dsl.unify_schedule.vector.broadcast.broadcast_tilingcase import BroadcastComputation
from tbe.dsl.unify_schedule.vector.broadcast.broadcast_tilingcase import TilingStrategy
from tbe.common.context import op_context
from tbe.dsl.base import operation

warnings.filterwarnings("ignore")
ut_case = OpUT("tilingCase", "broadcast_tilingcase.test_dynamic_broadcast_tilingcase_impl")


def test_empty_tiling(_):
    with op_context.OpContext("dynamic"):
        with tbe.dsl.compute():
            m = operation.var_inner("_m", (1, 10240))
            shape = (m, )
            input = tvm.placeholder(shape, name='data1', dtype="float16")
            out1 = tbe.dsl.cast_to(input, "float32")
            out0 = tbe.dsl.vrec(input)
            b_t = BroadcastComputation([out0, out1], None)
            operation.get_context().get_current_compute().add("_mode", "empty")
            tiling_case = b_t.do_tiling_case()
            return len(tiling_case) == 1 and \
                   tiling_case[0].tiling_key == 2**31 -1  and \
                   tiling_case[0].tiling_strategy == TilingStrategy.CONST


test_func_list = [
    test_empty_tiling,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)
