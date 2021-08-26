# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
from tbe import tvm
import tbe
from tbe.dsl.unify_schedule.vector.broadcast.broadcast_tilingcase import BroadcastTilingCase
from tbe.dsl.unify_schedule.vector.broadcast.broadcast_tilingcase import TilingStrategy
from tbe.dsl.unify_schedule.vector.broadcast.broadcast_schedule import BroadcastSchedule
from tbe.common.context import op_context
from tbe.dsl.base import operation

warnings.filterwarnings("ignore")
ut_case = OpUT("max_inputs", "broadcast_classify.test_dynamic_broadcast_schedule_impl")


def test_static_tiling(_):
    with op_context.OpContext("dynamic"):
        with tbe.dsl.compute():
            m = operation.var_inner("_m", (1, 10240))
            shape = (m, )
            input = tvm.placeholder(shape, name='data1', dtype="float16")
            out = tbe.dsl.cast_to(input, "float32")
            tiling_case = BroadcastTilingCase()
            tiling_case._tiling_key = 111
            tiling_case._tiling_strategy = TilingStrategy.STATIC
            tiling_case._block_split_axis = 0
            tiling_case._ub_split_axis = 0
            tiling_case._ub_factor_bound = 1024
            tiling_case._enable_db = False
            tiling_case._is_one_dim = True
            b_sch = BroadcastSchedule([out], tiling_case)
            b_sch._out = out
            b_sch._schedule = tvm.create_schedule(b_sch._out.op)
            b_sch._calc_tiling_static()
            b_sch._do_tiling_static()
            return True


def test_all_tiling(_):
    with op_context.OpContext("dynamic"):
        with tbe.dsl.compute():
            m = operation.var_inner("_m", (1, 10240))
            shape = (m, )
            input = tvm.placeholder(shape, name='data1', dtype="float16")
            out = tbe.dsl.cast_to(input, "float32")
            tiling_case = BroadcastTilingCase()
            tiling_case._tiling_key = 111
            tiling_case._tiling_strategy = TilingStrategy.ALL_CUT
            tiling_case._block_split_axis = 0
            tiling_case._ub_split_axis = 0
            tiling_case._ub_factor_bound = 1024
            tiling_case._enable_db = False
            tiling_case._is_one_dim = True
            b_sch = BroadcastSchedule([out], tiling_case)
            b_sch._out = out
            b_sch._schedule = tvm.create_schedule(b_sch._out.op)
            b_sch._calc_tiling_all_cut()
            b_sch._do_tiling_all_cut()
            return True


test_func_list = [
    test_static_tiling,
    test_all_tiling
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)
