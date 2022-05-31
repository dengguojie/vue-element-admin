# # -*- coding:utf-8 -*-
import warnings

from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.dsl.base.padding import simulator
from tbe.dsl.base.padding.graph import Graph, Node
from tbe.dsl.base.padding.simulator import Simulator, SimulatorManager

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_simulator_impl")


@add_cust_test_func(ut_case)
def test_get_simulator_when_exist(_):
    class S1(Simulator):
        def adjust_calc(self):
            pass
        @classmethod
        def get_type(cls):
            return "S1"

    shape = (2, 16)
    ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
    with tvm.tag_scope("S1"):
        add_1 = tvm.compute(shape, lambda *i: ph_1[i] + 2, name="add_1")
    node = Node(Graph([add_1]), add_1)

    assert_s1 = simulator.get_simulator(node) is not None
    return assert_s1


@add_cust_test_func(ut_case)
def test_get_simulator_when_not_exist(_):
    shape = (2, 16)
    ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
    with tvm.tag_scope("S2"):
        add_1 = tvm.compute(shape, lambda *i: ph_1[i] + 2, name="add_1")

    try:
        node = Node(Graph([add_1]), add_1)
        simulator.get_simulator(node)
    except RuntimeError as e:
        return "Can not find simulator" in e.args[1]
    return False


@add_cust_test_func(ut_case)
def test_add_class(_):
    class S1(Simulator): pass
    class S2: pass

    assert_s1 = S1 in SimulatorManager._simulator_classes
    assert_s2 = S2 not in SimulatorManager._simulator_classes

    return assert_s1 and assert_s2


@add_cust_test_func(ut_case)
def test_build(_):
    class S1(Simulator):
        def adjust_calc(self):
            pass
        @classmethod
        def get_type(cls):
            return "S1"
    class S2: pass

    assert_s1 = SimulatorManager.build("S1", None) is not None
    assert_s2 = SimulatorManager.build("S2", None) is None

    return assert_s1 and assert_s2


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        ret = v.test_func(None)
        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
