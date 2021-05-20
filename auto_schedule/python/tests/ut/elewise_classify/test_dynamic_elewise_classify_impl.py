# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
import tbe
from tbe.common import buildcfg
from tbe.dsl.base.classifier import classify_elewise

warnings.filterwarnings("ignore")
ut_case = OpUT("elewise_classify", "elewise_classify.test_dynamic_elewise_classify_impl")


def test_elewise_prebuild_classify(_):
    dynamic_config = buildcfg.default_buildcfg.dynamic_build_config_dict
    dynamic_config.update({"enable_op_prebuild": True})
    with buildcfg.build_config(**dynamic_config):
        with tbe.common.context.op_context.OpContext("dynamic"):
            inputs = [
                {"dtype": "float32", "shape": (-1, -1), "org_shape": (-1, -1), "range": [(1, None), (1, None)], },
                {"dtype": "float32", "shape": (-1, -1), "org_shape": (-1, -1), "range": [(1, None), (1, None)], },
            ]
            ins = classify_elewise(inputs, False)

            expect_ins = [
                [
                    {'shape': [-1], 'range': [(1, None)], 'mode': 'special', 'support_broadcast': False, 'pattern': ('common',)},
                    {'shape': [-1], 'range': [(1, None)], 'mode': 'special', 'support_broadcast': False, 'pattern': ('common',)}
                ]
            ]

            return ins == expect_ins


ut_case.add_cust_test_func(test_func=test_elewise_prebuild_classify)
