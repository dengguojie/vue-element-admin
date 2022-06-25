# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
import tbe
import copy
from tbe.common import buildcfg
from tbe.dsl.classifier import elewise_classifier

warnings.filterwarnings("ignore")
ut_case = OpUT("elewise_classify",
               "elewise_classify.test_dynamic_elewise_classify_impl")


def test_elewise_prebuild_classify(_):
    org_dynamic_config = buildcfg.default_buildcfg.dynamic_build_config_dict
    dynamic_config = copy.deepcopy(org_dynamic_config)
    dynamic_config.update({"enable_op_prebuild": True})
    with buildcfg.build_config(**dynamic_config):
        with tbe.common.context.op_context.OpContext("dynamic"):
            inputs = [
                {"dtype": "float32", "shape": (-1, -1), "org_shape": (-1, -1), "range": [
                    (1, None), (1, None)], },
                {"dtype": "float32", "shape": (-1, -1), "org_shape": (-1, -1), "range": [
                    (1, None), (1, None)], },
            ]
            ins = elewise_classifier.classify(inputs, False)

            expect_ins = [
                [
                    {'shape': [-1], 'range': [(1, None)], 'mode': 'special',
                     'support_broadcast': False, 'pattern': ('common',)},
                    {'shape': [-1], 'range': [(1, None)], 'mode': 'special',
                     'support_broadcast': False, 'pattern': ('common',)}
                ]
            ]

            return ins == expect_ins


def test_elewise_not_fuse_classify(_):
    org_dynamic_config = buildcfg.default_buildcfg.dynamic_build_config_dict
    dynamic_config = copy.deepcopy(org_dynamic_config)
    dynamic_config.update({"enable_op_prebuild": False})
    with buildcfg.build_config(**dynamic_config):
        with tbe.common.context.op_context.OpContext("dynamic"):
            inputs = [
                {"dtype": "float32", "shape": (-1, -1, -1, -1, 16),
                 "ori_shape": (-1, -1, -1, -1),
                 "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)],
                 "format": "NC1HWC0",
                 "ori_format": "NHWC"},
                {"dtype": "float32", "shape": (-1, -1, -1, -1, 16),
                 "ori_shape": (-1, -1, -1, -1),
                 "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)],
                 "format": "NC1HWC0",
                 "ori_format": "NHWC"},
            ]
            extra_params = {"ignore_fractal_format": False}
            ins = elewise_classifier.classify(inputs, False, extra_params)

            expect_ins = [
                [
                    {'shape': [-1, -1, -1, 16], 'range': [(1, None), (1, None), (1, None), (16, 16)],
                     'mode': 'special', 'support_broadcast': False, 'pattern': ('not_all_fuse',), 'ori_shape': [-1, -1, -1, -1],
                     'format':'NC1HWC0', 's_format': [['N'], ['C1'], ['H', 'W'], ['C0']], 'pad_axes': {'C': 3},
                     'np_mapping': {'C1': 'C', 'C0': 'C'}, 'mode_5hd': True},
                    {'shape': [-1, -1, -1, 16], 'range': [(1, None), (1, None), (1, None), (16, 16)],
                     'mode': 'special', 'support_broadcast': False, 'pattern': ('not_all_fuse',), 'ori_shape': [-1, -1, -1, -1],
                     'format':'NC1HWC0', 's_format': [['N'], ['C1'], ['H', 'W'], ['C0']], 'pad_axes': {'C': 3},
                     'np_mapping': {'C1': 'C', 'C0': 'C'}, 'mode_5hd': True}
                ],
                [
                    {'shape': [-1], 'range': [(16, None)], 'mode': 'special',
                     'support_broadcast': False, 'pattern': ('common',)},
                    {'shape': [-1], 'range': [(16, None)], 'mode': 'special',
                     'support_broadcast': False, 'pattern': ('common',)}
                ]
            ]

            return ins == expect_ins


def test_elewise_not_fuse_classify_ori_c_infer(_):
    org_dynamic_config = buildcfg.default_buildcfg.dynamic_build_config_dict
    dynamic_config = copy.deepcopy(org_dynamic_config)
    dynamic_config.update({"enable_op_prebuild": False})
    with buildcfg.build_config(**dynamic_config):
        with tbe.common.context.op_context.OpContext("dynamic"):
            inputs = [
                {"dtype": "float32", "shape": (-1, 1, -1, -1, 16),
                 "ori_shape": (-1, -1, -1, 2),
                 "range": [(1, None), (1, 1), (1, None), (1, None), (16, 16)],
                 "format": "NC1HWC0",
                 "ori_format": "NHWC"},
                {"dtype": "float32", "shape": (-1, -1, -1, -1, 16),
                 "ori_shape": (-1, -1, -1, -1),
                 "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)],
                 "format": "NC1HWC0",
                 "ori_format": "NHWC"},
            ]
            extra_params = {"ignore_fractal_format": False}
            ins = elewise_classifier.classify(inputs, False, extra_params)

            expect_ins = [
                [
                    {'shape': [-1, 1, -1, 16], 'range': [(1, None), (1, 1), (1, None), (16, 16)],
                     'mode': 'special', 'support_broadcast': False, 'pattern': ('not_all_fuse',), 'ori_shape': [-1, -1, -1, 2],
                     'format':'NC1HWC0', 's_format': [['N'], ['C1'], ['H', 'W'], ['C0']], 'pad_axes': {'C': 3},
                     'np_mapping': {'C1': 'C', 'C0': 'C'}, 'mode_5hd': True},
                    {'shape': [-1, 1, -1, 16], 'range': [(1, None), (1, 1), (1, None), (16, 16)],
                     'mode': 'special', 'support_broadcast': False, 'pattern': ('not_all_fuse',), 'ori_shape': [-1, -1, -1, 2],
                     'format':'NC1HWC0', 's_format': [['N'], ['C1'], ['H', 'W'], ['C0']], 'pad_axes': {'C': 3},
                     'np_mapping': {'C1': 'C', 'C0': 'C'}, 'mode_5hd': True}
                ],
            ]

            return ins == expect_ins


ut_case.add_cust_test_func(test_func=test_elewise_prebuild_classify)
ut_case.add_cust_test_func(test_func=test_elewise_not_fuse_classify)
ut_case.add_cust_test_func(
    test_func=test_elewise_not_fuse_classify_ori_c_infer)
