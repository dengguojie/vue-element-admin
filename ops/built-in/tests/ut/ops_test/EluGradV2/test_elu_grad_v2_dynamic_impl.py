#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("EluGradV2", "impl.dynamic.elu_grad_v2", "elu_grad_v2")


ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "output", "range": [(1, 100), (1, 100)]}, 1.0],
    "case_name": "elu_grad_v2_dynamic_success_case_1",
    "expect": "success",
    "formact_expect": [],
    "support_expect": True
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (-2, ), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input"},
               {"shape": (-2, ), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input"},
               {"shape": (-2, ), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "output"}, 1.0],
    "case_name": "elu_grad_v2_dynamic_success_case_2",
    "expect": "success",
    "formact_expect": [],
    "support_expect": True
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input", "range": [(1, None), (1, None)]},
               {"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input", "range": [(1, None), (1, None)]},
               {"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "output", "range": [(1, None), (1, None)]}, 1.0],
    "case_name": "elu_grad_v2_dynamic_success_case_3",
    "expect": "success",
    "formact_expect": [],
    "support_expect": True
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (100, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"shape": (100, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"shape": (100, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "output", "range": [(1, 100), (1, 100)]}, 1.0],
    "case_name": "elu_grad_v2_dynamic_success_case_4",
    "expect": "success",
    "formact_expect": [],
    "support_expect": True
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "output", "range": [(1, 100), (1, 100)]}, 1.0],
    "case_name": "elu_grad_v2_dynamic_failed_case_1",
    "expect": RuntimeError,
    "formact_expect": [],
    "support_expect": True
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float16",
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "output", "range": [(1, 100), (1, 100)]}, -1.0],
    "case_name": "elu_grad_v2_dynamic_failed_case_2",
    "expect": RuntimeError,
    "formact_expect": [],
    "support_expect": True
})

ut_case.add_case(support_soc="Ascend910A", case={
    "params": [{"shape": ( -1, ), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"shape": (-1, -1), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "input", "range": [(1, 100), (1, 100)]},
               {"shape": (-2, ), "ori_shape": (100, 100), "format": "ND", "ori_format": "ND", "dtype": "float32",
                "param_type": "output"}, 1.0],
    "case_name": "elu_grad_v2_dynamic_failed_case_3",
    "expect": RuntimeError,
    "formact_expect": [],
    "support_expect": True
})

if __name__ == "__main__":
    ut_case.run("Ascend910A")
    