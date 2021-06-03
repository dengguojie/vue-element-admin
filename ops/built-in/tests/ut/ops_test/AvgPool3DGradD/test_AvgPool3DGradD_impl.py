#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for AvgPool3DGradD
from op_test_frame.ut import OpUT


ut_case = OpUT("AvgPool3DGradD",
               "impl.avg_pool3d_grad_d",
               "avg_pool3d_grad_d")


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}

def _run_api_end_with_d(
    grads={'ori_shape': (1, 1, 1, 1, 1),
           'shape': (1, 1, 1, 1, 1, 16),
           'ori_format': 'NDHWC',
           'format': 'NDC1HWC0',
           'dtype': 'float16'},
    filter=None,
    multiplier=None,
    output={'ori_shape': (1, 3, 3, 3, 1),
            'shape': (1, 3, 1, 3, 3, 16),
            'ori_format': 'NDHWC',
            'format': 'NDC1HWC0',
            'dtype': 'float16'},
    orig_input_shape=(1, 3, 3, 3, 1),
    ksize=(1, 3, 3, 3, 1),
    strides=(1, 1, 1, 1, 1),
    pads=(0, 0, 0, 0, 0, 0),
    ceil_mode=False,
    count_include_pad=False,
    divisor_override=0,
    data_format="NDHWC"):
    return [grads, filter, multiplier, output, orig_input_shape,
            ksize, strides, pads, ceil_mode, count_include_pad,
            divisor_override, data_format]

def _test_op_get_op_support_info(test_arg):
    from impl.avg_pool3d_grad_d import get_op_support_info

    [grads, filter, multiplier, output, orig_input_shape,
     ksize, strides, pads, ceil_mode, count_include_pad,
     divisor_override, data_format] = _run_api_end_with_d()

    get_op_support_info(
        grads, filter, multiplier, output, orig_input_shape,
        ksize, strides, pads, ceil_mode, count_include_pad,
        divisor_override, data_format)

ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info)

# test_avg_pool3d_grad_d_succ in global mode
case1 = _run_api_end_with_d()

# test_avg_pool3d_grad_d_succ in cube mode
grads={'ori_shape': (9, 6, 4, 14, 48),
       'shape': (9, 6, 3, 4, 14, 16),
       'ori_format': 'NDHWC',
       'format': 'NDC1HWC0',
       'dtype': 'float16'
       }

filter={'ori_shape': (1, 2, 2, 1, 48),
        'shape': (12, 1, 16, 16),
        'ori_format': 'DHWCN',
        'format': 'NDC1HWC0',
        'dtype': 'float16'
        }

multiplier={'ori_shape': (9, 6, 4, 14, 48),
        'shape': (9, 6, 3, 4, 14, 16),
        'ori_format': 'NDHWC',
        'format': 'NDC1HWC0',
        'dtype': 'float32'
        }

output={'ori_shape': (9, 6, 28, 28, 48),
        'shape': (9, 6, 3, 28, 28, 16),
        'ori_format': 'NDHWC',
        'format': 'NDC1HWC0',
        'dtype': 'float16'
        }

orig_input_shape=(9, 6, 28, 28, 48)
ksize=(1, 1, 2, 2, 1)
strides=(1, 1, 9, 2, 1)
pads=(0, 0, 0, 1, 0, 0)

case2 = _run_api_end_with_d(grads=grads,
                            filter=filter,
                            multiplier=multiplier,
                            output=output,
                            orig_input_shape=orig_input_shape,
                            ksize=ksize,
                            strides=strides,
                            pads=pads)


# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case1, "success", "case1", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case2, "success", "case2", True))


if __name__ == '__main__':
    ut_case.run()
    exit(0)
