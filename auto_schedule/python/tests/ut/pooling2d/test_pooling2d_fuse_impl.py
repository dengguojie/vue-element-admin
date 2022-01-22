# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
from enum import Enum

from te import tvm
import te.lang.cce as tbe
from impl.ascend_anti_quant import ascend_anti_quant_compute
from impl.ascend_quant import ascend_quant_compute
from impl.strided_write import strided_write_compute
from sch_test_frame.ut import OpUT
import warnings
from enum import Enum

from te import tvm
import te.lang.cce as tbe
from impl.ascend_anti_quant import ascend_anti_quant_compute
from impl.ascend_quant import ascend_quant_compute
from impl.strided_write import strided_write_compute
from sch_test_frame.common import precision_info

warnings.filterwarnings("ignore")


class FuseType(Enum):
    ANTI_QUANT_POOLING ="1"
    POOLING_QUANT ="2"
    ANTI_QUANT_POOLING_QUANT = "3"
    POOLING_STRIDED_WRITE = "4"


def dsl_pooling2d_fuse(x, y, fuse_type, window, stride, pooling_mode, padding_mode, pad, dilation, data_mode,
                       ceil_mode, fusion_params, impl_mode, kernel_name='dsl_pooling2d_fuse'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    if fuse_type == 3:
        res = ascend_anti_quant_compute(data1, y, 0.9551185, 128)
        res = tbe.pooling2d(res, window, stride, pooling_mode, padding_mode, pad, dilation, data_mode, ceil_mode,
                            fusion_params, impl_mode)
    elif fuse_type == 4:
        res = tbe.pooling2d(data1, window, stride, pooling_mode, padding_mode, pad, dilation, data_mode, ceil_mode,
                            fusion_params, impl_mode)
        res = strided_write_compute(res, None, 1, 4)
    elif fuse_type == 5:
        res = tbe.pooling2d(data1, window, stride, pooling_mode, padding_mode, pad, dilation, data_mode, ceil_mode,
                            fusion_params, impl_mode)
        res = ascend_quant_compute(res, y, 36.9753761, -128, False)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("pooling2d_fuse", "pooling2d.test_pooling2d_fuse_impl", "dsl_pooling2d_fuse")


case1 = {
    "params": [{"shape": (4, 8, 28, 28, 32), "dtype": "int8"},
               {"shape": (4, 15, 28, 28, 16), "dtype": "float16", "ori_shape": (4, 280, 28, 28)},
               3
               ],
    "case_name": "test_pooling2d_anti_quant_avg_quant",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 3), "stride": (2, 2), "pooling_mode": "AVG", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case2 = {
    "params": [{"shape": (4, 8, 28, 28, 32), "dtype": "int8"},
               {"shape": (4, 15, 28, 28, 16), "dtype": "float16", "ori_shape": (4, 280, 28, 28)},
               3
               ],
    "case_name": "test_pooling2d_anti_quant_max_quant",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 3), "stride": (2, 2), "pooling_mode": "MAX", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case3 = {
    "params": [{"shape": (1, 4, 16, 16, 16), "dtype": "float16"},
               {"shape": (1, 1, 1, 1, 1), "dtype": "float16"},
               4
               ],
    "case_name": "test_pooling2d_max_strided_write",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 3), "stride": (2, 2), "pooling_mode": "MAX", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case4 = {
    "params": [{"shape": (1, 4, 16, 16, 16), "dtype": "float16"},
               {"shape": (1, 1, 1, 1, 1), "dtype": "float16"},
               4
               ],
    "case_name": "test_pooling2d_gmp_strided_write",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (16, 16), "stride": (2, 2), "pooling_mode": "GMP", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case5 = {
    "params": [{"shape": (4, 8, 28, 28, 32), "dtype": "int8"},
               {"shape": (4, 15, 28, 28, 16), "dtype": "float16", "ori_shape": (4, 280, 28, 28)},
               3
               ],
    "case_name": "test_pooling2d_anti_quant_gmp_quant",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (28, 28), "stride": (2, 2), "pooling_mode": "GMP", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case6 = {
    "params": [{"shape": (2, 4, 16, 16, 16), "dtype": "float16"},
               {"shape": (2, 1, 1, 1, 1), "dtype": "float16"},
               4
               ],
    "case_name": "test_pooling2d_gmp_strided_write1",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (16, 16), "stride": (2, 2), "pooling_mode": "GMP", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case7 = {
    "params": [{"shape": (2, 4, 16, 16, 16), "dtype": "float16"},
               {"shape": (2, 1, 1, 1, 1), "dtype": "float16"},
               4
               ],
    "case_name": "test_pooling2d_gmp_strided_write2",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (16, 16), "stride": (2, 2), "pooling_mode": "GMP", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {"l1_fusion_type":0}, "impl_mode": "high_performance"}
}

case8 = {
    "params": [{"shape": (48, 4, 16, 16, 16), "dtype": "float16"},
               {"shape": (48, 1, 1, 1, 1), "dtype": "float16"},
               4
               ],
    "case_name": "test_pooling2d_gmp_strided_write3",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (16, 16), "stride": (2, 2), "pooling_mode": "GMP", "padding_mode": "SAME",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case9 = {
    "params": [{"shape": (1, 4, 112, 112, 16), "dtype": "float16"},
               {"shape": (1, 4, 56, 56, 1), "dtype": "int8"},
               5
               ],
    "case_name": "test_pooling2d_max_quant_int8",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 3), "stride": (2, 2), "pooling_mode": "MAX", "padding_mode": "VALID",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

case10 = {
    "params": [{"shape": (1, 4, 112, 112, 16), "dtype": "float16"},
               {"shape": (1, 4, 56, 56, 1), "dtype": "int4"},
               5
               ],
    "case_name": "test_pooling2d_max_quant_int4",
    "expect": "success",
    "support_expect": True,
    "addition_params": {"window": (3, 3), "stride": (2, 2), "pooling_mode": "MAX", "padding_mode": "VALID",
                        "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                        "fusion_params": {}, "impl_mode": "high_performance"}
}

compile_case_list = [
    case1,
    case2,
    case3,
    case4,
    case5,
    case6,
    case7,
    case8,
    case9,
    # case10,
]
for item in compile_case_list:
    ut_case.add_case(case=item)


def calc_expect_func(x, y, fuse_type, window, stride, pooling_mode, padding_mode, pad, dilation, data_mode,
                    ceil_mode, fusion_params, impl_mode, kernel_name='dsl_pooling2d_fuse'):
    import numpy as np
    output = x.get("value").reshape(1,8,4,4,2,16).transpose(0,1,4,2,3,5).reshape(1,16,4,4,16).astype("float16")
    # output = x.get("value").astype("float16")
    scale = np.full(output.shape, 0.9551185, dtype="float16")
    output = (output + 128) * scale
    output = np.sum(output, axis=(2, 3), keepdims=1)
    output = output / 16
    return output


ut_case.add_precision_case(
    "Ascend310", {
        "params": [ {"shape": (1,8,4,4,32), "dtype": "int8", "value_range": [1, 10], "param_type": "input"},
                    {"shape": (1,16,1,1,16), "dtype": "float16", "ori_shape": (1, 256, 4, 4),"param_type": "output"},
                    3
                   ],
        "case_name": "test_cast_to_precision_fp322s32_faster_rcnn_resnet",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "addition_params": {"window": (4, 4), "stride": (4, 4), "pooling_mode": "GAP", "padding_mode": "SAME",
                            "pad": (0, 0, 0, 0), "dilation": (1, 1), "data_mode": 0, "ceil_mode": 0,
                            "fusion_params": {}, "impl_mode": "high_performance"}
    })

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
