import sys

from op_test_frame.ut import OpUT
from impl.dynamic.trans_data import trans_data_fusion_compute
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import tvm
from tbe.common.utils import shape_util

ut_case = OpUT("TransData", None, None)

def test_transdata_fusion_compute(test_arg):
    shape_nchw = [operation.var("n", [1, None]),
                  operation.var("c", [1, None]),
                  operation.var("h", [1, None]),
                  operation.var("w", [1, None]),
                 ]
    src = tvm.placeholder(shape_nchw, name="input0", dtype="float16", attrs={"format": "NCHW", "ori_format": "NCHW"})
    dst = {"dtype": "float16", "shape": [-1, -1, -1, -1, 16], "ori_shape": [-1, -1, -1, -1],
           "format": "NC1HWC0", "ori_format": "NCHW",
           "range": [[1, None], [1, None], [1, None],[1, None], [16, 16]]}
    trans_data_fusion_compute(src, dst)
ut_case.add_cust_test_func(test_func=test_transdata_fusion_compute)

def test_transdata_fusion_compute_1(test_arg):
    shape_nchw = [operation.var("n", [1, None]),
                  operation.var("c", [1, None]),
                  operation.var("h", [1, None]),
                  operation.var("w", [1, None]),
                 ]
    shape_nchw_3d = [shape_nchw[0], shape_nchw[1], shape_nchw[2] * shape_nchw[3]]
    src = tvm.placeholder(shape_nchw_3d, name="input0", dtype="float16", attrs={"format": "NCHW", "ori_format": "NCHW", "shape": shape_nchw})
    dst = {"dtype": "float16", "shape": [-1, -1, -1, -1, 16], "ori_shape": [-1, -1, -1, -1],
           "format": "NC1HWC0", "ori_format": "NCHW",
           "range": [[1, None], [1, None], [1, None],[1, None], [16, 16]]}
    trans_data_fusion_compute(src, dst)
ut_case.add_cust_test_func(test_func=test_transdata_fusion_compute_1)

def test_transdata_fusion_compute_2(test_arg):
    inputs = [
        {"ori_shape": [-1, -1, -1, -1], "shape": [-1, -1, -1, -1], "dtype": "float16", "ori_range": [[1, None], [1, None], [1, None], [1, None]], "format": "NCHW", "ori_format": "NCHW"},
        {"ori_shape": [-1, -1, -1, -1], "shape": [-1, -1, -1, -1], "dtype": "float16", "ori_range": [[1, None], [1, None], [1, None], [1, None]], "format": "NCHW", "ori_format": "NCHW"},
        {"ori_shape": [4], "shape": [4], "dtype": "int32", "ori_range": [[4, 4]], "format": "NCHW", "ori_format": "NCHW"},
    ]
    shape_util.variable_shape(inputs, op_mode="conv2d_backprop_filter")
ut_case.add_cust_test_func(test_func=test_transdata_fusion_compute_2)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend910A"])
    sys.exit(0)



