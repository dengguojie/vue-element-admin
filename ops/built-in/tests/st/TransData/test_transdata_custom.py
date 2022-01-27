import sys

from op_test_frame.ut import OpUT
from impl.dynamic.trans_data import trans_data_fusion_compute
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import tvm

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


if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend910A"])
    sys.exit(0)
