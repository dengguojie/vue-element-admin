import te
from te import tvm
from op_test_frame.ut import OpUT
import conv2D_ut_testcase as tc
from impl.conv2d_compress import get_op_support_info

ut_case = OpUT("Conv2D", "impl.conv2dcompress", "conv2dcompress")

def gen_kernel_name(input_shape, weights_shape):
    dedy_shape_info = '_'.join([str(i) for i in input_shape])
    w_shape_info = '_'.join([str(i) for i in weights_shape])

    kernel_name = 'conv2d_compress_x_{}_w_{}'.format(
        dedy_shape_info, w_shape_info)
    return kernel_name

def gen_trans_data_case(inputs, weight_compress, compress_index, bias, offset_w, outputs, strides, pads, dilations, expect):
    input_shape = inputs.get("ori_shape")
    weights_shape = weight_compress.get("ori_shape")
    kernel_name = gen_kernel_name(input_shape, weights_shape)
    return {"params": [inputs, weight_compress, compress_index, bias, offset_w, outputs, strides, pads, dilations],
            "case_name": kernel_name,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

def _test_get_op_support_info(test_arg):
    for test_case in tc.op_support_info_conv2d_compress_testcase:
        formatted_case = gen_trans_data_case(*test_case[1:])
        params = formatted_case["params"]
        get_op_support_info(*params)

print("adding conv2d compress split info testcase")
ut_case.add_cust_test_func(test_func=_test_get_op_support_info)

if __name__ == '__main__':
    ut_case.run(["Ascend910", "Ascend310"])
    exit(0)