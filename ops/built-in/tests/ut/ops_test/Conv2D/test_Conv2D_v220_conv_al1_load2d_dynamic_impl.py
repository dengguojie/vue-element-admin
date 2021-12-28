import tbe
from impl.util.platform_adapter import operation
from impl.dynamic.conv2d import conv2d_fusion_compute
from impl.dynamic.leaky_relu import leaky_relu_compute
from tbe.common.context import op_context
from te import platform as cceconf
from te import tvm
from te.utils.cce import auto_schedule
from tbe.dsl.unify_schedule.build import build
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")


def conv_v220_dynamic_fusion_case(dataflow,
                                  conv_type,
                                  in_nd2nz_flag,
                                  out_nz2nd_flag,
                                  shape_in,
                                  shape_w,
                                  dim_range,
                                  pads,
                                  strides,
                                  groups,
                                  bias_flag,
                                  relu_mode=None,
                                  quant_scale=0,
                                  quant_offset=0,
                                  sqrt_mode=False,
                                  cout_real=0):
    if in_nd2nz_flag:
        Ni, Hi, Wi, Ci = shape_in
        h_index, w_index = 1, 2
    else:
        Ni, Ci, Hi, Wi = shape_in
        h_index, w_index = 2, 3

    Co, w_Ci, Hk, Wk = shape_w

    range_in = dim_range
    range_w = [(Co, Co), (w_Ci, w_Ci), (Hk, Hk), (Wk, Wk)]

    Ci0_dict = {
        "float32": 8,
        "float16": 16,
        "int8": 32,
        "bfloat16": 16
    }
    Ci0 = Ci0_dict[conv_type]
    Ci1 = (Ci + Ci0 - 1) // Ci0

    Co0 = 16
    Co1 = (Co + Co0 - 1) // Co0

    if Ni == -1:
        Ni = operation.var("batch_n", range_in[0])
        operation.add_exclude_bound_var(Ni)
    if Hi == -1:
        Hi = operation.var("fmap_h", range_in[h_index])
        operation.add_exclude_bound_var(Hi)
    if Wi == -1:
        Wi = operation.var("fmap_w", range_in[w_index])
        operation.add_exclude_bound_var(Wi)

    shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
    shape_w_fracz = (Hk * Wk * Ci1, Co1, Co0, Ci0)

    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    bias_dtype_dict = {
        "float32": "float32",
        "float16": "float32",
        "bfloat16": "float32",
        "int8": "int32"
    }
    bias_dtype = bias_dtype_dict[conv_type]

    fmap = tvm.placeholder(shape_in_5HD,
                           name='fmap',
                           dtype=conv_type,
                           attrs={
                               'ori_shape': shape_in,
                               'ori_format': 'NCHW',
                               'range': range_in
                           })
    data_format = 'NCHW'

    weight = tvm.placeholder(shape_w_fracz,
                             name='weight',
                             dtype=conv_type,
                             attrs={
                                 'ori_shape': shape_w,
                                 'ori_format': 'NCHW',
                                 'range': range_w
                             })
    bias = tvm.placeholder(
        (Co1 * Co0, ), name='bias', dtype=bias_dtype) if bias_flag else None
    out_shape = [Ni, Co, Hi, Wi]
    res_dtype_dict = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "int8": "int32"
    }
    outputs = {
        "ori_shape": out_shape,
        "ori_format": "NCHW",
        "dtype": res_dtype_dict[conv_type]
    }

    conv_res = conv2d_fusion_compute(fmap,
                                     weight,
                                     bias,
                                     None,
                                     outputs,
                                     strides,
                                     pads,
                                     dilations,
                                     data_format=data_format)

    if dataflow == "conv2d":
        out = conv_res
    if dataflow == "conv2d_relu":
        out = leaky_relu_compute(conv_res, None)

    if dataflow in ("conv2d", "conv2d_relu"):
        tensor_list = [fmap, weight, out]

    if bias_flag:
        tensor_list.insert(2, bias)
    with tvm.target.cce():
        sch = auto_schedule(out)

    config = {
        "name": "conv2d_fusion",
        "tensor_list": tensor_list,
        "build_args": {
            "constant_realize_extent_in_infer_bound": False
        }
    }
    build(sch, config)


def run_testcase(config_dict):
    for i in config_dict:
        print("=" * 150)
        print("case {}".format(i))
        print()

        dataflow = i[0]
        in_nd2nz_flag = False
        out_nz2nd_flag = False
        cout_real = 0

        _, conv_type, shape_in, shape_w, dim_range, pads, strides, groups, bias_flag = i
        with op_context.OpContext("dynamic"):
            with tbe.dsl.base.operation.dynamic():
                with tbe.dsl.base.operation.compute():
                    conv_v220_dynamic_fusion_case(dataflow,
                                                  conv_type,
                                                  in_nd2nz_flag,
                                                  out_nz2nd_flag,
                                                  shape_in,
                                                  shape_w,
                                                  dim_range,
                                                  pads,
                                                  strides,
                                                  groups,
                                                  bias_flag,
                                                  cout_real=cout_real)


def test_conv2d_v220_conv_al1_load2d_dynamic_ut(test_arg):
    print('utmark:al1_load2d_dynamic')
    v220_al1_load2d_case = [
        ("conv2d", "float16", (-1, 24, 28, 28), (64, 24, 1, 1),
            [(1, 2), (24, 24), (28, 28), (28, 28)], (0, 0, 0, 0), (1, 1), 1, False),
        # ("conv2d", "float32", (-1, 24, 28, 28), (64, 24, 1, 1),
        #     [(1, 2), (24, 24), (28, 28), (28, 28)], (0, 0, 0, 0), (1, 1), 1, False),  # float32
        ("conv2d", "float16", (-1, 24, 28, 28), (64, 24, 1, 1),
            [(1, 2), (24, 24), (28, 28), (28, 28)], (0, 0, 0, 0), (1, 1), 3, False),  # group != 1
    ]
    cceconf.te_set_version('Ascend920A')
    with op_context.OpContext():
        run_testcase(v220_al1_load2d_case)


print("====> adding Conv2D v220 conv al1_load2d dynamic ut testcases start")
ut_case.add_cust_test_func("Ascend920A", test_func=test_conv2d_v220_conv_al1_load2d_dynamic_ut)
print("====> adding Conv2D v220 conv al1_load2d dynamic ut testcases end")


if __name__ == '__main__':
    ut_case.run("Ascend920A")
    exit(0)
