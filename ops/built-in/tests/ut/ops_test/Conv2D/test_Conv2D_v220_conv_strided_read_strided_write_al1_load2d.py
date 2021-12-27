import tbe
from impl.conv2d import conv2d_compute
from impl.strided_read import strided_read_compute
from impl.strided_write import strided_write_compute
from tbe.common.context import op_context
from te import platform as cceconf
from te import tvm
from topi import generic
import unittest

# _, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag = i
v220_strided_conv_case = [
    # strided read
    ("conv_sread", "float16", (2, 24, 28, 28), (64, 24, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),  # default
    ("conv_sread", "float32", (2, 24, 28, 28), (64, 24, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),  # float32
    ("conv_sread", "float16", (2, 24, 28, 28), (64, 24, 1, 1), (0, 0, 0, 0), (3, 1), 1, False),  # strideh_opti
    ("conv_sread", "float16", (2, 24, 28, 28), (64, 24, 1, 1), (0, 0, 0, 0), (1, 1), 1, False),  # al1_load2d

    # strided write
    ("conv_swrite", "float16", (2, 24, 28, 28), (64, 24, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
    ("conv_swrite", "float32", (2, 24, 28, 28), (64, 24, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),

    # strided read +conv+ strided write
    ("conv_sread_swrite", "float16", (2, 24, 28, 28), (64, 24, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
]
v220_al1_load2d_case = [
    ("conv2d", "float16", (2, 24, 28, 28), (64, 24, 1, 1), (0, 0, 0, 0), (1, 1), 1, False),
    # ("conv2d", "float32", (2, 24, 28, 28), (64, 24, 1, 1), (0, 0, 0, 0), (1, 1), 1, False),  # float32
    ("conv2d", "float16", (2, 24, 28, 28), (64, 24, 1, 1), (0, 0, 0, 0), (1, 1), 3, False),  # group != 1
]


def conv_v220_fusion_case(dataflow,
                          conv_type,
                          in_nd2nz_flag,
                          weight_nd2nz_flag,
                          out_nz2nd_flag,
                          shape_in,
                          shape_w,
                          pads,
                          strides,
                          groups,
                          bias_flag,
                          relu_mode=None,
                          quant_scale=0,
                          quant_offset=0,
                          sqrt_mode=False,
                          cout_real=0):
    Ni, Ci, Hi, Wi = shape_in

    Co, _, Hk, Wk = shape_w

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

    STRIDE_READ = 4*Ci1
    STRIDE_WRITE = 4*Co1

    with tvm.target.cce():
        if 'sread' in dataflow:
            fmap_ori = tvm.placeholder((Ni, STRIDE_READ*Ci1, Hi, Wi, Ci0),
                                       name='fmap_ori',
                                       dtype=conv_type)
            print("fmap_ori", fmap_ori)
            fmap = strided_read_compute(fmap_ori, {'shape': shape_in_5HD}, 1, STRIDE_READ)
        else:
            fmap = tvm.placeholder(shape_in_5HD, name='fmap', dtype=conv_type)

        weight = tvm.placeholder(shape_w_fracz,
                                 name='weight',
                                 dtype=conv_type,
                                 attrs={
                                     'ori_shape': shape_w,
                                     'ori_format': 'NCHW'
                                 })
        bias = tvm.placeholder(
            (1, Co1, 1, 1, Co0), name='bias', dtype=bias_dtype) if bias_flag else None

        conv_res = conv2d_compute(fmap,
                                  weight,
                                  bias,
                                  None,
                                  None,
                                  strides,
                                  pads,
                                  dilations,
                                  offset_x=0)

        if dataflow == "conv2d" or "sread" in dataflow or "swrite" in dataflow:
            out = conv_res

        if "swrite" in dataflow:
            out = strided_write_compute(out, None, 1, STRIDE_WRITE)

        if dataflow in ("conv2d", "conv2d_relu", "conv2d_quant") or "swrite" in dataflow:
            tensor_list = [fmap, weight, out]

        if "sread" in dataflow:
            tensor_list = [fmap_ori, weight, out]

        if bias_flag:
            tensor_list.insert(2, bias)

        sch = generic.auto_schedule(out)

    config = {
        "print_ir": False,
        "need_build": True,
        "name": "conv2d_fusion",
        "tensor_list": tensor_list
    }
    tbe.dsl.build(sch, config)


def run_testcase(config_dict):
    for i in config_dict:
        print("=" * 150)
        print("case {}".format(i))
        dataflow = i[0]
        in_nd2nz_flag = False
        out_nz2nd_flag = False
        cout_real = 0

        weight_nd2nz_flag = in_nd2nz_flag

        _, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag = i
        conv_v220_fusion_case(dataflow,
                              conv_type,
                              in_nd2nz_flag,
                              weight_nd2nz_flag,
                              out_nz2nd_flag,
                              shape_in,
                              shape_w,
                              pads,
                              strides,
                              groups,
                              bias_flag,
                              cout_real=cout_real)


class TestV220Conv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ut(self):
        cceconf.te_set_version('Ascend920A')
        with op_context.OpContext():
            run_testcase(v220_strided_conv_case)
            run_testcase(v220_al1_load2d_case)


if __name__ == '__main__':
    unittest.main() 
