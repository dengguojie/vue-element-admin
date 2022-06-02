from numpy import ndarray
from tbe.common.context import op_context
from te.platform import cce_conf
from te import tvm


def gen_static_softmaxV2_case(shape, ori_shape, dtype_x, dtype_y, format, ori_format):
    return [{
        "shape": shape,
        "dtype": dtype_x,
        "ori_shape": ori_shape,
        "ori_format": ori_format,
        "format": format
    }, {
        "shape": shape,
        "dtype": dtype_y,
        "ori_shape": ori_shape,
        "ori_format": ori_format,
        "format": format
    }]


def test_softmax_case():
    from impl.softmax_v2 import softmax_v2

    def softmax_v2_compute():
        softmax_v2(*(gen_static_softmaxV2_case((16, 16, 1, 16, 16, 16),
                                               (16, 16, 50, 50), "float16", "float16", "FRACTAL_NZ", "ND")),
                   axis=-2)
        softmax_v2(*(gen_static_softmaxV2_case((16, 16, 16, 16, 16, 16),
                                               (16, 16, 50, 50), "float32", "float32", "NDC1HWC0", "NDC1HWC0")),
                   axis=[2])
        softmax_v2(*(gen_static_softmaxV2_case((1, 16, 32, 32, 16, 16),
                                               (16, 16, 50, 50), "float32", "float32", "NDC1HWC0", "NDC1HWC0")),
                   axis=[2])
        softmax_v2(*(gen_static_softmaxV2_case((1, 1, 32, 16, 16, 16),
                                               (16, 16, 50, 50), "float32", "float32", "NDC1HWC0", "NDC1HWC0")),
                   axis=[2])
        softmax_v2(*(gen_static_softmaxV2_case((1, 1, 16, 1, 16, 16),
                                               (16, 16, 50, 50), "float32", "float32", "NDC1HWC0", "NDC1HWC0")),
                   axis=[2])
        softmax_v2(*(gen_static_softmaxV2_case((16, 16, 16, 16, 16, 16),
                                               (16, 16, 50, 50), "float32", "float32", "NDC1HWC0", "NDC1HWC0")),
                   axis=[2])
        softmax_v2(*(gen_static_softmaxV2_case((16, 16, 16, 16, 16, 16),
                                               (256, 256, 256, 256), "float32", "float32", "NDC1HWC0", "NDC1HWC0")),
                   axis=[2])
        softmax_v2(*(gen_static_softmaxV2_case((16, 16, 16, 16, 16, 16),
                                               (256, 256, 256, 256), "float32", "float32", "NDC1HWC0", "NDC1HWC0")),
                   axis=[2])

        try:
            softmax_v2(*(gen_static_softmaxV2_case((16, 16, 16, 16, 16, 16),
                                                   (256, 256, 257, 256), "float32", "float32", "NDC1HWC0", "NDC1HWC0")),
                       axis=[2])
        except Exception as e:
            print("This is an error scenario!")

        softmax_v2(*(gen_static_softmaxV2_case((16, 16, 16, 16, 16, 16),
                                               (16, 16, 16, 16), "float32", "float32", "NDC1HWC0", "NDC1HWC0")),
                   axis=[2])

        try:
            softmax_v2(*(gen_static_softmaxV2_case((1, 546, 16, 16),
                                                   (8732, 21841), "float16", "float16", "FRACTAL_NZ", "ND")),
                       axis=-1)
        except Exception as e:
            print("This is an error scenario!")

        softmax_v2(*(gen_static_softmaxV2_case((1, 546, 16, 21841),
                                               (8732, 21841), "float16", "float16", "FRACTAL_NZ", "ND")),
                   axis=-1)
        softmax_v2(*(gen_static_softmaxV2_case((1, 546, 16, 21842),
                                               (8732, 21841), "float16", "float16", "FRACTAL_NZ", "ND")),
                   axis=-1)
        softmax_v2(*(gen_static_softmaxV2_case((1, 1, 16, 21842),
                                               (8732, 21841), "float16", "float16", "FRACTAL_NZ", "ND")),
                   axis=-1)
        softmax_v2(*(gen_static_softmaxV2_case((1, 1, 32, 21842),
                                               (8732, 21841), "float16", "float16", "FRACTAL_NZ", "ND")),
                   axis=-1)
        softmax_v2(*(gen_static_softmaxV2_case((1, 546, 16, 21841),
                                               (8732, 21841), "float16", "float16", "FRACTAL_NZ", "ND")),
                   axis=[-1, 2])

    with op_context.OpContext():
        TEST_PLATFORM = ["Ascend910"]
        for soc in TEST_PLATFORM:
            cce_conf.te_set_version(soc)
            softmax_v2_compute()


def test_external_interface():

    def test_get_op_support_info():
        from impl.softmax_v2 import get_op_support_info
        get_op_support_info(*(gen_static_softmaxV2_case((16, 1, 4, 4, 16),
                                                        (16, 4, 4, 4), "float32", "float32", "NC1HWC0", "NCHW")),
                            axis=-4)
        get_op_support_info(*(gen_static_softmaxV2_case((16, 1, 4, 4, 16),
                                                        (16, 4, 4, 4), "float32", "float32", "NC1HWC0", "NCHW")),
                            axis=-3)
        get_op_support_info(*(gen_static_softmaxV2_case((16, 1, 4, 4, 16),
                                                        (16, 4, 4, 4), "float32", "float32", "NC1HWC0", "NCHW")),
                            axis=-2)
        get_op_support_info(*(gen_static_softmaxV2_case((16, 1, 4, 4, 16),
                                                        (16, 4, 4, 4), "float32", "float32", "NC1HWC0", "NCHW")),
                            axis=-1)
        get_op_support_info(*(gen_static_softmaxV2_case((16, 1, 4, 4, 16),
                                                        (16, 4, 4, 4), "float32", "float32", "NC1HWC0", "NHWC")),
                            axis=-4)
        get_op_support_info(*(gen_static_softmaxV2_case((16, 1, 4, 4, 16),
                                                        (16, 4, 4, 4), "float32", "float32", "NC1HWC0", "NHWC")),
                            axis=-3)
        get_op_support_info(*(gen_static_softmaxV2_case((16, 1, 4, 4, 16),
                                                        (16, 4, 4, 4), "float32", "float32", "NC1HWC0", "NHWC")),
                            axis=-2)

        get_op_support_info(*(gen_static_softmaxV2_case((16, 1, 4, 4, 16),
                                                        (16, 4, 4, 4), "float32", "float32", "FRACTAL_NZ", "NHWC")),
                            axis=-4)
        get_op_support_info(*(gen_static_softmaxV2_case((16, 1, 4, 4, 16),
                                                        (16, 4, 4, 4), "float32", "float32", "ND", "NHWC")),
                            axis=-1)

    def test_check_axis_is_last():
        from impl.softmax_v2 import check_axis_is_last
        check_axis_is_last([16, 16, 16], -1)
        check_axis_is_last([16, 16, 16], [2])

    def op_select_format():
        from impl.softmax_v2 import op_select_format
        op_select_format(*(gen_static_softmaxV2_case((1, 2), (1, 2), "float32", "float32", "NC1HWC0", "NHWC")), axis=-2)
        op_select_format(*(gen_static_softmaxV2_case((2, 2), (2, 2), "float32", "float32", "NC1HWC0", "NHWC")), axis=-2)
        op_select_format(*(gen_static_softmaxV2_case((8, 8732, 81),
                                                     (8, 8732, 81), "float32", "float32", "NC1HWC0", "NHWC")),
                         axis=-2)
        op_select_format(*(gen_static_softmaxV2_case((128, 24, 49, 49),
                                                     (128, 24, 50, 50), "float32", "float32", "NC1HWC0", "NHWC")),
                         axis=-2)
        op_select_format(*(gen_static_softmaxV2_case((128, 24, 48, 48),
                                                     (128, 24, 48, 48), "float32", "float32", "NC1HWC0", "NHWC")),
                         axis=-2)

    test_get_op_support_info()
    test_check_axis_is_last()

    with op_context.OpContext():
        TEST_PLATFORM = ["Ascend910", "SD3403", "Ascend710"]
        for soc in TEST_PLATFORM:
            cce_conf.te_set_version(soc)
            op_select_format()


def test_compute_pad_case():

    def test_compute_nopad_fp32():
        from impl.softmax_v2 import compute_nopad_fp32
        compute_nopad_fp32(
            tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="float32"),
            [32, 2, 4, 16, 4],
        )
        compute_nopad_fp32(
            tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="float16"),
            [32, 2, 4, 16, 4],
        )
        compute_nopad_fp32(
            tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="int16"),
            [32, 2, 4, 16, 4],
        )

    def test_compute_nz_nopad_fp32():
        from impl.softmax_v2 import compute_nz_nopad_fp32
        compute_nz_nopad_fp32(
            tvm.placeholder((32, 2, 4, 16), name='tensor_in', dtype="float32"),
            [32, 2, 4, 16],
        )
        compute_nz_nopad_fp32(
            tvm.placeholder((32, 2, 4, 16), name='tensor_in', dtype="float16"),
            [32, 2, 4, 16],
        )
        compute_nz_nopad_fp32(
            tvm.placeholder((32, 2, 4, 16), name='tensor_in', dtype="int16"),
            [32, 2, 4, 16],
        )

    def test_compute_nopad():
        from impl.softmax_v2 import compute_nopad
        compute_nopad(
            tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="float16"),
            [32, 2, 4, 16, 4],
        )

    def test_compute_nz_padding_fp32():
        from impl.softmax_v2 import compute_nz_padding_fp32
        tensor_in = tvm.placeholder((6, 546, 16, 16), name='tensor_in', dtype="float16")
        compute_nz_padding_fp32(tensor_in, (6, 546, 16, 16), [6, 15])
        compute_nz_padding_fp32(tensor_in, (6, 546, 16, 16), [6, 1])

        tensor_in = tvm.placeholder((6, 546, 16, 16), name='tensor_in', dtype="float32")
        compute_nz_padding_fp32(tensor_in, (6, 6, 16, 16), [6, 15])
        compute_nz_padding_fp32(tensor_in, (6, 546, 16, 16), [6, 1])
        compute_nz_padding_fp32(tensor_in, (6, 546, 16, 16), [1, 1])

    def test_compute_padding():
        from impl.softmax_v2 import compute_padding
        compute_padding(
            tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="float16"),
            [32, 2, 4, 16, 4],
            [6, 15],
        )
        compute_padding(
            tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="float16"),
            [32, 2, 4, 16, 4],
            [1, 20],
        )

    test_compute_nopad_fp32()
    test_compute_nz_nopad_fp32()
    test_compute_nopad()
    test_compute_nz_padding_fp32()
    test_compute_padding()


if __name__ == '__main__':
    test_softmax_case()
    test_external_interface()
    test_compute_pad_case()
