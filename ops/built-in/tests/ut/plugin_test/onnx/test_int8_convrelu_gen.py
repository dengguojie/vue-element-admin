import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_int8_conv_relu():
    strides = [2, 2]
    dilations = [1, 1]
    group = 1
    pads = [1, 1, 1, 1]
    kernels = [3, 3]
    Y_scale = 0.056753046810626984
    Y_zero_point = 0
    Y_scale_1 = 0.00206230441108346
    Y_zero_point_1 = 128
    Y_scale_2 = 0.00000808746881375555
    Y_zero_point_2 = 0
    Y_scale_3 = 0.003921568859368563
    Y_zero_point_3 = 0
    order = "NCHW"
    X1 = helper.make_tensor_value_info("x", TensorProto.INT8, [2, 3, 224, 224])
    X2 = helper.make_tensor_value_info(
        "filter", TensorProto.INT8, [2, 3, 3, 3])
    X3 = helper.make_tensor_value_info("bias", TensorProto.INT32, [2])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT8, [2, 3, 224, 224])
    node_def = helper.make_node('Int8ConvRelu',
                                inputs=['x', 'filter', 'bias'],
                                outputs=['Y'],
                                strides=strides,
                                dilations=dilations,
                                pads=pads,
                                kernels=kernels,
                                group=group,
                                Y_scale_55=Y_scale_1,
                                Y_zero_point_55=Y_zero_point_1,
                                Y_scale_56=Y_scale_2,
                                Y_zero_point_56=Y_zero_point_2,
                                Y_scale_57=Y_scale_3,
                                Y_zero_point_57=Y_zero_point_3,
                                Y_scale=Y_scale,
                                Y_zero_point=Y_zero_point,
                                order=order,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_conv_relu",
        [X1, X2, X3],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx_conv_relu")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_int8_conv_relu_case_1.onnx")


if __name__ == '__main__':
    make_int8_conv_relu()
