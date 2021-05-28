import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_transpose_case_1():
    strides = [1, 1]
    dilations = [1, 1]
    group = 1
    pads = [0, 0, 0, 0]
    Y_scale = 0.1291135549545288
    Y_zero_point = 142
    Y_scale_1 = 0.06032203510403633
    Y_zero_point_1 = 0
    Y_scale_2 = 0.00838792696595192
    Y_zero_point_2 = 0
    Y_scale_3 = 0.0005059768445789814
    Y_zero_point_3 = 0
    order = "NCHW"
    X1 = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1, 2, 2])
    X2 = helper.make_tensor_value_info("filter", TensorProto.INT8, [1, 1, 2, 2])
    X3 = helper.make_tensor_value_info("bias", TensorProto.INT32, [1])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 1, 1, 1])
    node_def = helper.make_node('Int8Conv',
                                inputs=['x', 'filter', 'bias'],
                                outputs=['Y'],
                                strides=strides,
                                dilations=dilations,
                                pads=pads,
                                groups=group,
                                Y_scale_1=Y_scale_1,
                                Y_zero_point_1=Y_zero_point_1,
                                Y_scale_2=Y_scale_2,
                                Y_zero_point_2=Y_zero_point_2,
                                Y_scale_3=Y_scale_3,
                                Y_zero_point_3=Y_zero_point_3,
                                Y_scale=Y_scale,
                                Y_zero_point=Y_zero_point,
                                order=order,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_conv",
        [X1, X2, X3],
        [Y],
    )

    model = helper.make_model(graph, producer_name="zyw_onnx_conv")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_int8_conv_1.onnx")

def make_int8conv_2():
    strides = [1, 1]
    dilations = [1, 1]
    group = 1
    pads = [0, 0, 0, 0]
    Y_scale = 0.1291135549545288
    Y_zero_point = 142
    Y_scale_1 = 0.06032203510403633
    Y_zero_point_1 = 0
    Y_scale_2 = 0.00838792696595192
    Y_zero_point_2 = 0
    Y_scale_3 = 0.0005059768445789814
    Y_zero_point_3 = 0
    order = "NHWC"
    X1 = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1, 2, 2])
    X2 = helper.make_tensor_value_info("filter", TensorProto.INT8, [1, 1, 2, 2])
    X3 = helper.make_tensor_value_info("bias", TensorProto.INT32, [2])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 1, 1, 1])
    node_def = helper.make_node('Int8Conv',
                                inputs=['x', 'filter', 'bias'],
                                outputs=['Y'],
                                strides=strides,
                                dilations=dilations,
                                pads=pads,
                                groups=group,
                                Y_scale_1=Y_scale_1,
                                Y_zero_point_1=Y_zero_point_1,
                                Y_scale_2=Y_scale_2,
                                Y_zero_point_2=Y_zero_point_2,
                                Y_scale_3=Y_scale_3,
                                Y_zero_point_3=Y_zero_point_3,
                                Y_scale=Y_scale,
                                Y_zero_point=Y_zero_point,
                                order=order,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_conv",
        [X1, X2, X3],
        [Y],
    )

    model = helper.make_model(graph, producer_name="zyw_onnx_conv")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_int8_conv_2.onnx")

def make_transpose_case_2():
    strides = [1, 1]
    dilations = [1, 1]
    group = 1
    pads = [0, 0, 0, 0]
    Y_scale = 0.1291135549545288
    Y_zero_point = 142
    Y_scale_1 = 0.06032203510403633
    Y_zero_point_1 = 0
    Y_scale_2 = 0.00838792696595192
    Y_zero_point_2 = 0
    Y_scale_3 = 0.0005059768445789814
    Y_zero_point_3 = 0
    order = "ND"
    X1 = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1, 2, 2])
    X2 = helper.make_tensor_value_info("filter", TensorProto.INT8, [1, 1, 2, 2])
    X3 = helper.make_tensor_value_info("bias", TensorProto.INT32, [1])
    Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 1, 1, 1])
    node_def = helper.make_node('Int8Conv',
                                inputs=['x', 'filter', 'bias'],
                                outputs=['Y'],
                                strides=strides,
                                dilations=dilations,
                                pads=pads,
                                groups=group,
                                Y_scale_1=Y_scale_1,
                                Y_zero_point_1=Y_zero_point_1,
                                Y_scale_2=Y_scale_2,
                                Y_zero_point_2=Y_zero_point_2,
                                Y_scale_3=Y_scale_3,
                                Y_zero_point_3=Y_zero_point_3,
                                Y_scale=Y_scale,
                                Y_zero_point=Y_zero_point,
                                order=order,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_conv",
        [X1, X2, X3],
        [Y],
    )

    model = helper.make_model(graph, producer_name="zyw_onnx_conv")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_int8_conv_3.onnx")

if __name__ == '__main__':
    make_transpose_case_1()
    make_int8conv_2()
    make_transpose_case_2()
