import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_fc_case_1():
    X = helper.make_tensor_value_info("x", TensorProto.INT8, [2, 1280])
    W = helper.make_tensor_value_info("filter", TensorProto.INT8, [1000, 1280])
    B = helper.make_tensor_value_info("bias", TensorProto.INT32, [1000])
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [2, 1000])
    node_def = helper.make_node('Int8FC',
                                inputs=['x', 'filter', 'bias'],
                                outputs=['Y'],
                                Y_scale_297=0.0021654849406331778,
                                Y_zero_point_297=0,
                                Y_scale_298=0.0003816144017037004,
                                Y_zero_point_298=128,
                                Y_scale_299=0.80263802442343149e-7,
                                Y_zero_point_299=0,
                                Y_scale=0.004688495770096779,
                                Y_zero_point=127,
                                )
    node_def_2 = helper.make_node('Int8Dequantize',
                                  inputs=['Y'],
                                  outputs=['output'],
                                  Y_scale=0.004688495770096779,
                                  Y_zero_point=127,
                                  )
    graph = helper.make_graph(
        [node_def, node_def_2],
        "test_int8fc",
        [X, W, B],
        [output],
    )

    model = helper.make_model(graph, producer_name="onnx_int8fc")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_int8_fc_case_1.onnx")


if __name__ == '__main__':
    make_fc_case_1()
