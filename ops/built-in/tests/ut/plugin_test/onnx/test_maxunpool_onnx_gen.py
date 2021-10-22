import onnx
import numpy as np
from onnx import helper

def make_maxunpool():
    X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [1, 1, 2, 2])
    I = helper.make_tensor_value_info("I", onnx.TensorProto.INT64, [1, 1, 2, 2])
    output_shape = helper.make_tensor_value_info("output_shape", onnx.TensorProto.INT64, [4])
    Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [1, 1, 5, 5])
    node = helper.make_node('MaxUnpool',
                            inputs=['X', 'I', 'output_shape'],
                            outputs=['Y'],
                            kernel_shape=[2, 2],
                            pads =[0, 0, 0, 0],
                            strides=[2, 2])
    graph = helper.make_graph([node],
                              "test_maxunpool_1",
                              [X, I, output_shape],
                              [Y],)

    model_def = onnx.helper.make_model(graph, producer_name='wdq')
    model_def.opset_import[0].version = 13
    onnx.save(model_def, "./test_maxunpool_case_V13.onnx")

def without_output_shape():
    X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [1, 1, 2, 2])
    I = helper.make_tensor_value_info("I", onnx.TensorProto.INT64, [1, 1, 2, 2])
    Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [1, 1, 4, 4])
    node = helper.make_node('MaxUnpool',
                            inputs=['X', 'I'],
                            outputs=['Y'],
                            kernel_shape=[2, 2],)
    graph = helper.make_graph([node],
                              "test_maxunpool_1",
                              [X, I],
                              [Y],)
    model_def = onnx.helper.make_model(graph, producer_name='wdq')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_maxunpool_case_V11.onnx")

def make_maxunpool_fail():
    X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [1, 1, 2, 2])
    I = helper.make_tensor_value_info("I", onnx.TensorProto.INT64, [1, 1, 2, 2])
    output_shape = helper.make_tensor_value_info("output_shape", onnx.TensorProto.INT64, [4])
    Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [1, 1, 5, 5])
    node = helper.make_node('MaxUnpool',
                            inputs=['X', 'I', 'output_shape'],
                            outputs=['Y'],
                            pads =[0, 0, 0, 0],
                            strides=[2, 2])
    graph = helper.make_graph([node],
                              "test_maxunpool_1",
                              [X, I, output_shape],
                              [Y],)

    model_def = onnx.helper.make_model(graph, producer_name='wdq')
    model_def.opset_import[0].version = 13
    onnx.save(model_def, "./test_maxunpool_case_fail.onnx")

if __name__ == "__main__":
    make_maxunpool()
    without_output_shape()
    make_maxunpool_fail()