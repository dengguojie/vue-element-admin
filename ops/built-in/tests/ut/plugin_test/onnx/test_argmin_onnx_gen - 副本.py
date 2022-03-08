import onnx
import numpy as np
from onnx import helper

def make_argmin():
    data = helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 2])
    Y = helper.make_tensor_value_info("Y", onnx.TensorProto.INT64, [2, 1])
    node = helper.make_node('ArgMin',
                            inputs=['data'],
                            outputs=['Y'],
                            keepdims=1,
                            axis = 1)
    graph = helper.make_graph([node],
                              "test_argmin_1",
                              [data],
                              [Y],)

    model_def = onnx.helper.make_model(graph, producer_name='wdq')
    model_def.opset_import[0].version = 13
    onnx.save(model_def, "./test_argmin_case_V13.onnx")

def export_keepdims_select_last_index():
    data = helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 2])
    Y = helper.make_tensor_value_info("Y", onnx.TensorProto.INT64, [2, 1])
    node = helper.make_node('ArgMin',
                            inputs=['data'],
                            outputs=['Y'],
                            keepdims=1,
                            axis = -1,
                            select_last_index=True)
    graph = helper.make_graph([node],
                              "test_argmin_2",
                              [data],
                              [Y],)
    model_def = onnx.helper.make_model(graph, producer_name='wdq')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_argmin_case_V11.onnx")

if __name__ == "__main__":
    make_argmin()
    export_keepdims_select_last_index()