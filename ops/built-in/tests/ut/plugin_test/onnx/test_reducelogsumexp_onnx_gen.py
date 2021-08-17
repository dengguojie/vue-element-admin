import onnx
from onnx import helper
import numpy as np

def make_reducelogsumexp():
    node = helper.make_node('ReduceLogSumExp',
                            ['data'],
                            ['Y'],
                            axes=[1],
                            keepdims = 0,)
    graph = helper.make_graph(
                            [node],
                            "test_ReduceLogSumExp",
                            [helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT16, [3, 2, 2, 2, 2])],
                            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [1, 1, 1, 1, 1])])

    model_def = onnx.helper.make_model(graph, producer_name='wdq')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "test_reducelogsumexp_case_v11.onnx")

def make_reducelogsumexp1():
    node = helper.make_node('ReduceLogSumExp',
                            ['data'],
                            ['Y'],)
    graph = helper.make_graph(
                            [node],
                            "test_ReduceLogSumExp",
                            [helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [3, 2, 2])],
                            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 2])])

    model_def = onnx.helper.make_model(graph, producer_name='wdq')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "test_reducelogsumexp_case1_v11.onnx")

if __name__ == '__main__':
    make_reducelogsumexp()
    make_reducelogsumexp1()
