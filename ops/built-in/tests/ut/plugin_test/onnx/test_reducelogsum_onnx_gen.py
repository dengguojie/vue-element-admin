import onnx
from onnx import helper

def make_reducelogsum():
    node = helper.make_node('ReduceLogSum',
                            ['data'],
                            ['Y'],
                            axes=[-2],)
    graph = helper.make_graph(
                            [node],
                            "test_ReduceLogSum",
                            [helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [3, 4, 5])],
                            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 1, 5])])

    model_def = onnx.helper.make_model(graph, producer_name='wdq')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "test_reducelogsum_case_v11.onnx")

def make_reducelogsum1():
    node = helper.make_node('ReduceLogSum',
                            ['data'],
                            ['Y'],
                            axes=[2, 1],
                            keepdims=0,)
    graph = helper.make_graph(
                            [node],
                            "test_ReduceLogSum",
                            [helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [3, 4, 5])],
                            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3,])])

    model_def = onnx.helper.make_model(graph, producer_name='wdq')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "test_reducelogsum_case1_v11.onnx")

if __name__ == '__main__':
    make_reducelogsum()
    make_reducelogsum1()


